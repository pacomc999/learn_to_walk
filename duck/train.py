"""Teach the duck to walk with PPO, saving snapshots and progress along the way.

Run from the project root, for example:
    python -m duck.train --run runs/duck_v1 --steps 30000000
Resume a stopped run, optionally with more curiosity:
    python -m duck.train --run runs/duck_v1 --steps 30000000 --resume --ent-coef 0.01
"""
import argparse
import ctypes
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from duck.duck_env import DuckWalkEnv
from duck.progress import append_progress
from duck.snapshots import (Snapshot, latest_snapshot, list_snapshots, save_snapshot,
                            set_aside_broken_snapshot, snapshot_paths)

# Brain shape: the actor (pi) and the critic (vf) each get 2 hidden layers of 256
NET_ARCH = {"pi": [256, 256], "vf": [256, 256]}

# A new brain starts with calmer random moves: log_std -1 is a spread of about 0.37
# per motor, instead of 1.0, so early attempts are not pure flailing
START_LOG_STD = -1.0

# PPO settings used unless a caller overrides them (8 envs x 1024 steps = 8192 steps per rollout)
DEFAULT_PPO_SETTINGS = {
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}

# Threads for the brain update in the main process, leaving cores for the env workers
MAIN_PROCESS_THREADS = 2

# Windows flags that keep the laptop from sleeping while training runs
KEEP_AWAKE_FLAGS = 0x80000000 | 0x00000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
NORMAL_SLEEP_FLAGS = 0x80000000  # ES_CONTINUOUS alone restores normal sleep

# Shown when a new run is started in a folder that already holds a run
USED_RUN_FOLDER_MESSAGE = (
    "This run folder already has snapshots. Use --resume to continue it, or pick a new run folder."
)


# One duck world with a Monitor that records each attempt's score, length and forward distance.
# It lives at module level so Windows can start it in a new process (spawn).
def make_duck_env() -> Monitor:
    return Monitor(DuckWalkEnv(), info_keywords=("forward_distance",))


# Several duck worlds side by side, each in its own process or all in this one
def _build_raw_vec_env(n_envs: int, seed: int, use_subprocess: bool) -> VecEnv:
    env_factories = [make_duck_env] * n_envs
    if use_subprocess:
        venv = SubprocVecEnv(env_factories, start_method="spawn")
    else:
        venv = DummyVecEnv(env_factories)
    venv.seed(seed)  # each world gets seed, seed + 1, seed + 2, ...
    return venv


# The duck worlds wrapped so observations and rewards are scaled to friendly sizes
def build_vec_env(n_envs: int, seed: int, use_subprocess: bool = True) -> VecNormalize:
    venv = _build_raw_vec_env(n_envs, seed, use_subprocess)
    return VecNormalize(venv, norm_obs=True, norm_reward=True)


# Save a snapshot unless one with this step count is already on disk.
# Overwriting could pair a new brain with an old statistics file after a crash.
def _save_snapshot_once(model: PPO, vec_normalize: VecNormalize, run_dir: Path, steps: int) -> bool:
    existing = snapshot_paths(run_dir, steps)
    if existing.model_path.exists() and existing.norm_path.exists():
        return False
    save_snapshot(model, vec_normalize, run_dir, steps)
    return True


# Saves a snapshot at the first rollout boundary past each multiple of snapshot_every, plus at the end
class SnapshotCallback(BaseCallback):
    def __init__(self, run_dir: Path, vec_normalize: VecNormalize, snapshot_every: int):
        super().__init__()
        self.run_dir = run_dir
        self.vec_normalize = vec_normalize
        self.snapshot_every = snapshot_every
        self.next_snapshot_steps = 0
        self.last_time = 0.0
        self.last_steps = 0

    # Work out the next multiple to reach and start the speed clock
    def _on_training_start(self) -> None:
        steps = self.num_timesteps
        self.next_snapshot_steps = (steps // self.snapshot_every + 1) * self.snapshot_every
        self.last_time = time.perf_counter()
        self.last_steps = steps

    # Nothing to do per step; snapshots are checked between rollouts
    def _on_step(self) -> bool:
        return True

    # A rollout starts right after the previous brain update, so a snapshot
    # saved here holds everything learned from the steps it is named after
    def _on_rollout_start(self) -> None:
        if self.num_timesteps >= self.next_snapshot_steps:
            self._save_and_report()
            # Skip ahead past every multiple already passed
            self.next_snapshot_steps = (self.num_timesteps // self.snapshot_every + 1) * self.snapshot_every

    # The final snapshot, saved when learning stops
    def _on_training_end(self) -> None:
        self._save_and_report()

    # Save a snapshot and print how fast training ran since the previous one
    def _save_and_report(self) -> None:
        steps = self.num_timesteps
        if not _save_snapshot_once(self.model, self.vec_normalize, self.run_dir, steps):
            return
        now = time.perf_counter()
        seconds = max(now - self.last_time, 1e-9)
        speed = (steps - self.last_steps) / seconds
        print(f"Snapshot saved at {steps:,} steps. Speed: {speed:,.0f} steps per second.", flush=True)
        self.last_time = now
        self.last_steps = steps


# Writes the average score and distance of recent attempts to the progress log after each rollout
class ProgressCallback(BaseCallback):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir

    # Nothing to do per step; the work happens at the end of each rollout
    def _on_step(self) -> bool:
        return True

    # Log the mean score and distance of the finished attempts the model remembers, if there are any
    def _on_rollout_end(self) -> None:
        episodes = self.model.ep_info_buffer
        if not episodes:
            return
        mean_reward = float(np.mean([episode["r"] for episode in episodes]))
        mean_distance = float(np.mean([episode["forward_distance"] for episode in episodes]))
        append_progress(self.run_dir, self.num_timesteps, mean_reward, mean_distance, datetime.now())


# Keep Windows awake inside this block, and always restore normal sleep afterwards
@contextmanager
def keep_awake():
    is_windows = hasattr(ctypes, "windll")
    if is_windows:
        ctypes.windll.kernel32.SetThreadExecutionState(KEEP_AWAKE_FLAGS)
    try:
        yield
    finally:
        if is_windows:
            ctypes.windll.kernel32.SetThreadExecutionState(NORMAL_SLEEP_FLAGS)


# Stop before a new run mixes its snapshots and progress log with an older run's
def _refuse_used_run_folder(run_dir: Path) -> None:
    if latest_snapshot(run_dir) is not None:
        raise FileExistsError(USED_RUN_FOLDER_MESSAGE)


# A brand new untrained brain, saved as the step 0 snapshot
def _new_model(run_dir: Path, n_envs: int, use_subprocess: bool,
               ent_coef: float | None, ppo_overrides: dict | None) -> tuple[PPO, VecNormalize]:
    _refuse_used_run_folder(run_dir)
    vec_normalize = build_vec_env(n_envs, seed=0, use_subprocess=use_subprocess)
    settings = {**DEFAULT_PPO_SETTINGS, **(ppo_overrides or {})}
    if ent_coef is not None:
        settings["ent_coef"] = ent_coef
    model = PPO("MlpPolicy", vec_normalize, device="cpu", seed=0, verbose=0,
                policy_kwargs={"net_arch": NET_ARCH, "log_std_init": START_LOG_STD}, **settings)
    if _save_snapshot_once(model, vec_normalize, run_dir, steps=0):
        print("Untrained snapshot saved at 0 steps.", flush=True)
    return model, vec_normalize


# The brain and statistics of one snapshot, attached to the given duck worlds
def _load_snapshot(snap: Snapshot, venv: VecEnv) -> tuple[PPO, VecNormalize]:
    vec_normalize = VecNormalize.load(str(snap.norm_path), venv)
    vec_normalize.training = True  # keep updating the statistics while learning
    model = PPO.load(str(snap.model_path), env=vec_normalize, device="cpu")
    return model, vec_normalize


# The brain and statistics from the newest snapshot that loads, ready to keep learning.
# A broken snapshot (for example a damaged file) is set aside with a warning,
# and the one before it is tried instead
def load_latest_snapshot(run_dir: Path, n_envs: int, use_subprocess: bool) -> tuple[PPO, VecNormalize]:
    snapshots = list_snapshots(run_dir)
    if not snapshots:
        raise FileNotFoundError(f"No complete snapshot to resume from in {run_dir}")
    # A new seed per resume so the worlds do not replay the same starts
    venv = _build_raw_vec_env(n_envs, seed=snapshots[-1].steps, use_subprocess=use_subprocess)
    for snap in reversed(snapshots):
        try:
            model, vec_normalize = _load_snapshot(snap, venv)
        except Exception:
            print(f"Warning: could not load the snapshot {snap.model_path.name}. "
                  "Setting it aside and trying the one before it.", flush=True)
            set_aside_broken_snapshot(snap)
            continue
        print(f"Resuming from the snapshot at {snap.steps:,} steps.", flush=True)
        return model, vec_normalize
    venv.close()
    raise FileNotFoundError(f"No snapshot in {run_dir} could be loaded")


# The newest snapshot's brain, with a new curiosity value if one is given
def _resumed_model(run_dir: Path, n_envs: int, use_subprocess: bool,
                   ent_coef: float | None) -> tuple[PPO, VecNormalize]:
    model, vec_normalize = load_latest_snapshot(run_dir, n_envs, use_subprocess)
    if ent_coef is not None:
        model.ent_coef = ent_coef  # more curiosity from here on
    return model, vec_normalize


# Train a new run or resume one until total_steps in total, saving snapshots and progress
def train(run_dir: Path, total_steps: int, snapshot_every: int = 1_000_000, n_envs: int = 8,
          ent_coef: float | None = None, resume: bool = False, use_subprocess: bool = True,
          ppo_overrides: dict | None = None) -> PPO:
    run_dir = Path(run_dir)
    if resume:
        model, vec_normalize = _resumed_model(run_dir, n_envs, use_subprocess, ent_coef)
    else:
        model, vec_normalize = _new_model(run_dir, n_envs, use_subprocess, ent_coef, ppo_overrides)

    try:
        steps_left = total_steps - model.num_timesteps
        if steps_left <= 0:
            return model  # nothing left to learn (for example total_steps=0)
        callbacks = [SnapshotCallback(run_dir, vec_normalize, snapshot_every), ProgressCallback(run_dir)]
        with keep_awake():
            model.learn(total_timesteps=steps_left, callback=callbacks, reset_num_timesteps=False)
        return model
    finally:
        vec_normalize.close()  # stop the worker processes


# Read the command line options and start training
def main() -> None:
    parser = argparse.ArgumentParser(description="Teach the duck to walk with PPO.")
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/duck_v1")
    parser.add_argument("--steps", type=int, required=True, help="total training steps for the whole run")
    parser.add_argument("--resume", action="store_true", help="continue from the newest snapshot")
    parser.add_argument("--ent-coef", type=float, default=None, help="curiosity, for example 0.01")
    parser.add_argument("--envs", type=int, default=8, help="duck worlds running side by side")
    parser.add_argument("--snapshot-every", type=int, default=1_000_000, help="steps between snapshots")
    args = parser.parse_args()

    torch.set_num_threads(MAIN_PROCESS_THREADS)
    try:
        train(args.run, args.steps, snapshot_every=args.snapshot_every, n_envs=args.envs,
              ent_coef=args.ent_coef, resume=args.resume)
    except FileExistsError as error:
        print(error)  # a plain message, no traceback
        raise SystemExit(1)
    print("Training finished.")


if __name__ == "__main__":
    main()
