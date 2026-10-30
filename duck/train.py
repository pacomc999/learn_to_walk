"""Teach the duck to walk with PPO, saving snapshots and progress along the way.

New runs use the left/right symmetric brain. Resumed runs keep whatever brain
their snapshot holds, so older plain runs still continue as they were.

Run from the project root, for example:
    python -m duck.train --run runs/duck_v1 --steps 30000000
Resume a stopped run, optionally with more curiosity:
    python -m duck.train --run runs/duck_v1 --steps 30000000 --resume --ent-coef 0.01
Start a new run from another run's brain (a run folder, its best/ folder or a duck's brain/ folder):
    python -m duck.train --run runs/duck_03/a --steps 30000000 --start-from duck_catalogue/duck_02_1/brain
"""
import argparse
import ctypes
import functools
import socket
import subprocess
import sys
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

from duck.dashboard import DEFAULT_PORT
from duck.bodies import Body
from duck.duck_env import OBSERVATION_SIZE, DuckWalkEnv
from duck.worlds import World
from duck.exam import append_exam, best_exam_score, best_folder, parts_in_words, take_full_exam
from duck.mirror import MirroredRunningMeanStd
from duck.progress import append_progress
from duck.run_settings import RunSettings, duck_settings, read_run_settings, write_run_settings
from duck.snapshots import (Snapshot, latest_snapshot, list_snapshots, remove_other_snapshots, save_snapshot,
                            set_aside_broken_snapshot, snapshot_paths)
from duck.symmetric_policy import SymmetricPolicy
from duck.training_stats import append_stats, latest_update_stats

# Brain shape, used by new runs with the symmetric brain:
# the actor (pi) and the critic (vf) each get 2 hidden layers of 256
NET_ARCH = {"pi": [256, 256], "vf": [256, 256]}

# The brain class every new run starts with (resumed runs keep the class they were saved with)
NEW_RUN_POLICY = SymmetricPolicy

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

# The adaptive step size (Rudin and others 2021, the legged_gym / rsl_rl defaults).
# After each brain update: if the brain changed more than twice the target change
# (approx_kl), the learning rate is divided by the factor; if less than half the target,
# multiplied by it. The learning rate stays between the two limits
STEP_SIZE_RULE = {
    "target_change": 0.01,
    "factor": 1.5,
    "min_learning_rate": 1e-5,
    "max_learning_rate": 1e-2,
}

# Steps between noise-free exams (each takes about 15 seconds)
DEFAULT_EXAM_EVERY = 250_000

# Training stops by itself after this many exams in a row without a new best score
# (8 exams of 250,000 steps: about 35 minutes). 0 means never stop early
DEFAULT_PATIENCE = 8

# Shown when training stops early
NO_IMPROVEMENT_MESSAGE = ("No new best exam score for {exams} exams in a row. Stopping early; "
                          "the best brain is in {best_dir}.")

# Threads for the brain update in the main process, leaving cores for the env workers
MAIN_PROCESS_THREADS = 2

# Where the dashboard started by training writes its messages
DASHBOARD_LOG = Path("runs/dashboard.log")

# Windows flags that keep the laptop from sleeping while training runs
KEEP_AWAKE_FLAGS = 0x80000000 | 0x00000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
NORMAL_SLEEP_FLAGS = 0x80000000  # ES_CONTINUOUS alone restores normal sleep

# Shown when a new run is started in a folder that already holds a run
USED_RUN_FOLDER_MESSAGE = (
    "This run folder already has snapshots. Use --resume to continue it, or pick a new run folder."
)


# One duck (a body) in a world, with a Monitor that records each attempt's score, length and forward
# distance. It lives at module level so Windows can start it in a new process (spawn)
def make_duck_env(world: World, body: Body) -> Monitor:
    return Monitor(DuckWalkEnv(world=world, body=body), info_keywords=("forward_distance",))


# Several duck worlds side by side, each in its own process or all in this one
def _build_raw_vec_env(n_envs: int, seed: int, use_subprocess: bool, world: World, body: Body) -> VecEnv:
    env_factories = [functools.partial(make_duck_env, world, body)] * n_envs
    if use_subprocess:
        venv = SubprocVecEnv(env_factories, start_method="spawn")
    else:
        venv = DummyVecEnv(env_factories)
    venv.seed(seed)  # each world gets seed, seed + 1, seed + 2, ...
    return venv


# The duck worlds wrapped so observations and rewards are scaled to friendly sizes
def build_vec_env(n_envs: int, seed: int, world: World, body: Body, use_subprocess: bool = True) -> VecNormalize:
    venv = _build_raw_vec_env(n_envs, seed, use_subprocess, world, body)
    vec_normalize = VecNormalize(venv, norm_obs=True, norm_reward=True)
    # Scale mirrored senses together so left and right keep the same scaling
    vec_normalize.obs_rms = MirroredRunningMeanStd(shape=(OBSERVATION_SIZE,))
    return vec_normalize


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


# The learning rate for the next update, given how much the brain changed in the last one
def adjusted_learning_rate(learning_rate: float, change: float, rule: dict = STEP_SIZE_RULE) -> float:
    if change > 2 * rule["target_change"]:
        learning_rate /= rule["factor"]  # the brain moved too far: smaller steps
    elif change < rule["target_change"] / 2:
        learning_rate *= rule["factor"]  # the brain barely moved: bigger steps
    return min(max(learning_rate, rule["min_learning_rate"]), rule["max_learning_rate"])


# Adapts the learning rate after every brain update so each update changes the brain by about
# the target amount. The new value is stored on the model, so snapshots keep it when resumed
class AdaptiveStepSizeCallback(BaseCallback):
    # Nothing to do per step
    def _on_step(self) -> bool:
        return True

    # Right after an update, its change size is in the logger; set the learning rate for the next one
    def _on_rollout_start(self) -> None:
        change = self.model.logger.name_to_value.get("train/approx_kl")
        if change is None:
            return  # no update yet
        self.model.learning_rate = adjusted_learning_rate(float(self.model.lr_schedule(1.0)), float(change))
        self.model._setup_lr_schedule()


# Records PPO's own numbers (step size, judge quality, wobble) after each brain update.
# A rollout starts right after an update, so its numbers are still in the logger here
class TrainingStatsCallback(BaseCallback):
    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = run_dir

    # Nothing to do per step
    def _on_step(self) -> bool:
        return True

    # Log the numbers of the update that just finished, if there was one
    def _on_rollout_start(self) -> None:
        self._log_latest_update()

    # The last update of a session has no rollout after it, so it is logged here
    def _on_training_end(self) -> None:
        self._log_latest_update()

    # Write the logger's numbers of the latest update to the stats log
    def _log_latest_update(self) -> None:
        stats = latest_update_stats(self.model.logger.name_to_value)
        if stats is not None:
            append_stats(self.run_dir, self.num_timesteps, stats, datetime.now())


# Gives the brain the noise-free exam at the start, after every exam_every steps and at the end.
# Each result goes to the exam log, and a brain that beats the best score so far is kept in <run>/best/.
# After `patience` exams in a row without a new best, training stops (patience 0: never)
class ExamCallback(BaseCallback):
    def __init__(self, run_dir: Path, vec_normalize: VecNormalize, exam_every: int, world: World, body: Body,
                 patience: int = DEFAULT_PATIENCE):
        super().__init__()
        self.run_dir = run_dir
        self.world = world
        self.body = body
        self.vec_normalize = vec_normalize
        self.exam_every = exam_every
        self.patience = patience
        self.next_exam_steps = 0
        self.best_score = None
        self.last_exam_steps = None
        self.exams_without_best = 0

    # Remember the best score of earlier sessions of this run, then examine the starting brain
    def _on_training_start(self) -> None:
        self.best_score = best_exam_score(self.run_dir)
        self._examine()

    # Returning False tells PPO to stop learning
    def _on_step(self) -> bool:
        return not self._out_of_patience()

    # True once too many exams in a row brought no new best score
    def _out_of_patience(self) -> bool:
        return self.patience > 0 and self.exams_without_best >= self.patience

    # Right after a brain update, examine it if the next multiple of exam_every was passed
    def _on_rollout_start(self) -> None:
        if self.num_timesteps >= self.next_exam_steps:
            self._examine()

    # The final exam when learning stops
    def _on_training_end(self) -> None:
        self._examine()

    # Take the exam once per step count, log it, and keep the brain if it is the best so far
    def _examine(self) -> None:
        steps = self.num_timesteps
        self.next_exam_steps = (steps // self.exam_every + 1) * self.exam_every
        if steps == self.last_exam_steps:
            return
        self.last_exam_steps = steps
        result = take_full_exam(self.model, self.vec_normalize, self.world, self.body)
        append_exam(self.run_dir, steps, result, datetime.now())
        message = (f"Exam at {steps:,} steps: score {result.score:.0f}, falls {result.falls} of {result.attempts} "
                   f"({parts_in_words(result)}).")
        if self.best_score is None or result.score > self.best_score:
            self.best_score = result.score
            self.exams_without_best = 0
            _save_snapshot_once(self.model, self.vec_normalize, best_folder(self.run_dir), steps)
            remove_other_snapshots(best_folder(self.run_dir), steps)  # only the newest best is ever used
            message += " New best brain saved."
        else:
            self.exams_without_best += 1
        print(message, flush=True)
        if self._out_of_patience():
            print(NO_IMPROVEMENT_MESSAGE.format(exams=self.exams_without_best,
                                                best_dir=best_folder(self.run_dir)), flush=True)


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
def _new_model(run_dir: Path, n_envs: int, use_subprocess: bool, ent_coef: float | None,
               ppo_overrides: dict | None, world: World, body: Body) -> tuple[PPO, VecNormalize]:
    _refuse_used_run_folder(run_dir)
    vec_normalize = build_vec_env(n_envs, seed=0, world=world, body=body, use_subprocess=use_subprocess)
    settings = {**DEFAULT_PPO_SETTINGS, **(ppo_overrides or {})}
    if ent_coef is not None:
        settings["ent_coef"] = ent_coef
    model = PPO(NEW_RUN_POLICY, vec_normalize, device="cpu", seed=0, verbose=0,
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
def load_latest_snapshot(run_dir: Path, n_envs: int, use_subprocess: bool, world: World,
                         body: Body) -> tuple[PPO, VecNormalize]:
    snapshots = list_snapshots(run_dir)
    if not snapshots:
        raise FileNotFoundError(f"No complete snapshot to resume from in {run_dir}")
    # A new seed per resume so the worlds do not replay the same starts
    venv = _build_raw_vec_env(n_envs, seed=snapshots[-1].steps, use_subprocess=use_subprocess,
                              world=world, body=body)
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
def _resumed_model(run_dir: Path, n_envs: int, use_subprocess: bool, ent_coef: float | None,
                   world: World, body: Body) -> tuple[PPO, VecNormalize]:
    model, vec_normalize = load_latest_snapshot(run_dir, n_envs, use_subprocess, world, body)
    if ent_coef is not None:
        model.ent_coef = ent_coef  # more curiosity from here on
    return model, vec_normalize


# Stop if --start-from and --resume are both given, since they ask for two different starts
def _refuse_start_from_with_resume(start_from: Path | None, resume: bool) -> None:
    if start_from is not None and resume:
        raise ValueError("Use either --start-from (a new run) or --resume (continue a run), not both.")


# A new run that begins with the newest complete brain in another folder, counting steps from 0.
# The source folder is only read. The started brain is saved as this run's step 0 snapshot
def _started_model(run_dir: Path, start_from: Path, n_envs: int, use_subprocess: bool,
                   ent_coef: float | None, world: World, body: Body) -> tuple[PPO, VecNormalize]:
    _refuse_used_run_folder(run_dir)
    source = latest_snapshot(start_from)
    if source is None:
        raise FileNotFoundError(f"No complete snapshot to start from in {start_from}")
    venv = _build_raw_vec_env(n_envs, seed=0, use_subprocess=use_subprocess, world=world, body=body)
    model, vec_normalize = _load_snapshot(source, venv)
    model.num_timesteps = 0  # the new run counts from 0
    if ent_coef is not None:
        model.ent_coef = ent_coef
    _save_snapshot_once(model, vec_normalize, run_dir, steps=0)
    print(f"Starting from the brain at {source.steps:,} steps in {start_from}.", flush=True)
    return model, vec_normalize


# The world and body a run trains in: a resumed run's own settings file comes first, then the settings
# given (written next to the run, so it keeps them). With neither, stop with a plain message
def _run_settings(run_dir: Path, settings: RunSettings | None, resume: bool) -> RunSettings:
    own = read_run_settings(run_dir) if resume else None
    if own is not None:
        return own
    if settings is None:
        raise ValueError("This run has no world and body yet. Start a new run with --duck DUCK_FOLDER.")
    write_run_settings(run_dir, settings)
    return settings


# Train a new run or resume one until total_steps in total, saving snapshots and progress.
# settings gives the world and body of a new run; a resumed run uses its own run_settings.json
def train(run_dir: Path, total_steps: int, settings: RunSettings | None = None, snapshot_every: int = 1_000_000,
          n_envs: int = 8, ent_coef: float | None = None, resume: bool = False, use_subprocess: bool = True,
          ppo_overrides: dict | None = None, exam_every: int = DEFAULT_EXAM_EVERY,
          patience: int = DEFAULT_PATIENCE, start_from: Path | None = None) -> PPO:
    run_dir = Path(run_dir)
    _refuse_start_from_with_resume(start_from, resume)
    if not resume:
        _refuse_used_run_folder(run_dir)  # before writing anything into the folder
    settings = _run_settings(run_dir, settings, resume)
    world, body = settings.world, settings.body
    if start_from is not None:
        model, vec_normalize = _started_model(run_dir, Path(start_from), n_envs, use_subprocess, ent_coef,
                                              world, body)
    elif resume:
        model, vec_normalize = _resumed_model(run_dir, n_envs, use_subprocess, ent_coef, world, body)
    else:
        model, vec_normalize = _new_model(run_dir, n_envs, use_subprocess, ent_coef, ppo_overrides, world, body)

    try:
        steps_left = total_steps - model.num_timesteps
        if steps_left <= 0:
            return model  # nothing left to learn (for example total_steps=0)
        # The stats are logged before the step size changes, so each row shows the rate its update used
        callbacks = [SnapshotCallback(run_dir, vec_normalize, snapshot_every), ProgressCallback(run_dir),
                     TrainingStatsCallback(run_dir), AdaptiveStepSizeCallback(),
                     ExamCallback(run_dir, vec_normalize, exam_every, world, body, patience)]
        with keep_awake():
            model.learn(total_timesteps=steps_left, callback=callbacks, reset_num_timesteps=False)
        return model
    finally:
        vec_normalize.close()  # stop the worker processes


# ---------- The live dashboard ----------

# Whether something already answers on the dashboard's port (an earlier dashboard still running)
def dashboard_running(port: int = DEFAULT_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


# Start the dashboard for this run as its own process, so it stays open after training ends.
# Its messages go to runs/dashboard.log. An already running dashboard is left alone
def start_dashboard(run_dir: Path, total_steps: int) -> None:
    address = f"http://127.0.0.1:{DEFAULT_PORT}/"
    if dashboard_running():
        print(f"A dashboard is already running at {address} (it may be showing another run).", flush=True)
        return
    DASHBOARD_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = DASHBOARD_LOG.open("a")
    subprocess.Popen([sys.executable, "-m", "duck.dashboard", str(run_dir), "--steps", str(total_steps)],
                     stdout=log_file, stderr=subprocess.STDOUT)
    print(f"Dashboard starting at {address}", flush=True)


# Read the command line options and start training
def main() -> None:
    parser = argparse.ArgumentParser(description="Teach the duck to walk with PPO.")
    parser.add_argument("--duck", type=Path, default=None,
                        help="the duck this new run belongs to, for its world and body, for example duck_catalogue/duck_04")
    parser.add_argument("--run", type=Path, required=True, help="run folder, for example runs/duck_04/from_zero")
    parser.add_argument("--steps", type=int, required=True, help="total training steps for the whole run")
    parser.add_argument("--resume", action="store_true", help="continue from the newest snapshot")
    parser.add_argument("--start-from", type=Path, default=None,
                        help="start a new run from the newest brain in this folder (run, best/ or brain/)")
    parser.add_argument("--ent-coef", type=float, default=None, help="curiosity, for example 0.01")
    parser.add_argument("--envs", type=int, default=8, help="duck worlds running side by side")
    parser.add_argument("--snapshot-every", type=int, default=1_000_000, help="steps between snapshots")
    parser.add_argument("--exam-every", type=int, default=DEFAULT_EXAM_EVERY, help="steps between noise-free exams")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="stop after this many exams without a new best (0: never)")
    parser.add_argument("--no-dashboard", action="store_true", help="do not open the live dashboard")
    args = parser.parse_args()

    # A new run needs its duck: check that before anything starts, the dashboard included
    try:
        if args.duck is None and not args.resume:
            raise ValueError("A new run needs --duck DUCK_FOLDER, so it knows its world and body.")
        settings = duck_settings(args.duck) if args.duck else None
    except ValueError as error:
        print(error)  # a plain message, no traceback
        raise SystemExit(1)
    if not args.no_dashboard:
        start_dashboard(args.run, args.steps)
    torch.set_num_threads(MAIN_PROCESS_THREADS)
    try:
        train(args.run, args.steps, settings=settings, snapshot_every=args.snapshot_every, n_envs=args.envs,
              ent_coef=args.ent_coef, resume=args.resume, exam_every=args.exam_every,
              patience=args.patience, start_from=args.start_from)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(error)  # a plain message, no traceback
        raise SystemExit(1)
    print("Training finished.")


if __name__ == "__main__":
    main()
