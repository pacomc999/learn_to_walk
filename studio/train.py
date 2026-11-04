"""A training session: the walker practises, takes exams and keeps its best brain.

The studio's server starts this as its own process for one training folder:
    python -m studio.train library/trainings/<id>          (a new session)
    python -m studio.train library/trainings/<id> --resume (carry on after a stop)
Everything the page shows comes from the files it writes in that folder:
    status.json      what it is doing right now
    progress.csv     the practice scores after every lesson
    exams.csv        the exam results
    brain/           the latest and the best brain
    replays/         3D recordings: the best walk so far and one from every exam
    log.txt          everything it printed
Creating a file called stop.request in the folder makes it save and stop at the next chance.
"""
import argparse
import csv
import functools
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# Each practice world runs in its own process; one thread each keeps them from fighting for the cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: E402

from studio.brain import clean_design, design_for_body, design_senses, policy_for  # noqa: E402
from studio.creatures import build_creature  # noqa: E402
from studio.env import PART_NAMES, WalkEnv, observation_size  # noqa: E402
from studio.learning import clean_learning, ppo_settings  # noqa: E402
from studio.mirror import MirroredRunningMeanStd, mirror_tables  # noqa: E402
from studio.replay import record, save_scaling  # noqa: E402
from studio.scene import build_scene  # noqa: E402
from studio.worlds import world_from_dict  # noqa: E402

# The exam: the same attempts every time, without random wobble
EXAM_ATTEMPTS = 8
EXAM_FIRST_SEED = 1000

# Training gets harder step by step over this share of the session (for gradually harder worlds)
RAMP_SHARE = 0.5

# The adaptive learning speed (learn_to_walk's rule: Rudin and others 2021)
STEP_SIZE_RULE = {"target_change": 0.01, "factor": 1.5, "min": 1e-5, "max": 1e-2}

# How many exam replays to keep; older ones are thinned out evenly
MOST_EXAM_REPLAYS = 30

PROGRESS_HEADER = ["time", "steps", "score", "distance", "seconds", "fall_share", "steps_per_second", "difficulty"] + [
    f"pts_{name}" for name in PART_NAMES]
EXAM_HEADER = ["time", "steps", "score", "falls", "attempts", "distance", "quality", "speed_error"]


# ---------- Files ----------

class Folder:
    """The training folder and everything written into it."""

    def __init__(self, path: Path):
        self.path = path
        self.brain = path / "brain"
        self.replays = path / "replays"
        self.brain.mkdir(parents=True, exist_ok=True)
        self.replays.mkdir(parents=True, exist_ok=True)

    def setup(self) -> dict:
        return json.loads((self.path / "setup.json").read_text(encoding="utf-8"))

    def write_json(self, name: str, data: dict) -> None:
        target = self.path / name
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        for attempt in range(5):  # Windows may hold the file open for a moment
            try:
                temporary.replace(target)
                return
            except PermissionError:
                time.sleep(0.2)

    def append_row(self, name: str, header: list[str], row: dict) -> None:
        target = self.path / name
        new = not target.exists()
        try:
            with target.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(header)
                writer.writerow([row.get(column, "") for column in header])
        except OSError:
            print(f"Could not write {name} (is it open elsewhere?). Training goes on.", flush=True)

    @property
    def stop_requested(self) -> bool:
        return (self.path / "stop.request").exists()


class Status:
    """status.json: what the session is doing, read by the page every few seconds."""

    def __init__(self, folder: Folder, total_steps: int):
        self.folder = folder
        previous = {}
        if (folder.path / "status.json").exists():
            try:
                previous = json.loads((folder.path / "status.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        self.data = {"state": "starting", "steps": previous.get("steps", 0), "total_steps": total_steps,
                     "steps_per_second": 0, "best_score": previous.get("best_score"),
                     "best_steps": previous.get("best_steps"), "best_quality": previous.get("best_quality"),
                     "last_exam": previous.get("last_exam"), "difficulty": previous.get("difficulty", 1.0),
                     "message": "Getting ready", "pid": os.getpid(), "started": datetime.now().isoformat(),
                     "exams_without_best": 0}

    def update(self, **values) -> None:
        self.data.update(values)
        self.data["updated"] = datetime.now().isoformat()
        self.folder.write_json("status.json", self.data)


# ---------- The practice worlds ----------

# One practice world. It lives at module level so Windows can start it in a new process
def make_env(animal: str, creature_values: dict, world_data: dict, senses: dict) -> Monitor:
    build = build_creature(animal, creature_values)
    world = world_from_dict(world_data)
    return Monitor(WalkEnv(build, world, senses),
                   info_keywords=("forward_distance", *[f"pts_{name}" for name in PART_NAMES]))


def build_vec_env(setup: dict, senses: dict, worlds: int, seed: int, mirror) -> VecNormalize:
    creature = setup["creature"]
    factory = functools.partial(make_env, creature["animal"], creature["values"], setup["world"], senses)
    if worlds > 1:
        venv = SubprocVecEnv([factory] * worlds, start_method="spawn")
    else:
        venv = DummyVecEnv([factory])
    venv.seed(seed)
    vec_normalize = VecNormalize(venv, norm_obs=True, norm_reward=True)
    if mirror is not None:
        size = vec_normalize.observation_space.shape
        vec_normalize.obs_rms = MirroredRunningMeanStd(mirror.observation_order, mirror.observation_signs, shape=size)
    return vec_normalize


# ---------- The exam ----------

# The asked speeds and ground heights of the exam: spread evenly over the world's ranges
def exam_plan(world, attempts: int = EXAM_ATTEMPTS) -> list[tuple[float, float]]:
    low, high = world.speed_range
    speeds = [low + (high - low) * i / max(attempts - 1, 1) for i in range(attempts)]
    speeds = [speeds[(i * 3) % attempts] for i in range(attempts)]  # mix speeds and heights
    tallest = world.tallest
    if world.values["mixed_heights"]:
        heights = [tallest * (i + 1) / attempts for i in range(attempts)]
    else:
        heights = [tallest] * attempts
    return list(zip(speeds, heights))


# The walker's best moves, without wobble, for a raw observation
def deterministic_actor(model, vec_normalize):
    def act(observation):
        scaled = vec_normalize.normalize_obs(np.asarray(observation, dtype=np.float32)[None, :])
        return model.predict(scaled, deterministic=True)[0][0]
    return act


def take_exam(model, vec_normalize, build, world, senses, scene) -> dict:
    act = deterministic_actor(model, vec_normalize)
    scores, falls, distances, qualities, speed_errors = [], 0, [], [], []
    for index, (speed, height) in enumerate(exam_plan(world)):
        env = WalkEnv(build, world, senses, target_speed=speed, terrain_height=height, scene=scene)
        observation, _ = env.reset(seed=EXAM_FIRST_SEED + index)
        total, fell, steps = 0.0, False, 0
        while True:
            observation, reward, fell, timed_out, info = env.step(act(observation))
            total += reward
            steps += 1
            if fell or timed_out:
                break
        seconds = steps * env.control_seconds
        distance = info["forward_distance"]
        asked = speed * env.world.values["attempt_seconds"]
        scores.append(total)
        falls += int(fell)
        distances.append(distance)
        qualities.append(min(max(distance / asked, 0.0), 1.0))
        speed_errors.append(abs(distance / max(seconds, 1e-9) - speed))
    return {"score": float(np.mean(scores)), "falls": falls, "attempts": len(scores),
            "distance": float(np.mean(distances)), "quality": float(np.mean(qualities) * 100),
            "speed_error": float(np.mean(speed_errors))}


# A replay of the brain walking at the middle speed on fairly tall ground
def best_walk_replay(model, vec_normalize, build, world, senses, scene) -> dict:
    speed = float(np.mean(world.speed_range))
    height = world.tallest * 0.75 if world.values["mixed_heights"] else world.tallest
    env = WalkEnv(build, world, senses, target_speed=speed, terrain_height=height, scene=scene)
    return record(env, deterministic_actor(model, vec_normalize), seed=EXAM_FIRST_SEED)


# ---------- Callbacks ----------

class SessionCallback(BaseCallback):
    """Everything that happens between lessons: progress, exams, difficulty, saving and stopping."""

    def __init__(self, folder: Folder, status: Status, vec_normalize, setup: dict, learning: dict, senses: dict,
                 total_steps: int):
        super().__init__()
        self.folder, self.status, self.vec_normalize = folder, status, vec_normalize
        self.setup, self.learning, self.senses = setup, learning, senses
        self.total_steps = total_steps
        self.build = build_creature(setup["creature"]["animal"], setup["creature"]["values"])
        self.world = world_from_dict(setup["world"])
        self.scene = build_scene(self.build, self.world)
        self.exam_every = int(learning["exam_every"]) * 1000
        self.patience = int(learning["patience"])
        self.adaptive = bool(learning["adaptive"])
        self.max_steps = round(self.world.values["attempt_seconds"] / 0.02)
        self.next_exam = 0
        self.next_save = 0
        self.clock = time.perf_counter()
        self.clock_steps = 0
        self.calls = 0
        self.stopped_by_request = False
        self.out_of_patience = False

    def _on_training_start(self) -> None:
        steps = self.num_timesteps
        self.next_exam = steps  # an exam at the start shows where it began
        self.next_save = steps + 250_000
        self.clock_steps = steps
        self._set_difficulty()
        self.status.update(state="training", message="Practising")

    def _on_step(self) -> bool:
        self.calls += 1
        if self.calls % 100 == 0 and self.folder.stop_requested:
            self.stopped_by_request = True
            return False
        return not self.out_of_patience

    def _on_rollout_start(self) -> None:
        if self.adaptive:
            self._adjust_learning_speed()
        self._set_difficulty()
        if self.num_timesteps >= self.next_exam:
            self._examine()
            self.next_exam = (self.num_timesteps // self.exam_every + 1) * self.exam_every
        if self.num_timesteps >= self.next_save:
            self.save("latest")
            self.next_save = self.num_timesteps + 250_000

    def _on_rollout_end(self) -> None:
        self._log_progress()

    # Gradually harder worlds: the bumps grow from flat to full height over the first half
    def _difficulty(self) -> float:
        if not self.world.values["gradually_harder"]:
            return 1.0
        return float(min(1.0, self.num_timesteps / max(RAMP_SHARE * self.total_steps, 1)))

    def _set_difficulty(self) -> None:
        self.training_env.env_method("set_difficulty", self._difficulty())

    def _adjust_learning_speed(self) -> None:
        change = self.model.logger.name_to_value.get("train/approx_kl")
        if change is None:
            return
        rate = self.model.learning_rate
        rule = STEP_SIZE_RULE
        if change > 2 * rule["target_change"]:
            rate /= rule["factor"]
        elif change < rule["target_change"] / 2:
            rate *= rule["factor"]
        self.model.learning_rate = float(min(max(rate, rule["min"]), rule["max"]))
        self.model._setup_lr_schedule()

    def _log_progress(self) -> None:
        episodes = list(self.model.ep_info_buffer)
        now = time.perf_counter()
        speed = (self.num_timesteps - self.clock_steps) / max(now - self.clock, 1e-9)
        self.clock, self.clock_steps = now, self.num_timesteps
        row = {"time": datetime.now().isoformat(timespec="seconds"), "steps": self.num_timesteps,
               "steps_per_second": round(speed), "difficulty": round(self._difficulty(), 3)}
        if episodes:
            row.update(score=round(float(np.mean([e["r"] for e in episodes])), 2),
                       distance=round(float(np.mean([e.get("forward_distance", 0.0) for e in episodes])), 3),
                       seconds=round(float(np.mean([e["l"] for e in episodes])) * 0.02, 2),
                       fall_share=round(float(np.mean([e["l"] < self.max_steps for e in episodes])), 3))
            for name in PART_NAMES:  # where the points came from, per attempt
                row[f"pts_{name}"] = round(float(np.mean([e.get(f"pts_{name}", 0.0) for e in episodes])), 2)
        self.folder.append_row("progress.csv", PROGRESS_HEADER, row)
        self.status.update(steps=self.num_timesteps, steps_per_second=round(speed), difficulty=row["difficulty"],
                           state="training", message="Practising")

    def _examine(self) -> None:
        self.status.update(state="examining", message="Taking an exam", steps=self.num_timesteps)
        result = take_exam(self.model, self.vec_normalize, self.build, self.world, self.senses, self.scene)
        row = {"time": datetime.now().isoformat(timespec="seconds"), "steps": self.num_timesteps,
               **{key: round(value, 3) if isinstance(value, float) else value for key, value in result.items()}}
        self.folder.append_row("exams.csv", EXAM_HEADER, row)
        replay = best_walk_replay(self.model, self.vec_normalize, self.build, self.world, self.senses, self.scene)
        replay["steps"] = self.num_timesteps
        replay["exam"] = row
        self.folder.write_json(f"replays/exam_{self.num_timesteps:012d}.json", replay)
        self._thin_replays()
        best = self.status.data.get("best_score")
        is_best = best is None or result["score"] > best
        if is_best:
            self.save("best")
            self.folder.write_json("replays/best.json", replay)
            self.status.data.update(best_score=result["score"], best_steps=self.num_timesteps,
                                    best_quality=result["quality"], exams_without_best=0)
            print(f"New best at {self.num_timesteps:,} steps: exam score {result['score']:.0f}, "
                  f"{result['falls']} falls, {result['quality']:.0f}% of the asked distance.", flush=True)
        else:
            self.status.data["exams_without_best"] = self.status.data.get("exams_without_best", 0) + 1
            if self.patience and self.status.data["exams_without_best"] >= self.patience:
                self.out_of_patience = True
                print(f"No new best for {self.patience} exams in a row: stopping early.", flush=True)
        self.status.update(state="training", message="Practising", last_exam=row)

    # Keep at most MOST_EXAM_REPLAYS exam replays, spread evenly over the session
    def _thin_replays(self) -> None:
        replays = sorted(self.folder.replays.glob("exam_*.json"))
        while len(replays) > MOST_EXAM_REPLAYS:
            # drop the one whose neighbours are closest together, but never the first or the last
            gaps = [int(replays[i + 1].stem[5:]) - int(replays[i - 1].stem[5:]) for i in range(1, len(replays) - 1)]
            replays.pop(1 + int(np.argmin(gaps))).unlink(missing_ok=True)

    def save(self, which: str) -> None:
        save_brain(self.model, self.vec_normalize, self.folder, which)


def save_brain(model, vec_normalize, folder: Folder, which: str) -> None:
    temporary = folder.brain / f"{which}.tmp.zip"
    model.save(temporary)
    temporary.replace(folder.brain / f"{which}.zip")
    vec_normalize.save(str(folder.brain / f"{which}_scaling.pkl"))
    save_scaling(vec_normalize, folder.brain / f"{which}_norm.npz")


# ---------- Keeping the computer awake ----------

# Windows flags that keep the computer from going to sleep while training runs (learn_to_walk's keep_awake)
KEEP_AWAKE_FLAGS = 0x80000000 | 0x00000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
NORMAL_SLEEP_FLAGS = 0x80000000              # ES_CONTINUOUS alone allows sleep again


def keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(KEEP_AWAKE_FLAGS if on else NORMAL_SLEEP_FLAGS)


# ---------- The session ----------

def _new_model(design, mirror, venv, settings, seed, motor_type):
    policy, policy_kwargs = policy_for(design, mirror.to_dict() if mirror else None, motor_type)
    return PPO(policy, venv, policy_kwargs=policy_kwargs, device="cpu", seed=seed, verbose=0, **settings)


def _start_from(model, venv, source: Path) -> None:
    model.set_parameters(str(source / "brain" / "best.zip"), exact_match=True, device="cpu")
    scaling = source / "brain" / "best_scaling.pkl"
    if scaling.exists():
        old = VecNormalize.load(str(scaling), venv.venv)
        venv.obs_rms = old.obs_rms
        venv.ret_rms = old.ret_rms


def run(path: Path, resume: bool) -> None:
    folder = Folder(path)
    setup = folder.setup()
    design = clean_design(setup.get("brain"))
    learning = clean_learning(setup.get("learning"))
    senses = design_senses(design)
    total_steps = int(learning["million_steps"] * 1_000_000)
    status = Status(folder, total_steps)
    (folder.path / "stop.request").unlink(missing_ok=True)
    status.update(state="starting", message="Building the practice worlds", total_steps=total_steps)

    build = build_creature(setup["creature"]["animal"], setup["creature"]["values"])
    design = design_for_body(design, build)
    mirror = mirror_tables(build, senses) if design["values"]["symmetric"] else None
    worlds = int(learning["parallel_worlds"])
    seed = int(learning["seed"])
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    venv = build_vec_env(setup, senses, worlds, seed, mirror)
    settings = ppo_settings(learning)

    if resume and (folder.brain / "latest.zip").exists():
        saved = VecNormalize.load(str(folder.brain / "latest_scaling.pkl"), venv.venv)
        venv = saved
        model = PPO.load(str(folder.brain / "latest.zip"), env=venv, device="cpu")
        print(f"Carrying on from {model.num_timesteps:,} steps.", flush=True)
    else:
        model = _new_model(design, mirror, venv, settings, seed, build.motor_type)
        source = setup.get("start_from")
        if source:
            _start_from(model, venv, path.parent / source)
            print(f"Starting from the best brain of {source}.", flush=True)
        else:
            print(f"A new brain with {sum(p.numel() for p in model.policy.parameters()):,} numbers to learn.",
                  flush=True)

    callback = SessionCallback(folder, status, venv, setup, learning, senses, total_steps)
    remaining = max(total_steps - model.num_timesteps, 0)
    keep_awake(True)
    try:
        if remaining:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
        else:
            callback.init_callback(model)  # nothing left to practise, but the final save and exam still need it
        callback.save("latest")
        if not callback.stopped_by_request:
            callback._examine()
        state, message = _ending(callback, model, total_steps)
        status.update(state=state, message=message, steps=model.num_timesteps)
        print(message, flush=True)
    finally:
        keep_awake(False)
        venv.close()


def _ending(callback: SessionCallback, model, total_steps: int) -> tuple[str, str]:
    if callback.stopped_by_request:
        return "stopped", "Paused. Press Continue to carry on from here."
    if callback.out_of_patience:
        return "finished", "Finished early: it had stopped getting better."
    return "finished", f"Finished all {total_steps:,} practice steps."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one training session of the studio.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    try:
        run(arguments.folder, arguments.resume)
    except Exception as error:  # the page shows the message; the full story goes to the log
        traceback.print_exc()
        folder = Folder(arguments.folder)
        status_path = folder.path / "status.json"
        data = {}
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        data.update(state="error", message=f"Something went wrong: {error}", updated=datetime.now().isoformat())
        folder.write_json("status.json", data)
        sys.exit(1)


if __name__ == "__main__":
    main()
