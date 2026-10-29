"""The exam: a fixed, noise-free test of the brain, taken regularly during training.

The training score includes the brain's random experimenting and a random target
speed per attempt, so it jumps around a lot. The exam removes that noise: the same
12 starts and target speeds every time, and the brain's best action without wobble.
Results go to <run>/exam.csv; the best brain so far is kept in <run>/best/.
"""
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from duck.duck_env import DuckWalkEnv

# The same attempts every exam: seeds 1000 to 1011, each with its own random target speed
EXAM_ATTEMPTS = 12
EXAM_FIRST_SEED = 1000

# The exam log and the folder holding the best brain
EXAM_LOG_NAME = "exam.csv"
EXAM_LOG_HEADER = ["time", "timesteps", "score", "falls", "distance"]
BEST_FOLDER_NAME = "best"

# Printed if the exam log is open in another program
LOCKED_EXAM_LOG_WARNING = "Warning: could not write the exam log (is it open elsewhere?). Training goes on."


# The result of one exam: average score and distance per attempt, and how many attempts fell
@dataclass(frozen=True)
class ExamResult:
    score: float
    falls: int
    distance: float


# One noise-free attempt; returns its score, whether it fell and its forward distance
def _exam_attempt(model, vec_normalize, env: DuckWalkEnv, seed: int) -> tuple[float, bool, float]:
    observation, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        scaled = vec_normalize.normalize_obs(np.asarray(observation, dtype=np.float32)[None, :])
        action = model.predict(scaled, deterministic=True)[0][0]
        observation, reward, fell, timed_out, info = env.step(action)
        total += reward
        if fell or timed_out:
            return total, fell, info["forward_distance"]


# Give the brain the full exam, using the sense scaling it trains with (without changing it)
def take_exam(model, vec_normalize, attempts: int = EXAM_ATTEMPTS) -> ExamResult:
    env = DuckWalkEnv()
    results = [_exam_attempt(model, vec_normalize, env, EXAM_FIRST_SEED + i) for i in range(attempts)]
    env.close()
    scores, falls, distances = zip(*results)
    return ExamResult(float(np.mean(scores)), int(sum(falls)), float(np.mean(distances)))


# Add one exam result to the log, writing the header first if the log is new
def append_exam(run_dir: Path, timesteps: int, result: ExamResult, when: datetime) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / EXAM_LOG_NAME
    is_new_log = not log_path.exists()
    try:
        with log_path.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            if is_new_log:
                writer.writerow(EXAM_LOG_HEADER)
            writer.writerow([when.isoformat(), timesteps, result.score, result.falls, result.distance])
    except OSError:
        print(LOCKED_EXAM_LOG_WARNING, flush=True)


# The best exam score in the log so far, or None if there is no usable row yet
def best_exam_score(run_dir: Path) -> float | None:
    log_path = run_dir / EXAM_LOG_NAME
    if not log_path.exists():
        return None
    scores = []
    with log_path.open(newline="") as log_file:
        for row in csv.DictReader(log_file):
            try:
                scores.append(float(row["score"]))
            except (KeyError, TypeError, ValueError):
                continue  # skip a broken row
    return max(scores) if scores else None


# The folder that holds the best brain, laid out like a run folder
def best_folder(run_dir: Path) -> Path:
    return run_dir / BEST_FOLDER_NAME
