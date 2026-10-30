"""The exam: a fixed, noise-free test of the brain, taken regularly during training.

The training score includes the brain's random experimenting and a random target
speed per attempt, so it jumps around a lot. The exam removes that noise: the same
12 starts and target speeds every time, and the brain's best action without wobble.
It has two parts, a flat floor and a rough floor, and the average of the two is the score.
Results go to <run>/exam.csv; the best brain so far is kept in <run>/best/.
"""
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from duck.duck_env import DuckWalkEnv
from duck.floor import MAX_ROUGHNESS

# The same attempts every exam: seeds 1000 to 1011, each with its own random target speed
EXAM_ATTEMPTS = 12
EXAM_FIRST_SEED = 1000

# The exam log and the folder holding the best brain
EXAM_LOG_NAME = "exam.csv"
EXAM_LOG_HEADER = ["time", "timesteps", "score", "falls", "distance", "attempts",
                   "flat_score", "flat_falls", "rough_score", "rough_falls"]
BEST_FOLDER_NAME = "best"

# Printed if the exam log is open in another program
LOCKED_EXAM_LOG_WARNING = "Warning: could not write the exam log (is it open elsewhere?). Training goes on."


# The result of one exam: average score and distance per attempt, and how many attempts fell
@dataclass(frozen=True)
class ExamResult:
    score: float
    falls: int
    distance: float


# The result of the whole exam: the two parts and their combination
# (score and distance are averages, falls and attempts are totals)
@dataclass(frozen=True)
class FullExamResult:
    score: float
    falls: int
    distance: float
    attempts: int
    flat: ExamResult
    rough: ExamResult


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


# Give the brain one part of the exam (on a floor of the given roughness),
# using the sense scaling it trains with (without changing it)
def take_exam(model, vec_normalize, attempts: int = EXAM_ATTEMPTS, roughness: float = 0.0) -> ExamResult:
    env = DuckWalkEnv(roughness=roughness)
    results = [_exam_attempt(model, vec_normalize, env, EXAM_FIRST_SEED + i) for i in range(attempts)]
    env.close()
    scores, falls, distances = zip(*results)
    return ExamResult(float(np.mean(scores)), int(sum(falls)), float(np.mean(distances)))


# Give the brain both parts of the exam, flat floor and rough floor, and combine them
def take_full_exam(model, vec_normalize) -> FullExamResult:
    flat = take_exam(model, vec_normalize, roughness=0.0)
    rough = take_exam(model, vec_normalize, roughness=MAX_ROUGHNESS)
    return FullExamResult(
        score=(flat.score + rough.score) / 2,
        falls=flat.falls + rough.falls,
        distance=(flat.distance + rough.distance) / 2,
        attempts=2 * EXAM_ATTEMPTS,
        flat=flat,
        rough=rough,
    )


# The values of one log row, by column name
def _exam_row_values(timesteps: int, result: FullExamResult, when: datetime) -> dict:
    return {
        "time": when.isoformat(), "timesteps": timesteps, "score": result.score, "falls": result.falls,
        "distance": result.distance, "attempts": result.attempts,
        "flat_score": result.flat.score, "flat_falls": result.flat.falls,
        "rough_score": result.rough.score, "rough_falls": result.rough.falls,
    }


# The columns an existing log has (an old log keeps its own, shorter header)
def _log_header(log_path: Path) -> list[str]:
    if not log_path.exists():
        return EXAM_LOG_HEADER
    with log_path.open(newline="") as log_file:
        return next(csv.reader(log_file), EXAM_LOG_HEADER)


# Add one exam result to the log, writing the header first if the log is new
def append_exam(run_dir: Path, timesteps: int, result: FullExamResult, when: datetime) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / EXAM_LOG_NAME
    is_new_log = not log_path.exists()
    header = _log_header(log_path)
    values = _exam_row_values(timesteps, result, when)
    try:
        with log_path.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            if is_new_log:
                writer.writerow(header)
            writer.writerow([values[column] for column in header])
    except OSError:
        print(LOCKED_EXAM_LOG_WARNING, flush=True)


# A number from a log column, or None if the column is missing or empty
def _optional_number(row: dict, column: str) -> float | None:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError):
        return None


# Every usable exam in the log, as dictionaries; broken rows are skipped.
# Old logs have no attempts column (12 then) and no part scores (None then)

def read_exams(run_dir: Path) -> list[dict]:
    log_path = run_dir / EXAM_LOG_NAME
    if not log_path.exists():
        return []
    exams = []
    with log_path.open(newline="") as log_file:
        for row in csv.DictReader(log_file):
            try:
                exam = {"steps": int(row["timesteps"]), "score": float(row["score"]), "falls": int(row["falls"])}
                attempts = _optional_number(row, "attempts")
                exam["attempts"] = int(attempts) if attempts is not None else EXAM_ATTEMPTS
                exam["flat_score"] = _optional_number(row, "flat_score")
                exam["rough_score"] = _optional_number(row, "rough_score")
                exams.append(exam)
            except (KeyError, TypeError, ValueError):
                continue  # skip a broken row
    return exams


# The best exam score in the log so far, or None if there is no usable row yet
def best_exam_score(run_dir: Path) -> float | None:
    scores = [exam["score"] for exam in read_exams(run_dir)]
    return max(scores) if scores else None


# The exam taken at this step count, or None if there was none
def exam_at(run_dir: Path, steps: int) -> dict | None:
    return next((exam for exam in read_exams(run_dir) if exam["steps"] == steps), None)


# The folder that holds the best brain, laid out like a run folder
def best_folder(run_dir: Path) -> Path:
    return run_dir / BEST_FOLDER_NAME
