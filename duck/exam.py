"""The exam: a fixed, noise-free test of the brain, taken regularly during training.

The training score includes the brain's random experimenting and a random target
speed per attempt, so it jumps around a lot. The exam removes that noise: the same
12 starts every time, asked speeds spread evenly across the training range
(from Duck 3.2's second run; before, each attempt's seed drew a random speed),
and the brain's best action without wobble.
It has two parts, a flat floor and a rough floor, and the average of the two is the score.
The rough part climbs: its 12 attempts are spread evenly from 1/12 of TRAINING_ROUGHNESS up to all of it.
When training is flat (TRAINING_ROUGHNESS 0) there is no rough part and the score is the flat part's.
Results go to <run>/exam.csv; the best brain so far is kept in <run>/best/.
"""
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from duck.duck_env import TARGET_SPEED_RANGE, DuckWalkEnv
from duck.floor import TRAINING_ROUGHNESS

# The same attempts every exam: seeds 1000 to 1011 (each its own start), asked speeds from exam_speeds
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
    rough: ExamResult | None  # None when training (and so the exam) is flat


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


# The floor height of each rough exam attempt: evenly spread up to TRAINING_ROUGHNESS, like training,
# which picks any height from 0 to the most (at 3 cm: 0.25 cm, 0.5 cm, ... 3 cm)
def rough_exam_heights(attempts: int = EXAM_ATTEMPTS) -> list[float]:
    return [TRAINING_ROUGHNESS * (i + 1) / attempts for i in range(attempts)]


# The asked speed of each exam attempt: evenly spread from the slowest to the fastest training speed
# (at 0.2 to 1.5 m/s: 0.20, 0.32, ... 1.50), so the exam covers the same speeds as training
def exam_speeds(attempts: int = EXAM_ATTEMPTS) -> list[float]:
    low, high = TARGET_SPEED_RANGE
    if attempts == 1:
        return [low]
    return [low + (high - low) * i / (attempts - 1) for i in range(attempts)]


# One attempt on its own floor height. Each attempt keeps its seed, so its target speed
# and stone layout are the same at every height; only the stones' height changes
def _exam_attempt_on(model, vec_normalize, seed: int, roughness: float, speed: float) -> tuple[float, bool, float]:
    env = DuckWalkEnv(roughness=roughness, target_speed=speed)
    result = _exam_attempt(model, vec_normalize, env, seed)
    env.close()
    return result


# Give the brain one part of the exam, using the sense scaling it trains with (without changing it).
# roughness is one floor height for every attempt, or a list with one height per attempt
def take_exam(model, vec_normalize, attempts: int = EXAM_ATTEMPTS,
              roughness: float | list[float] = 0.0) -> ExamResult:
    heights = roughness if isinstance(roughness, list) else [roughness] * attempts
    speeds = exam_speeds(attempts)
    results = [_exam_attempt_on(model, vec_normalize, EXAM_FIRST_SEED + i, height, speeds[i])
               for i, height in enumerate(heights[:attempts])]
    scores, falls, distances = zip(*results)
    return ExamResult(float(np.mean(scores)), int(sum(falls)), float(np.mean(distances)))


# Give the brain the exam: the flat floor, and when training is rough also the climbing rough floor, combined
def take_full_exam(model, vec_normalize) -> FullExamResult:
    flat = take_exam(model, vec_normalize, roughness=0.0)
    if TRAINING_ROUGHNESS == 0.0:
        return FullExamResult(score=flat.score, falls=flat.falls, distance=flat.distance,
                              attempts=EXAM_ATTEMPTS, flat=flat, rough=None)
    rough = take_exam(model, vec_normalize, roughness=rough_exam_heights())
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
        "rough_score": result.rough.score if result.rough else "",
        "rough_falls": result.rough.falls if result.rough else "",
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
