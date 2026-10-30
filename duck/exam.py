"""The exam: a fixed, noise-free test of the brain, taken regularly during training.

The training score includes the brain's random experimenting and a random target
speed per attempt, so it jumps around a lot. The exam removes that noise: the same
12 starts every time, asked speeds spread evenly across the training range
(from Duck 3.2's second run; before, each attempt's seed drew a random speed),
and the brain's best action without wobble.
The exam follows the duck's world. In a flat world it is 12 attempts on the flat floor.
In a world with stones it matches training: 24 attempts with floor heights spread evenly
from flat up to the world's tallest stones (from Duck 3.1's second run). It is reported in two halves, the
lower stones and the taller stones, and the score is the average of all 24.
The world's extras (rocks, a backpack) are on in every attempt, each attempt's seed fixing them.
Older logs have a flat part and a rough part instead (12 flat, 12 climbing rough), kept readable.
Results go to <run>/exam.csv; the best brain so far is kept in <run>/best/.
"""
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from duck.bodies import Body
from duck.duck_env import DuckWalkEnv
from duck.worlds import World

# The same attempts every exam: seeds from 1000 (each its own start), asked speeds from exam_speeds.
# A flat exam has 12 attempts, a stone exam 24 (seeds 1000 to 1023)
EXAM_ATTEMPTS = 12
STONE_EXAM_ATTEMPTS = 24
EXAM_FIRST_SEED = 1000

# The stone exam's speeds: attempt i asks for speed number (i x 5) mod 12 of the 12 exam speeds,
# so every half gets all 12 speeds, mixed, and the tallest stones do not always get the fastest speed
SPEED_MIX_STEP = 5

# The named parts an exam can have: flat (and rough) in older exams, lower and taller in stone exams
EXAM_PART_NAMES = ["flat", "rough", "lower", "taller"]

# The exam log and the folder holding the best brain
EXAM_LOG_NAME = "exam.csv"
EXAM_LOG_HEADER = ["time", "timesteps", "score", "falls", "distance", "attempts"] + [
    f"{name}_{column}" for name in EXAM_PART_NAMES for column in ("score", "falls")]
BEST_FOLDER_NAME = "best"

# Printed if the exam log is open in another program
LOCKED_EXAM_LOG_WARNING = "Warning: could not write the exam log (is it open elsewhere?). Training goes on."


# The result of one exam: average score and distance per attempt, and how many attempts fell
@dataclass(frozen=True)
class ExamResult:
    score: float
    falls: int
    distance: float


# The result of the whole exam: its parts by name and their combination
# (score and distance are averages, falls and attempts are totals).
# parts is {"flat": ...} for a flat exam and {"lower": ..., "taller": ...} for a stone exam
@dataclass(frozen=True)
class FullExamResult:
    score: float
    falls: int
    distance: float
    attempts: int
    parts: dict[str, ExamResult]


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


# The floor height of each stone exam attempt: evenly spread from flat up to the world's tallest stones,
# like training, which picks any height from 0 to the most (at 4 cm: 0, 0.17 cm, 0.35 cm, ... 4 cm)
def stone_exam_heights(world: World, attempts: int = STONE_EXAM_ATTEMPTS) -> list[float]:
    return [world.tallest_stones * i / (attempts - 1) for i in range(attempts)]


# The asked speed of each stone exam attempt: each half uses all 12 exam speeds, mixed (see SPEED_MIX_STEP)
def stone_exam_speeds(world: World, attempts: int = STONE_EXAM_ATTEMPTS) -> list[float]:
    speeds = exam_speeds(world, EXAM_ATTEMPTS)
    return [speeds[(i * SPEED_MIX_STEP) % EXAM_ATTEMPTS] for i in range(attempts)]


# The asked speed of each exam attempt: evenly spread from the slowest to the fastest speed of the world
# (at 0.2 to 1.5 m/s: 0.20, 0.32, ... 1.50), so the exam covers the same speeds as training
def exam_speeds(world: World, attempts: int = EXAM_ATTEMPTS) -> list[float]:
    low, high = world.target_speed_range
    if attempts == 1:
        return [low]
    return [low + (high - low) * i / (attempts - 1) for i in range(attempts)]


# One attempt on its own floor height. Each attempt keeps its seed, so its target speed
# and stone layout are the same at every height; only the stones' height changes
def _exam_attempt_on(model, vec_normalize, world: World, body: Body, seed: int, roughness: float,
                     speed: float) -> tuple[float, bool, float]:
    env = DuckWalkEnv(world=world, body=body, roughness=roughness, target_speed=speed)
    result = _exam_attempt(model, vec_normalize, env, seed)
    env.close()
    return result


# A group of attempts, each with its own seed, floor height and asked speed, summed up as one part
def _exam_part(model, vec_normalize, world: World, body: Body, seeds: list[int], heights: list[float],
               speeds: list[float]) -> ExamResult:
    results = [_exam_attempt_on(model, vec_normalize, world, body, seed, height, speed)
               for seed, height, speed in zip(seeds, heights, speeds)]
    scores, falls, distances = zip(*results)
    return ExamResult(float(np.mean(scores)), int(sum(falls)), float(np.mean(distances)))


# Give the brain one part of the exam, using the sense scaling it trains with (without changing it).
# roughness is one floor height for every attempt, or a list with one height per attempt
def take_exam(model, vec_normalize, world: World, body: Body, attempts: int = EXAM_ATTEMPTS,
              roughness: float | list[float] = 0.0) -> ExamResult:
    heights = roughness if isinstance(roughness, list) else [roughness] * attempts
    seeds = [EXAM_FIRST_SEED + i for i in range(attempts)]
    return _exam_part(model, vec_normalize, world, body, seeds, heights[:attempts], exam_speeds(world, attempts))


# The stone exam: 24 attempts from flat to the tallest stones, scored in a lower and a taller half
def _take_stone_exam(model, vec_normalize, world: World, body: Body) -> FullExamResult:
    seeds = [EXAM_FIRST_SEED + i for i in range(STONE_EXAM_ATTEMPTS)]
    heights, speeds = stone_exam_heights(world), stone_exam_speeds(world)
    half = STONE_EXAM_ATTEMPTS // 2
    lower = _exam_part(model, vec_normalize, world, body, seeds[:half], heights[:half], speeds[:half])
    taller = _exam_part(model, vec_normalize, world, body, seeds[half:], heights[half:], speeds[half:])
    return FullExamResult(
        score=(lower.score + taller.score) / 2,  # both halves have 12 attempts: the average of all 24
        falls=lower.falls + taller.falls,
        distance=(lower.distance + taller.distance) / 2,
        attempts=STONE_EXAM_ATTEMPTS,
        parts={"lower": lower, "taller": taller},
    )


# Give the brain the exam of its world: flat when the world is flat, otherwise the stone exam
def take_full_exam(model, vec_normalize, world: World, body: Body) -> FullExamResult:
    if world.tallest_stones > 0.0:
        return _take_stone_exam(model, vec_normalize, world, body)
    flat = take_exam(model, vec_normalize, world, body, roughness=0.0)
    return FullExamResult(score=flat.score, falls=flat.falls, distance=flat.distance,
                          attempts=EXAM_ATTEMPTS, parts={"flat": flat})


# The part scores in words, for printed messages, for example "lower 990, taller 870"
def parts_in_words(result: FullExamResult) -> str:
    return ", ".join(f"{name} {part.score:.0f}" for name, part in result.parts.items())


# The values of one log row, by column name; parts the exam did not have stay empty
def _exam_row_values(timesteps: int, result: FullExamResult, when: datetime) -> dict:
    values = {
        "time": when.isoformat(), "timesteps": timesteps, "score": result.score, "falls": result.falls,
        "distance": result.distance, "attempts": result.attempts,
    }
    for name in EXAM_PART_NAMES:
        part = result.parts.get(name)
        values[f"{name}_score"] = part.score if part else ""
        values[f"{name}_falls"] = part.falls if part else ""
    return values


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
                for name in EXAM_PART_NAMES:
                    exam[f"{name}_score"] = _optional_number(row, f"{name}_score")
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
