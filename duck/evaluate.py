"""Judge a snapshot: does the duck really walk?

Run from the project root, for example:
    python -m duck.evaluate runs/duck_v1
    python -m duck.evaluate runs/duck_v1 --snapshot 1007616
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

from duck.duck_env import DuckWalkEnv
from duck.policy import choose_snapshot_or_exit, load_policy
from duck.snapshots import Snapshot

# What counts as a good walk
MIN_AVERAGE_SPEED = 0.2  # metres per second
MAX_SIDEWAYS = 0.5       # metres away from the straight line
REQUIRED_SUCCESSES = 8   # successful attempts needed out of 10


# How many attempts walked, out of how many, and whether that is enough
@dataclass
class EvaluationResult:
    successes: int
    attempts: int
    passed: bool


# One attempt is a success if the duck stayed up, moved forward steadily and stayed near the line
def judge_attempt(fell: bool, seconds: float, distance_x: float, final_y: float) -> bool:
    if fell:
        return False
    return distance_x / seconds >= MIN_AVERAGE_SPEED and abs(final_y) <= MAX_SIDEWAYS


# Let the duck walk one attempt and judge it
def _run_attempt(env: DuckWalkEnv, policy, seed: int) -> bool:
    observation, info = env.reset(seed=seed)
    start_x = info["x_position"]
    start_time = env.data.time
    fell = False
    finished = False
    while not finished:
        observation, _, fell, timed_out, info = env.step(policy(observation))
        finished = fell or timed_out
    seconds = env.data.time - start_time
    return judge_attempt(fell, seconds, info["x_position"] - start_x, info["y_position"])


# Run several attempts with different starting wobbles and count the successes
def evaluate_snapshot(snapshot: Snapshot, attempts: int = 10, seed: int = 0) -> EvaluationResult:
    policy = load_policy(snapshot)
    env = DuckWalkEnv()
    successes = sum(_run_attempt(env, policy, seed + i) for i in range(attempts))
    env.close()
    needed = REQUIRED_SUCCESSES * attempts / 10
    return EvaluationResult(successes, attempts, successes >= needed)


# The sentence printed after an evaluation
def result_message(result: EvaluationResult) -> str:
    verdict = "The duck can walk!" if result.passed else "Not there yet."
    return f"{result.successes} of {result.attempts} attempts walked. {verdict}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge a snapshot of the duck.")
    parser.add_argument("run_dir", type=Path, help="run folder, for example runs/duck_v1")
    parser.add_argument("--snapshot", type=int, default=None, help="step count of the snapshot (default: latest)")
    arguments = parser.parse_args()
    snapshot = choose_snapshot_or_exit(arguments.run_dir, arguments.snapshot)
    print(result_message(evaluate_snapshot(snapshot)))


if __name__ == "__main__":
    main()
