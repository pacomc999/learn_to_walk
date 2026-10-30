"""Judge a snapshot: does the duck really walk?

Run from the project root, for example:
    python -m duck.evaluate runs/duck_v1
    python -m duck.evaluate runs/duck_v1 --snapshot 1007616
    python -m duck.evaluate duck_catalogue/duck_03_1/brain --roughness 0.03 --body strong_legs
It uses the folder's own world and body; --world and --body try others.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

from duck.bodies import Body
from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv
from duck.floor import roughness_argument
from duck.policy import choose_snapshot_or_exit, load_policy
from duck.run_settings import add_settings_arguments, settings_or_exit
from duck.snapshots import Snapshot
from duck.worlds import World

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


# Run several attempts with different starting wobbles and count the successes, in a world with a body.
# roughness is the height of the cobblestones in metres (0.0 is the flat floor)
def evaluate_snapshot(snapshot: Snapshot, world: World, body: Body, attempts: int = 10, seed: int = 0,
                      roughness: float = 0.0) -> EvaluationResult:
    policy = load_policy(snapshot)
    env = DuckWalkEnv(world=world, body=body, target_speed=EVALUATION_TARGET_SPEED, roughness=roughness)
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
    parser.add_argument("--roughness", type=roughness_argument, default=0.0, help="height of the cobblestones in metres (default 0, a flat floor)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first attempt; the others count up from it (default 0)")
    parser.add_argument("--attempts", type=int, default=10, help="number of attempts (default 10)")
    add_settings_arguments(parser)
    arguments = parser.parse_args()
    snapshot = choose_snapshot_or_exit(arguments.run_dir, arguments.snapshot)
    world, body = settings_or_exit(arguments.run_dir, arguments.world, arguments.body)
    result = evaluate_snapshot(snapshot, world, body, attempts=arguments.attempts, seed=arguments.seed,
                               roughness=arguments.roughness)
    print(result_message(result))


if __name__ == "__main__":
    main()
