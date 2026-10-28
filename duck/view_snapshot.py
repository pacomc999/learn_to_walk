"""Watch a snapshot of the duck walk in a live window.

Run from the project root, for example:
    python -m duck.view_snapshot runs/duck_v1
    python -m duck.view_snapshot runs/duck_v1 --snapshot 1007616
    python -m duck.view_snapshot runs/duck_v1 --speed 0.6
The duck starts again after each fall or after 20 seconds. Close the window to stop.
"""
import argparse
import time
from pathlib import Path

import mujoco.viewer

from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv
from duck.policy import choose_snapshot_or_exit, load_policy

# Real time seconds that one control step should take (the control step is 0.02 s)
SECONDS_PER_STEP = 0.02


# Walk until the duck falls or time runs out, keeping pace with the clock
def _walk_one_attempt(env: DuckWalkEnv, policy, viewer, seed: int) -> None:
    observation, _ = env.reset(seed=seed)
    finished = False
    while viewer.is_running() and not finished:
        step_started = time.perf_counter()
        observation, _, fell, timed_out, _ = env.step(policy(observation))
        viewer.sync()
        finished = fell or timed_out
        # Sleep for whatever is left of this step's real time
        time.sleep(max(0.0, SECONDS_PER_STEP - (time.perf_counter() - step_started)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a snapshot of the duck.")
    parser.add_argument("run_dir", type=Path, help="run folder, for example runs/duck_v1")
    parser.add_argument("--snapshot", type=int, default=None, help="step count of the snapshot (default: latest)")
    parser.add_argument("--speed", type=float, default=EVALUATION_TARGET_SPEED, help="speed to ask for, in metres per second")
    arguments = parser.parse_args()
    snapshot = choose_snapshot_or_exit(arguments.run_dir, arguments.snapshot)
    policy = load_policy(snapshot)
    env = DuckWalkEnv(target_speed=arguments.speed)
    print(f"Showing the snapshot from {snapshot.steps:,} steps at {arguments.speed} m/s. Close the window to stop.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        attempt = 0
        while viewer.is_running():
            _walk_one_attempt(env, policy, viewer, seed=attempt)
            attempt += 1
    env.close()


if __name__ == "__main__":
    main()
