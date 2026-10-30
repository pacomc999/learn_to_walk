"""Load a saved snapshot as a plain function: observation in, action out.

This is for watching and judging only. The loaded statistics stay frozen.
"""
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from duck.bodies import STANDARD_BODY
from duck.duck_env import DuckWalkEnv
from duck.snapshots import Snapshot, list_snapshots
from duck.worlds import load_world


# A duck world that only carries the shapes of the senses and motors (the same for every world and body)
def _shape_carrier() -> DuckWalkEnv:
    return DuckWalkEnv(world=load_world("flat_walking"), body=STANDARD_BODY, roughness=0.0)


# Load the brain and its statistics. The one duck world only carries the shapes;
# callers step their own DuckWalkEnv.
def _load_model_and_scaler(snapshot: Snapshot) -> tuple[PPO, Callable[[np.ndarray], np.ndarray]]:
    venv = DummyVecEnv([_shape_carrier])
    vec_normalize = VecNormalize.load(str(snapshot.norm_path), venv)
    vec_normalize.training = False  # keep the statistics frozen
    vec_normalize.norm_reward = False
    model = PPO.load(str(snapshot.model_path), device="cpu")

    # Scale a raw observation the same way as in training, as a batch of one
    def scale(observation: np.ndarray) -> np.ndarray:
        return vec_normalize.normalize_obs(np.asarray(observation, dtype=np.float32)[None, :])

    return model, scale


# A function from a raw observation to the brain's best action
def load_policy(snapshot: Snapshot) -> Callable[[np.ndarray], np.ndarray]:
    model, scale = _load_model_and_scaler(snapshot)

    def policy(observation: np.ndarray) -> np.ndarray:
        return model.predict(scale(observation), deterministic=True)[0][0]

    return policy


# The snapshot a command line should use: the one with the given step count, or the newest.
# Prints a plain message and exits with code 1 when there is nothing suitable.
def choose_snapshot_or_exit(run_dir: Path, steps: int | None) -> Snapshot:
    snapshots = list_snapshots(run_dir)
    if not snapshots:
        print(f"No snapshots found in {run_dir}")
        sys.exit(1)
    if steps is None:
        return snapshots[-1]
    for snapshot in snapshots:
        if snapshot.steps == steps:
            return snapshot
    available = ", ".join(f"{snapshot.steps:,}" for snapshot in snapshots)
    print(f"No snapshot with {steps} steps. Available step counts: {available}")
    sys.exit(1)
