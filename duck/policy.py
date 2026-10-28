"""Load a saved snapshot as a plain function: observation in, action out.

This is for watching and judging only. The loaded statistics stay frozen.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from duck.duck_env import DuckWalkEnv
from duck.snapshots import Snapshot, list_snapshots


# What the brain did with one observation: the action it chose,
# and how strongly each cell of its two hidden layers fired (-1 to 1)
@dataclass
class BrainReading:
    action: np.ndarray
    layer1: np.ndarray
    layer2: np.ndarray


# Load the brain and its statistics. The one duck world only carries the shapes;
# callers step their own DuckWalkEnv.
def _load_model_and_scaler(snapshot: Snapshot) -> tuple[PPO, Callable[[np.ndarray], np.ndarray]]:
    venv = DummyVecEnv([DuckWalkEnv])
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


# A function from a raw observation to a BrainReading. It runs the same layers as
# load_policy one at a time, so the cell activity can be kept along the way
def load_brain(snapshot: Snapshot) -> Callable[[np.ndarray], BrainReading]:
    model, scale = _load_model_and_scaler(snapshot)
    first_layer = model.policy.mlp_extractor.policy_net[0:2]   # Linear then Tanh
    second_layer = model.policy.mlp_extractor.policy_net[2:4]  # Linear then Tanh
    action_layer = model.policy.action_net

    def brain(observation: np.ndarray) -> BrainReading:
        with torch.no_grad():
            layer1 = first_layer(torch.as_tensor(scale(observation)))
            layer2 = second_layer(layer1)
            action = action_layer(layer2).clamp(-1.0, 1.0)  # the same limits predict uses
        return BrainReading(action[0].numpy(), layer1[0].numpy(), layer2[0].numpy())

    return brain


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
