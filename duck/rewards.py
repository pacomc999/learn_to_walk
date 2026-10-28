"""Reward and fall rules for the duck walker."""
from dataclasses import dataclass

import numpy as np


# How much each part of the reward counts for
@dataclass
class RewardWeights:
    forward: float = 1.0   # points per metre per second of forward speed
    alive: float = 0.5     # points for staying upright every step
    effort: float = 0.01   # penalty per squared motor action
    sideways: float = 0.5  # penalty per metre of sideways drift


# Points for moving forward. Walking backwards gives a negative value
def _forward_part(forward_velocity: float, weights: RewardWeights) -> float:
    return weights.forward * forward_velocity


# Points for staying upright, the same on every step
def _alive_part(weights: RewardWeights) -> float:
    return weights.alive


# Penalty for using large motor actions, so the duck saves energy
def _effort_part(action: np.ndarray, weights: RewardWeights) -> float:
    return -weights.effort * float(np.sum(np.square(action)))


# Penalty for drifting away from the straight line, the same on both sides
def _sideways_part(sideways_offset: float, weights: RewardWeights) -> float:
    return -weights.sideways * abs(sideways_offset)


# Work out all four parts and their total for one step
def compute_reward(
    forward_velocity: float,
    sideways_offset: float,
    action: np.ndarray,
    weights: RewardWeights = RewardWeights(),
) -> tuple[float, dict[str, float]]:
    parts = {
        "forward": _forward_part(forward_velocity, weights),
        "alive": _alive_part(weights),
        "effort": _effort_part(action, weights),
        "sideways": _sideways_part(sideways_offset, weights),
    }
    total = sum(parts.values())
    return total, parts


# The attempt is over if the torso is too low or tilted too far
def has_fallen(height: float, up_z: float, min_height: float, min_up_z: float = 0.64) -> bool:
    too_low = height < min_height
    too_tilted = up_z < min_up_z  # 0.64 is about 50 degrees of tilt
    return too_low or too_tilted
