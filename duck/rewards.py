"""Reward and fall rules for the duck walker."""
import math
from dataclasses import dataclass


# How much each part of the reward counts for
@dataclass
class RewardWeights:
    forward: float = 3.0   # points per metre per second of forward speed
    alive: float = 0.2     # points for staying upright every step (small, so standing still does not pay)
    energy: float = 1.5    # penalty per joule of work the motors do
    sideways: float = 0.5  # penalty per metre of sideways drift


# Points for moving forward, up to the target speed and no more.
# Walking backwards gives a negative value
def _forward_part(forward_velocity: float, target_speed: float, weights: RewardWeights) -> float:
    return weights.forward * min(forward_velocity, target_speed)


# Points for staying upright, the same on every step
def _alive_part(weights: RewardWeights) -> float:
    return weights.alive


# Penalty for the real energy the motors used, so wasteful moves like hopping cost more
def _energy_part(energy_joules: float, weights: RewardWeights) -> float:
    return -weights.energy * energy_joules


# Penalty for drifting away from the straight line, the same on both sides
def _sideways_part(sideways_offset: float, weights: RewardWeights) -> float:
    return -weights.sideways * abs(sideways_offset)


# Work out all four parts and their total for one step.
# Without a target speed, forward points have no upper limit
def compute_reward(
    forward_velocity: float,
    sideways_offset: float,
    energy_joules: float,
    weights: RewardWeights = RewardWeights(),
    target_speed: float = math.inf,
) -> tuple[float, dict[str, float]]:
    parts = {
        "forward": _forward_part(forward_velocity, target_speed, weights),
        "alive": _alive_part(weights),
        "energy": _energy_part(energy_joules, weights),
        "sideways": _sideways_part(sideways_offset, weights),
    }
    total = sum(parts.values())
    return total, parts


# The attempt is over if the torso is too low or tilted too far
def has_fallen(height: float, up_z: float, min_height: float, min_up_z: float = 0.64) -> bool:
    too_low = height < min_height
    too_tilted = up_z < min_up_z  # 0.64 is about 50 degrees of tilt
    return too_low or too_tilted
