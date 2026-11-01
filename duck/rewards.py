"""Reward and fall rules for the duck walker."""
import math
from dataclasses import dataclass


# How much each part of the reward counts for (frozen, so a world can hold it)
@dataclass(frozen=True)
class RewardWeights:
    forward: float = 3.0   # points per metre per second of forward speed
    alive: float = 0.2     # points for staying upright every step (small, so standing still does not pay)
    energy: float = 1.5    # penalty per joule of work the motors do
    sideways: float = 0.5  # penalty per metre of sideways drift
    end_stops: float = 0.0  # penalty per unit of end stop closeness (0: none, as in every world before Duck 4.1)


# The share of a joint's range, at each end, that counts as "near the end stop"
END_STOP_ZONE = 0.05


# Points for moving forward. The most points come at the target speed, and going
# faster or slower than it loses the same amount per metre per second.
# With no target speed (infinite), the points just grow with speed.
# Walking backwards gives a negative value
def _forward_part(forward_velocity: float, target_speed: float, weights: RewardWeights) -> float:
    # No target: no upper limit, so the points are just speed times the weight
    if math.isinf(target_speed):
        return weights.forward * forward_velocity
    # Distance away from the target speed is subtracted from the target speed
    gap_from_target = abs(forward_velocity - target_speed)
    return weights.forward * (target_speed - gap_from_target)


# Points for staying upright, the same on every step
def _alive_part(weights: RewardWeights) -> float:
    return weights.alive


# Penalty for the real energy the motors used, so wasteful moves like hopping cost more
def _energy_part(energy_joules: float, weights: RewardWeights) -> float:
    return -weights.energy * energy_joules


# Penalty for drifting away from the straight line, the same on both sides
def _sideways_part(sideways_offset: float, weights: RewardWeights) -> float:
    return -weights.sideways * abs(sideways_offset)


# How close the joints are to their end stops, added up: each joint counts 0 until it enters the
# last END_STOP_ZONE of its range (5% of the range at each end), then rises to 1 at the stop
def end_stop_closeness(angles, ranges) -> float:
    total = 0.0
    for angle, (low, high) in zip(angles, ranges):
        room = min(angle - low, high - angle) / (high - low)  # 0.5 in the middle, 0 at a stop
        total += min(max((END_STOP_ZONE - room) / END_STOP_ZONE, 0.0), 1.0)
    return total


# Penalty for resting joints on their end stops, which hold the duck up for free
def _end_stop_part(closeness: float, weights: RewardWeights) -> float:
    return -weights.end_stops * closeness


# Work out all the parts (four, or five with an end stop penalty) and their total for one step.
# Without a target speed, forward points have no upper limit
def compute_reward(
    forward_velocity: float,
    sideways_offset: float,
    energy_joules: float,
    weights: RewardWeights = RewardWeights(),
    target_speed: float = math.inf,
    end_stop_closeness: float = 0.0,
) -> tuple[float, dict[str, float]]:
    parts = {
        "forward": _forward_part(forward_velocity, target_speed, weights),
        "alive": _alive_part(weights),
        "energy": _energy_part(energy_joules, weights),
        "sideways": _sideways_part(sideways_offset, weights),
    }
    # Only worlds with an end stop penalty have this part, so older results stay as they were
    if weights.end_stops:
        parts["end_stops"] = _end_stop_part(end_stop_closeness, weights)
    total = sum(parts.values())
    return total, parts


# The attempt is over if the torso is too low or tilted too far
def has_fallen(height: float, up_z: float, min_height: float, min_up_z: float = 0.64) -> bool:
    too_low = height < min_height
    too_tilted = up_z < min_up_z  # 0.64 is about 50 degrees of tilt
    return too_low or too_tilted
