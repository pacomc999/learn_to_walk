"""The left/right mirror table of the duck.

Mirroring is a reflection through the vertical plane along the walking line
(y becomes -y). Both functions work on numpy arrays and torch tensors whose last
axis holds the 39 observation numbers or the 12 motor numbers.
"""
import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd

# Motor order: neck_base, neck_top, left_hip_swing, left_hip_side, left_knee, left_ankle,
# left_toes, right_hip_swing, right_hip_side, right_knee, right_ankle, right_toes.
# Mirrored value number i is taken from number ACTION_ORDER[i], times ACTION_SIGNS[i]:
# the neck stays, the two legs swap, and the hip side motors change sign
ACTION_ORDER = [0, 1, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6]
ACTION_SIGNS = [1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1]

# The motors of the left leg, and of the neck (which has no partner)
LEFT_MOTORS = [2, 3, 4, 5, 6]
NECK_MOTORS = [0, 1]

# Where the groups sit in the 39 observation numbers
JOINT_ANGLE_START = 12
JOINT_SPEED_START = 24

# Observation parts that stay in place, with the sign they get when mirrored:
# 0 height (same), 1 sideways position (flips), 2 to 5 quaternion (w, -x, y, -z),
# 6 to 8 linear velocity (vx, -vy, vz), 9 to 11 angular velocity (-wx, wy, -wz)
BODY_SIGNS = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]


# Build the observation table: which number each mirrored number comes from, and its sign
def build_observation_table() -> tuple[list[int], list[int]]:
    order = list(range(12))
    signs = list(BODY_SIGNS)
    order += [JOINT_ANGLE_START + motor for motor in ACTION_ORDER]
    signs += ACTION_SIGNS
    order += [JOINT_SPEED_START + motor for motor in ACTION_ORDER]
    signs += ACTION_SIGNS
    order += [37, 36]    # left and right foot contact swap
    signs += [1, 1]
    order += [38]        # target speed stays
    signs += [1]
    return order, signs


OBSERVATION_ORDER, OBSERVATION_SIGNS = build_observation_table()


# Torch copies of the tables, made once and kept per (order, signs, device, dtype)
_TORCH_TABLES = {}


# The index and sign tensors for a table on this device and dtype, built only the first time
def torch_table(order: list[int], signs: list[int], device: torch.device, dtype: torch.dtype):
    key = (tuple(order), tuple(signs), device, dtype)
    if key not in _TORCH_TABLES:
        index = torch.as_tensor(order, device=device)
        sign_tensor = torch.as_tensor(signs, dtype=dtype, device=device)
        _TORCH_TABLES[key] = (index, sign_tensor)
    return _TORCH_TABLES[key]


# Reorder the last axis by the order list and multiply by the signs, for numpy or torch
def apply_table(values, order: list[int], signs: list[int]):
    if isinstance(values, torch.Tensor):
        index, sign_tensor = torch_table(order, signs, values.device, values.dtype)
        return values[..., index] * sign_tensor
    values = np.asarray(values)
    return values[..., order] * np.asarray(signs, dtype=values.dtype)


# The observation as the mirror image duck would sense it
def mirror_observation(observation):
    return apply_table(observation, OBSERVATION_ORDER, OBSERVATION_SIGNS)


# The action the mirror image duck would take
def mirror_action(action):
    return apply_table(action, ACTION_ORDER, ACTION_SIGNS)


# Sense scaling that learns from every observation and its mirror image together,
# so a sense and its mirror partner always get the same average and spread
class MirroredRunningMeanStd(RunningMeanStd):
    def update(self, batch):
        batch = np.asarray(batch)
        super().update(np.concatenate([batch, mirror_observation(batch)]))
