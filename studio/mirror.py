"""Left/right mirror tables for any animal, worked out from its build.

Mirroring reflects the walker through the vertical plane along its walking line (y becomes -y).
A sense or a motor command of the mirror image walker is some other sense or command of the
real walker, maybe with its sign flipped. The tables say which one and which sign, so the
symmetric brain can treat the left and right side alike (learn_to_walk's duck/mirror.py,
made general).
"""
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd

from studio.creatures.kit import Build
from studio.env import GROUND_SCAN

# Torso senses that keep their place: height (same), sideways position (flips),
# orientation (w, -x, y, -z), linear velocity (vx, -vy, vz), angular velocity (-wx, wy, -wz)
BODY_SIGNS = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]


# The mirror tables of one walker with one set of senses
@dataclass(frozen=True)
class MirrorTables:
    action_order: list[int]
    action_signs: list[int]
    observation_order: list[int]
    observation_signs: list[int]
    left_motors: list[int]       # motors on the left side
    middle_motors: list[int]     # motors on the middle line (neck, tail)

    def to_dict(self) -> dict:
        return {key: list(value) for key, value in self.__dict__.items()}


def mirror_tables(build: Build, senses: dict) -> MirrorTables:
    names = build.motor_names
    action_order = [names.index(motor.mirror) for motor in build.motors]
    action_signs = [motor.mirror_sign for motor in build.motors]
    count = len(names)
    order, signs = list(range(12)), list(BODY_SIGNS)
    order += [12 + index for index in action_order]
    signs += action_signs
    order += [12 + count + index for index in action_order]
    signs += action_signs
    position = 12 + 2 * count
    if senses.get("feet", True):
        feet = [foot.name for foot in build.feet]
        order += [position + feet.index(foot.mirror) for foot in build.feet]
        signs += [1] * len(feet)
        position += len(feet)
    if senses.get("ground", False):
        order += [position + i for i in range(len(GROUND_SCAN))]  # the ground ahead lies on the middle line
        signs += [1] * len(GROUND_SCAN)
        position += len(GROUND_SCAN)
    order.append(position)  # the asked speed stays
    signs.append(1)
    left = [i for i, motor in enumerate(build.motors) if motor.side == "left"]
    middle = [i for i, motor in enumerate(build.motors) if motor.side == "middle"]
    return MirrorTables(action_order, action_signs, order, signs, left, middle)


# Reorder the last axis by the order list and multiply by the signs, for numpy or torch
def apply_table(values, order: list[int], signs: list[int]):
    if isinstance(values, torch.Tensor):
        index = torch.as_tensor(order, device=values.device)
        sign = torch.as_tensor(signs, dtype=values.dtype, device=values.device)
        return values[..., index] * sign
    values = np.asarray(values)
    return values[..., order] * np.asarray(signs, dtype=values.dtype)


# Sense scaling that learns from every observation and its mirror image together,
# so a sense and its mirror partner always get the same average and spread
class MirroredRunningMeanStd(RunningMeanStd):
    def __init__(self, order: list[int], signs: list[int], epsilon: float = 1e-4, shape=()):
        super().__init__(epsilon=epsilon, shape=shape)
        self.order = list(order)
        self.signs = list(signs)

    def update(self, batch):
        batch = np.asarray(batch)
        super().update(np.concatenate([batch, apply_table(batch, self.order, self.signs)]))
