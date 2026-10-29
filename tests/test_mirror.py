import mujoco
import numpy as np
import torch

from duck.duck_env import FREE_JOINT_QPOS_SIZE, FREE_JOINT_QVEL_SIZE, DuckWalkEnv
from duck.mirror import (
    ACTION_ORDER, ACTION_SIGNS, mirror_action, mirror_observation,
)

# How many control steps the physics comparison runs
COMPARISON_STEPS = 25

# How closely the mirrored duck must follow the mirror image of the first duck
TOLERANCE = 1e-3


# Mirror the 12 joint numbers (angles or speeds) using the action table.
# The signs can be swapped by a test to break the hip side sign on purpose
def mirror_joints(joint_values, signs=ACTION_SIGNS):
    return np.asarray(joint_values)[ACTION_ORDER] * np.asarray(signs)


# The mirror image of a full qpos: y flips, quaternion (w, -x, y, -z), joints mirrored
def mirror_qpos(qpos, signs=ACTION_SIGNS):
    x, y, z = qpos[0:3]
    w, qx, qy, qz = qpos[3:7]
    free_part = np.array([x, -y, z, w, -qx, qy, -qz])
    return np.concatenate([free_part, mirror_joints(qpos[FREE_JOINT_QPOS_SIZE:], signs)])


# The mirror image of a full qvel: linear (vx, -vy, vz), angular (-wx, wy, -wz), joints mirrored
def mirror_qvel(qvel, signs=ACTION_SIGNS):
    vx, vy, vz, wx, wy, wz = qvel[0:6]
    free_part = np.array([vx, -vy, vz, -wx, wy, -wz])
    return np.concatenate([free_part, mirror_joints(qvel[FREE_JOINT_QVEL_SIZE:], signs)])


# Two ducks, B starting as the mirror image of A
def make_mirrored_pair(signs=ACTION_SIGNS):
    duck_a = DuckWalkEnv(target_speed=0.4)
    duck_a.reset(seed=0)
    duck_b = DuckWalkEnv(target_speed=0.4)
    duck_b.reset(seed=0)
    duck_b.data.qpos[:] = mirror_qpos(duck_a.data.qpos, signs)
    duck_b.data.qvel[:] = mirror_qvel(duck_a.data.qvel, signs)
    mujoco.mj_forward(duck_b.model, duck_b.data)
    return duck_a, duck_b


# Step both ducks with mirrored actions and return the largest observation mismatch
def largest_mismatch(signs=ACTION_SIGNS):
    duck_a, duck_b = make_mirrored_pair(signs)
    rng = np.random.default_rng(1)
    largest = 0.0
    for _ in range(COMPARISON_STEPS):
        action = rng.uniform(-1, 1, 12)
        observation_a, *_ = duck_a.step(action)
        observation_b, *_ = duck_b.step(mirror_action(action))
        difference = np.abs(observation_b - mirror_observation(observation_a)).max()
        largest = max(largest, float(difference))
    print("largest mirror difference:", largest)
    return largest


def test_mirroring_twice_gives_back_the_original():
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(5, 39)).astype(np.float32)
    actions = rng.normal(size=(5, 12)).astype(np.float32)
    assert np.array_equal(mirror_observation(mirror_observation(observations)), observations)
    assert np.array_equal(mirror_action(mirror_action(actions)), actions)
    torch_observations = torch.as_tensor(observations)
    torch_actions = torch.as_tensor(actions)
    assert torch.equal(mirror_observation(mirror_observation(torch_observations)), torch_observations)
    assert torch.equal(mirror_action(mirror_action(torch_actions)), torch_actions)
    assert isinstance(mirror_action(torch_actions), torch.Tensor)
    assert mirror_observation(observations).dtype == np.float32


def test_mirror_swaps_feet_and_flips_drift():
    observation = np.zeros(39, dtype=np.float32)
    observation[1] = 0.1
    observation[36] = 1.0
    mirrored = mirror_observation(observation)
    assert np.isclose(mirrored[1], -0.1)
    assert mirrored[36] == 0.0 and mirrored[37] == 1.0


def test_mirrored_duck_stays_a_mirror_image():
    assert largest_mismatch() < TOLERANCE


def test_physics_guard_catches_a_wrong_sign():
    wrong_signs = np.array(ACTION_SIGNS, dtype=float)
    wrong_signs[3] = 1.0  # left hip side sign not flipped
    wrong_signs[8] = 1.0  # right hip side sign not flipped
    assert largest_mismatch(wrong_signs) > TOLERANCE
