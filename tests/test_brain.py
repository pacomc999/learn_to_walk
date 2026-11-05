"""The mirror tables and the symmetric brain: the mirror image walker gets the mirror image commands."""
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from studio.brain import clean_design, count_weights, design_senses, policy_for
from studio.creatures import ANIMALS, build_creature
from studio.env import WalkEnv, observation_size
from studio.mirror import apply_table, mirror_tables
from studio.worlds import builtin_worlds

SENSES = [{"feet": True, "ground": False}, {"feet": True, "ground": True}, {"feet": False, "ground": False}]


@pytest.mark.parametrize("animal", list(ANIMALS))
@pytest.mark.parametrize("senses", SENSES)
def test_mirroring_twice_changes_nothing(animal, senses):
    build = build_creature(animal)
    tables = mirror_tables(build, senses)
    size = observation_size(build, senses)
    assert len(tables.observation_order) == size
    values = np.random.default_rng(0).normal(size=size)
    twice = apply_table(apply_table(values, tables.observation_order, tables.observation_signs),
                        tables.observation_order, tables.observation_signs)
    assert twice == pytest.approx(values)
    actions = np.random.default_rng(1).normal(size=len(build.motors))
    assert apply_table(apply_table(actions, tables.action_order, tables.action_signs),
                       tables.action_order, tables.action_signs) == pytest.approx(actions)


def _mirror_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


def _hopper_with_toes():
    from studio.creatures.custom import TEMPLATES
    parts = [dict(part) for part in TEMPLATES["hopper"]["parts"]]
    parts.append({"id": "toe", "parent": "foot", "x": 1, "y": 1, "pitch": 10, "yaw": 40, "joint": "hip", "mirror": True})
    return {"torso": TEMPLATES["hopper"]["torso"], "parts": parts}


def _bug():
    from studio.creatures.custom import TEMPLATES
    return TEMPLATES["bug"]


MIRROR_CASES = [("duck", None), ("puppy", None), ("custom", _bug), ("custom", _hopper_with_toes)]


@pytest.mark.parametrize("animal,body", MIRROR_CASES)
def test_the_mirror_table_matches_the_physics(animal, body):
    # A random pose and its exact mirror pose: every part must land mirrored, and the senses must match the table
    import mujoco
    build = build_creature(animal, {"body": body()} if body else {})
    senses = {"feet": True, "ground": True, "turn": True}
    tables = mirror_tables(build, senses)
    real, mirror = (WalkEnv(build, builtin_worlds()[0], senses, target_speed=0.4) for _ in range(2))
    real.reset(seed=0)
    mirror.reset(seed=0)
    rng = np.random.default_rng(3)
    q, v = real.data.qpos, real.data.qvel
    q[0:3] = [0.1, 0.05, real.scene.stand_height + 0.05]
    turn = rng.normal(size=4) * [1, 0.2, 0.2, 0.2] + [3, 0, 0, 0]
    q[3:7] = turn / np.linalg.norm(turn)
    v[0:6] = rng.normal(size=6)
    q[real.joint_qpos] = rng.uniform(-0.3, 0.3, len(build.motors))
    v[real.joint_qvel] = rng.normal(size=len(build.motors))
    m_q, m_v = mirror.data.qpos, mirror.data.qvel
    real.asked_heading, real.turn_rate = 0.3, 0.2         # asked to turn left...
    mirror.asked_heading, mirror.turn_rate = -0.3, -0.2   # ...is asked to turn right in the mirror
    m_q[0:3] = q[0:3] * [1, -1, 1]
    m_q[3:7] = q[3:7] * [1, -1, 1, -1]
    m_v[0:6] = v[0:6] * [1, -1, 1, -1, 1, -1]
    for index, motor in enumerate(build.motors):
        partner = build.motor_names.index(motor.mirror)
        m_q[mirror.joint_qpos[index]] = motor.mirror_sign * q[real.joint_qpos[partner]]
        m_v[mirror.joint_qvel[index]] = motor.mirror_sign * v[real.joint_qvel[partner]]
    mujoco.mj_forward(real.model, real.data)
    mujoco.mj_forward(mirror.model, mirror.data)
    for body in range(1, real.model.nbody):
        name = real.model.body(body).name
        expected = real.data.xpos[body] * [1, -1, 1]
        assert mirror.data.body(_mirror_name(name)).xpos == pytest.approx(expected, abs=1e-6), name
    mirrored = apply_table(real._observation(), tables.observation_order, tables.observation_signs)
    assert mirror._observation() == pytest.approx(mirrored, abs=1e-5)


@pytest.mark.parametrize("animal", list(ANIMALS))
def test_symmetric_brain_answers_the_mirror_image_with_the_mirror_image(animal):
    build = build_creature(animal)
    design = clean_design({"layers": [32, 32]})
    senses = design_senses(design)
    tables = mirror_tables(build, senses)
    env = DummyVecEnv([lambda: WalkEnv(build, builtin_worlds()[0], senses)])
    policy, kwargs = policy_for(design, tables.to_dict())
    model = PPO(policy, env, policy_kwargs=kwargs, device="cpu", n_steps=64, batch_size=32)
    for parameter in model.policy.parameters():  # make the network do something interesting
        torch.nn.init.normal_(parameter, std=0.3)
    observation = torch.as_tensor(np.random.default_rng(2).normal(size=(5, env.observation_space.shape[0])),
                                  dtype=torch.float32)
    mirrored = apply_table(observation, tables.observation_order, tables.observation_signs)
    with torch.no_grad():
        actions = model.policy.mlp_extractor.forward_actor(observation)
        mirrored_actions = model.policy.mlp_extractor.forward_actor(mirrored)
        values = model.policy.mlp_extractor.forward_critic(observation)
        mirrored_values = model.policy.mlp_extractor.forward_critic(mirrored)
    expected = apply_table(actions, tables.action_order, tables.action_signs)
    assert torch.allclose(mirrored_actions, expected, atol=1e-5)
    assert torch.allclose(values, mirrored_values, atol=1e-5)


def test_weight_count_matches_the_real_network():
    build = build_creature("duck")
    design = clean_design({"layers": [64, 32], "values": {"symmetric": False}})
    senses = design_senses(design)
    env = DummyVecEnv([lambda: WalkEnv(build, builtin_worlds()[0], senses)])
    policy, kwargs = policy_for(design, None)
    model = PPO(policy, env, policy_kwargs=kwargs, device="cpu", n_steps=64, batch_size=32)
    real = sum(p.numel() for p in model.policy.parameters()) - len(build.motors)  # minus the wobble
    assert count_weights(observation_size(build, senses), len(build.motors), design)["total"] == real


def test_designs_are_cleaned():
    design = clean_design({"layers": [100000, "x", -5], "values": {"activation": "nonsense"}})
    assert design["layers"] == [1024, 4]
    assert design["values"]["activation"] == "tanh"
    assert design["judge_layers"] == design["layers"]
