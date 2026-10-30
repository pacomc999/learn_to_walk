import mujoco
import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
import duck.duck_env as duck_env_module
from tests.worlds_for_tests import BODY, FLAT, STONES
from duck.duck_env import DuckWalkEnv
from duck.floor import MAX_ROUGHNESS

def test_passes_gymnasium_checks():
    check_env(DuckWalkEnv(world=STONES, body=BODY), skip_render_check=True)

def test_spaces():
    env = DuckWalkEnv(world=STONES, body=BODY)
    assert env.observation_space.shape == (39,)
    assert env.action_space.shape == (12,)
    assert env.action_space.low.min() == -1 and env.action_space.high.max() == 1

def test_same_seed_same_start():
    env = DuckWalkEnv(world=STONES, body=BODY)
    first, _ = env.reset(seed=0)
    second, _ = env.reset(seed=0)
    assert np.array_equal(first, second)

def test_attempt_stops_after_20_seconds(monkeypatch):
    monkeypatch.setattr(duck_env_module, "has_fallen", lambda *args, **kwargs: False)
    env = DuckWalkEnv(world=STONES, body=BODY)
    env.reset(seed=0)
    for step in range(1, 1001):
        _, _, terminated, truncated, _ = env.step(np.zeros(12))
        assert not terminated
        assert truncated == (step == 1000)

def test_lying_down_ends_the_attempt():
    env = DuckWalkEnv(world=STONES, body=BODY)
    env.reset(seed=0)
    env.data.qpos[2] = 0.05
    _, _, terminated, _, _ = env.step(np.zeros(12))
    assert terminated

def test_extreme_actions_stay_finite():
    env = DuckWalkEnv(world=STONES, body=BODY)
    obs, _ = env.reset(seed=0)
    for step in range(1000):
        action = np.ones(12) if step % 2 else -np.ones(12)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()

def test_render_gives_an_image():
    env = DuckWalkEnv(world=STONES, body=BODY, render_mode="rgb_array", width=160, height=120, roughness=0.0)
    env.reset(seed=0)
    assert env.render().shape == (120, 160, 3)

# The forward reward uses the distance moved over the whole control step, not one speed sample
def test_forward_reward_uses_distance_moved():
    env = DuckWalkEnv(world=STONES, body=BODY, roughness=0.0)
    env.reset(seed=0)
    control_step_seconds = env.model.opt.timestep * duck_env_module.FRAME_SKIP
    x_before = float(env.data.qpos[0])
    _, _, _, _, info = env.step(np.ones(12))
    speed = (info["x_position"] - x_before) / control_step_seconds
    # Points peak at the target speed and fall off on both sides of it
    expected = env.reward_weights.forward * (env.target_speed - abs(speed - env.target_speed))
    assert info["reward_parts"]["forward"] == pytest.approx(expected)
    assert info["reward_parts"]["forward"] != pytest.approx(env.reward_weights.forward * float(env.data.qvel[0]))

# The duck can sense how far it has drifted sideways (the second number it sees)
def test_observation_includes_sideways_position():
    env = DuckWalkEnv(world=STONES, body=BODY)
    env.reset(seed=0)
    env.data.qpos[1] = 0.3
    assert env._observation()[1] == pytest.approx(0.3)

# Each step reports how far the duck has moved forward since the attempt began
def test_info_has_forward_distance():
    env = DuckWalkEnv(world=STONES, body=BODY)
    _, info = env.reset(seed=0)
    assert info["forward_distance"] == 0.0
    start_x = info["x_position"]
    for _ in range(10):
        _, _, _, _, info = env.step(np.ones(12))
    assert info["forward_distance"] == pytest.approx(info["x_position"] - start_x)


# Once settled, holding the standing pose uses almost no motor energy.
# The first 2 seconds cost a little while the duck sways out the reset jitter
def test_standing_uses_almost_no_energy():
    env = DuckWalkEnv(world=STONES, body=BODY, roughness=0.0)
    env.reset(seed=0)
    for _ in range(200):
        _, _, _, _, info = env.step(np.zeros(12))
    assert -0.001 < info["reward_parts"]["energy"] <= 0.0


# Swinging the legs hard costs energy points
def test_flailing_costs_energy():
    env = DuckWalkEnv(world=STONES, body=BODY, roughness=0.0)
    env.reset(seed=0)
    energy = 0.0
    for step in range(10):
        action = np.ones(12) if step % 2 else -np.ones(12)
        _, _, _, _, info = env.step(action)
        energy += info["reward_parts"]["energy"]
    assert energy < -0.1


# Each attempt picks a new target speed inside the range, and the duck senses it as its last number
def test_random_target_speed_each_attempt():
    env = DuckWalkEnv(world=STONES, body=BODY)
    low, high = STONES.target_speed_range
    targets = set()
    for seed in range(20):
        observation, _ = env.reset(seed=seed)
        assert low <= env.target_speed <= high
        assert observation[-1] == pytest.approx(env.target_speed)
        targets.add(round(env.target_speed, 4))
    assert len(targets) > 10


# A fixed target speed, as used for judging and videos, stays the same every attempt
def test_fixed_target_speed():
    env = DuckWalkEnv(world=STONES, body=BODY, target_speed=0.4)
    for seed in range(3):
        observation, _ = env.reset(seed=seed)
        assert env.target_speed == 0.4
        assert observation[-1] == pytest.approx(0.4)

# The hip swing gap between the legs at the start, minus the gap in the standing pose
def _start_hip_gap(env):
    left = env.model.joint("left_hip_swing").qposadr[0]
    right = env.model.joint("right_hip_swing").qposadr[0]
    stand = env.model.key("stand").qpos
    return (env.data.qpos[left] - stand[left]) - (env.data.qpos[right] - stand[right])

def test_stride_start_never_leaves_the_hips_level():
    env = DuckWalkEnv(world=STONES, body=BODY, roughness=0.0)
    gaps = []
    for seed in range(200):
        env.reset(seed=seed)
        gaps.append(_start_hip_gap(env))
    gaps = np.array(gaps)
    # The stride is bigger than the random nudge can undo, so a gap always remains
    assert np.abs(gaps).min() >= 2 * duck_env_module.START_NOISE - 1e-9
    # A coin flip picks the leading leg, so both legs lead sometimes
    assert (gaps > 0).sum() > 50 and (gaps < 0).sum() > 50


# The seed 0 start of the old flat world, captured before the rough floor was added
OLD_FLAT_SEED_0_START = np.array([
    0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.009208531118929386, -0.018361059948801994, 0.02066110633313656, 0.01253080926835537,
    0.01651022396981716, 0.004265430849045515, 0.009179862216114998, -0.0382549986243248,
    0.017402896657586098, 0.01263414230197668, -0.01989045925438404, 0.014296171255409718,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.5184808373451233,
], dtype=np.float32)


# Roughness exactly 0 is the old world: plane floor, nothing drawn, same start as before
def test_zero_roughness_is_the_old_flat_world():
    env = DuckWalkEnv(world=STONES, body=BODY, roughness=0.0)
    assert env.model.geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE
    first, _ = env.reset(seed=0)
    assert env.floor is None and env.roughness == 0.0
    # Same start pose and senses; the last sense is the asked speed, drawn from today's speed range
    assert np.array_equal(first[:-1], OLD_FLAT_SEED_0_START[:-1])
    assert STONES.target_speed_range[0] <= first[-1] <= STONES.target_speed_range[1]


# With no roughness given in a world with stones, every attempt draws its own roughness up to the world's tallest
def test_random_roughness_each_attempt():
    env = DuckWalkEnv(world=STONES, body=BODY)
    values = [(env.reset(seed=s), env.roughness)[1] for s in range(30)]
    assert all(0.0 <= v <= MAX_ROUGHNESS for v in values) and len(set(values)) == 30


# A flat world trains on the old flat plane (no height field, so faster physics)
def test_flat_training_uses_the_plane():
    env = DuckWalkEnv(world=FLAT, body=BODY)
    env.reset(seed=0)
    assert env.model.geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE and env.roughness == 0.0


# A new layout every attempt, and the same layout again for the same seed
def test_new_layout_every_attempt_same_for_same_seed():
    env = DuckWalkEnv(world=STONES, body=BODY, roughness=MAX_ROUGHNESS)
    env.reset(seed=4)
    a = env.floor.copy()
    env.reset(seed=5)
    assert not np.array_equal(a, env.floor)
    env.reset(seed=4)
    assert np.array_equal(a, env.floor)


# Deepest overlap of any contact at the start of each seed, in metres (negative means inside)
def _deepest_start_overlaps(env, seeds):
    overlaps = []
    for seed in seeds:
        env.reset(seed=seed)
        overlaps.append(min((c.dist for c in env.data.contact[:env.data.ncon]), default=0.0))
    return np.array(overlaps)


# The duck is lifted above the stones, so a rough start sinks in no deeper than the flat start does.
# The flat start already overlaps a few millimetres at the toes because of the stride stance
def test_rough_start_never_starts_inside_the_floor():
    flat = _deepest_start_overlaps(DuckWalkEnv(world=STONES, body=BODY, roughness=0.0), range(40))
    rough = _deepest_start_overlaps(DuckWalkEnv(world=STONES, body=BODY, roughness=MAX_ROUGHNESS), range(40))
    assert rough.min() > flat.min() - 0.001


# The renderer gets the new floor after every reset
def test_new_layout_is_sent_to_the_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(duck_env_module.mujoco, "mjr_uploadHField", lambda *args: calls.append(args))
    env = DuckWalkEnv(world=STONES, body=BODY, render_mode="rgb_array", roughness=MAX_ROUGHNESS)
    env.reset(seed=0)
    env.reset(seed=1)
    assert len(calls) >= 2


# The environment draws its speeds and stones from its world
def test_env_follows_the_world():
    from dataclasses import replace
    world = replace(STONES, target_speed_range=(0.3, 0.31))
    env = DuckWalkEnv(world=world, body=BODY)
    env.reset(seed=1)
    assert 0.3 <= env.target_speed <= 0.31 and 0 < env.roughness <= 0.04


# A flat world trains on the plain plane, with no floor drawn
def test_flat_world_uses_the_plain_plane():
    env = DuckWalkEnv(world=FLAT, body=BODY)
    env.reset(seed=0)
    assert env.floor is None and env.model.nhfield == 0


# The reward comes from the world's weights
def test_env_uses_the_worlds_reward():
    from dataclasses import replace
    from duck.rewards import RewardWeights
    world = replace(FLAT, reward=RewardWeights(forward=0.0, alive=1.0, energy=0.0, sideways=0.0))
    env = DuckWalkEnv(world=world, body=BODY)
    env.reset(seed=0)
    assert env.step(np.zeros(12))[1] == pytest.approx(1.0)


# A backpack world carries the backpack
def test_backpack_world_carries_it():
    from dataclasses import replace
    from duck.worlds import Backpack
    env = DuckWalkEnv(world=replace(FLAT, backpack=Backpack(0.5)), body=BODY)
    assert env.model.geom("backpack").size[0] > 0
    assert env.model.body("torso").mass[0] == pytest.approx(1.1 + 0.5)
