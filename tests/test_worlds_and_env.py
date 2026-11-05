"""Worlds, terrain and the practice environment."""
import numpy as np
import pytest

from studio.creatures import build_creature
from studio.env import WalkEnv, observation_size
from studio.terrain import SHAPES, capacity, floor_layout, ground_heights
from studio.worlds import builtin_worlds, world_from_dict


def test_there_are_eleven_builtin_worlds_with_unique_ids():
    worlds = builtin_worlds()
    assert len(worlds) == 11
    assert len({world.id for world in worlds}) == 11
    assert worlds[0].id == "flat_meadow"


def test_world_values_are_cleaned():
    world = world_from_dict({"id": "x", "name": "X", "values": {"gravity": -5, "terrain": "lava"}})
    assert world.values["gravity"] == 1.0
    assert world.values["terrain"] == "flat"
    assert world.terrain == "flat"


@pytest.mark.parametrize("terrain", list(SHAPES))
def test_terrain_heights_stay_within_what_the_floor_holds(terrain):
    layout = floor_layout(terrain, 0.7, 20, 1.0)
    heights = ground_heights(terrain, np.random.default_rng(0), layout, 0.03)
    assert heights.shape == (layout.rows, layout.columns)
    assert heights.min() >= 0
    assert heights.max() <= capacity(terrain, 0.03) + 1e-9
    assert heights.max() > 0


def test_floor_reaches_far_enough_for_the_fastest_speed():
    layout = floor_layout("hills", 1.5, 20, 1.0)
    assert layout.x_max >= 1.5 * 20


@pytest.mark.parametrize("world", builtin_worlds(), ids=lambda world: world.id)
@pytest.mark.parametrize("animal", ["duck", "puppy"])
def test_every_animal_can_practise_in_every_world(world, animal):
    build = build_creature(animal)
    env = WalkEnv(build, world, {"feet": True, "ground": True})
    observation, _ = env.reset(seed=1)
    assert observation.shape == (observation_size(build, {"feet": True, "ground": True}),)
    for _ in range(30):
        observation, reward, fell, timed_out, info = env.step(env.action_space.sample() * 0.2)
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        if fell or timed_out:
            env.reset()


def test_standing_still_is_not_a_fall_and_earns_the_alive_points():
    # Springy muscles hold still for free; direct ones pay a little energy to hold the pose
    env = WalkEnv(build_creature("duck", {"motor_type": "position"}), builtin_worlds()[0], target_speed=0.4)
    env.reset(seed=0)
    from studio.replay import holder
    hold = holder(env)
    total = 0.0
    for _ in range(100):
        _, reward, fell, _, info = env.step(hold(None))
        total += reward
        assert not fell
    assert total == pytest.approx(100 * 0.2, abs=2.0)


def test_attempt_ends_after_its_time():
    world = world_from_dict({"values": {"attempt_seconds": 5}})
    env = WalkEnv(build_creature("puppy"), world)
    env.reset(seed=0)
    from studio.replay import holder
    hold = holder(env)
    steps = 0
    while True:
        _, _, fell, timed_out, _ = env.step(hold(None))
        steps += 1
        if fell or timed_out:
            break
    assert timed_out and steps == 250


def test_gradually_harder_starts_flat():
    world = world_from_dict({"values": {"terrain": "cobblestones", "terrain_height": 3, "gradually_harder": True}})
    env = WalkEnv(build_creature("duck"), world)
    env.set_difficulty(0.0)
    env.reset(seed=0)
    assert env.attempt_height == 0.0
    env.set_difficulty(1.0)
    heights = [env.reset(seed=seed) and env.attempt_height for seed in range(10)]
    assert max(heights) > 0.0


def test_wind_pushes_and_rocks_fly():
    world = world_from_dict({"values": {"wind": True, "wind_strength": 50, "rocks": True, "rock_every": 0.5}})
    env = WalkEnv(build_creature("duck"), world, target_speed=0.4)
    env.reset(seed=0)
    from studio.replay import holder
    hold = holder(env)
    pushed = False
    rock_start = env.data.qpos[env.rock["qpos"]:env.rock["qpos"] + 3].copy()
    rock_moved = False
    for _ in range(150):
        env.step(hold(None))
        pushed = pushed or np.any(env.data.xfrc_applied[env.torso, :3] != 0)
        rock_moved = rock_moved or np.linalg.norm(env.data.qpos[env.rock["qpos"]:env.rock["qpos"] + 3] - rock_start) > 0.1
    assert pushed and rock_moved


def test_turning_worlds_ask_for_turns_and_score_along_the_asked_heading():
    import math
    world = world_from_dict({"values": {"turning": True, "turn_rate": 45}})
    assert world.turning
    assert not world_from_dict({"values": {"turning": True, "terrain": "stairs", "terrain_height": 3}}).turning
    build = build_creature("puppy")
    env = WalkEnv(build, world, {"feet": True, "turn": True}, target_speed=0.4, turn_rate=30)
    observation, _ = env.reset(seed=0)
    assert observation.shape == (observation_size(build, {"feet": True, "turn": True}),)
    from studio.replay import holder
    hold = holder(env)
    for _ in range(50):
        env.step(hold(None))
    assert env.asked_heading == pytest.approx(math.radians(30) * 50 * env.control_seconds)
    # moving sideways to the left counts as forward once the asked heading points left
    env.asked_heading = math.pi / 2
    before = env.progress
    env.data.qpos[1] += 0.1
    env._reward(float(env.data.qpos[0]), float(env.data.qpos[1]) - 0.1, 0.0, np.zeros(len(build.motors)), False)
    assert env.progress - before == pytest.approx(0.1)
