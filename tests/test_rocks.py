"""Tests for the rock test."""
import numpy as np
import pytest

from tests.worlds_for_tests import BODY, FLAT, STONES
from duck.duck_env import DuckWalkEnv, EVALUATION_TARGET_SPEED
from duck.rocks import FIRST_THROW, PARKED_POSITION, ROCK_LIFETIME, RockAttempt, RockWorld, film_page, rock_radius


# Hold every motor at the standing pose
def stand_still(observation):
    return np.zeros(12)


# Step a world with no commands until the given simulated time
def wait_until(env, seconds):
    while env.data.time < seconds:
        env.step(np.zeros(12))


# A 100 g stone ball is about 2.1 cm across at granite density
def test_rock_radius():
    assert rock_radius(0.1) * 100 == pytest.approx(2.12, abs=0.01)


# The rock world gives the duck exactly the senses of the normal flat world: the rock is not in them
def test_rock_world_senses_match_the_normal_world():
    rock_world = RockWorld(0.1, throw_seed=0, world=FLAT, body=BODY)
    plain_world = DuckWalkEnv(world=FLAT, body=BODY, target_speed=EVALUATION_TARGET_SPEED, roughness=0.0)
    rock_senses, _ = rock_world.reset(seed=3)
    plain_senses, _ = plain_world.reset(seed=3)
    assert rock_senses.shape == (39,)
    assert np.array_equal(rock_senses, plain_senses)
    for _ in range(20):  # before the first throw both worlds move the same way
        rock_senses = rock_world.step(np.zeros(12))[0]
        plain_senses = plain_world.step(np.zeros(12))[0]
    assert np.allclose(rock_senses, plain_senses, atol=1e-6)


# The first throw at 2 s hits the duck, and the rock is taken away again a second later
def test_a_throw_hits_and_the_rock_is_parked():
    env = RockWorld(0.1, throw_seed=0, world=FLAT, body=BODY)
    env.reset(seed=0)
    wait_until(env, FIRST_THROW + 0.4)
    assert env.throws == 1 and env.hits == 1
    wait_until(env, FIRST_THROW + ROCK_LIFETIME + 0.1)
    # Back at its waiting spot (it then settles onto the floor, so only x and y are checked)
    assert env.data.qpos[env.rock_qpos:env.rock_qpos + 2] == pytest.approx(PARKED_POSITION[:2], abs=0.01)


# The film page shows every attempt with how it ended
def test_film_page_lists_every_attempt():
    page = film_page("Duck 3.1", 150, [RockAttempt(20.0, False, 9, 9), RockAttempt(5.0, True, 2, 2)])
    assert "1 of 2 attempts walk the full 20 s" in page
    assert "attempt_2.mp4" in page and "falls after 5.0 s" in page and "walks all 20 s" in page


# A world with rocks throws them, on the flat plane and on the stone floor
@pytest.mark.parametrize("world, roughness", [(FLAT, 0.0), (STONES, 0.02)])
def test_rock_world_throws_on_both_floors(world, roughness):
    from dataclasses import replace
    from duck.worlds import Rocks
    env = DuckWalkEnv(world=replace(world, rocks=Rocks(0.1)), body=BODY, roughness=roughness, target_speed=0.4)
    env.reset(seed=0)
    wait_until(env, FIRST_THROW + 0.5)
    assert env.rock_thrower.throws == 1


# The attempt's seed fixes where the rocks come from, so exams with rocks repeat exactly
def test_same_seed_same_throws():
    from dataclasses import replace
    from duck.worlds import Rocks
    positions = []
    for _ in range(2):
        env = DuckWalkEnv(world=replace(FLAT, rocks=Rocks(0.1)), body=BODY, target_speed=0.4)
        env.reset(seed=5)
        wait_until(env, FIRST_THROW + 0.2)
        positions.append(env.data.qpos[env.rock_thrower.qpos_start:env.rock_thrower.qpos_start + 3].copy())
    assert np.array_equal(positions[0], positions[1])
