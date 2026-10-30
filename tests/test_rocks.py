"""Tests for the rock test."""
import csv

import numpy as np
import pytest

from duck.duck_env import DuckWalkEnv, EVALUATION_TARGET_SPEED
from duck.rocks import (FIRST_THROW, PARKED_POSITION, ROCK_LIFETIME, RockAttempt, RockRound, RockWorld,
                        film_page, rock_radius, rock_round, write_results)


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
    rock_world = RockWorld(0.1, throw_seed=0)
    plain_world = DuckWalkEnv(target_speed=EVALUATION_TARGET_SPEED, roughness=0.0)
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
    env = RockWorld(0.1, throw_seed=0)
    env.reset(seed=0)
    wait_until(env, FIRST_THROW + 0.4)
    assert env.throws == 1 and env.hits == 1
    wait_until(env, FIRST_THROW + ROCK_LIFETIME + 0.1)
    # Back at its waiting spot (it then settles onto the floor, so only x and y are checked)
    assert env.data.qpos[env.rock_qpos:env.rock_qpos + 2] == pytest.approx(PARKED_POSITION[:2], abs=0.01)


# A round sums up its attempts
def test_rock_round_sums_its_attempts():
    one_round = rock_round(stand_still, grams=25, attempts=2)
    assert one_round.attempts == 2 and 0 <= one_round.walked <= 2
    assert one_round.hits <= one_round.throws and one_round.radius_cm == pytest.approx(1.3, abs=0.1)


# The results file has one row per rock size with every number
def test_write_results(tmp_path):
    rounds = [RockRound(100, 2.1, 10, 10, 20.0, 90, 90), RockRound(200, 2.7, 0, 10, 6.2, 26, 26)]
    write_results(rounds, tmp_path / "rocks" / "results.csv")
    rows = list(csv.DictReader((tmp_path / "rocks" / "results.csv").open()))
    assert [row["grams"] for row in rows] == ["100", "200"] and rows[1]["walked"] == "0"


# The film page shows every attempt with how it ended
def test_film_page_lists_every_attempt():
    page = film_page("Duck 3.1", 150, [RockAttempt(20.0, False, 9, 9), RockAttempt(5.0, True, 2, 2)])
    assert "1 of 2 attempts walk the full 20 s" in page
    assert "attempt_2.mp4" in page and "falls after 5.0 s" in page and "walks all 20 s" in page
