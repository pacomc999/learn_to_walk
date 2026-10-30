"""Tests for the challenges."""
import numpy as np
import pytest

from duck.challenges import (CHALLENGES, BackpackWorld, TallStoneWorld, challenge_row, write_readme, write_table)
from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv


# Hold every motor at the standing pose
def stand_still(observation):
    return np.zeros(12)


# The backpack adds its weight to the duck but nothing to its senses
def test_backpack_adds_weight_not_senses():
    packed = BackpackWorld(0.2)
    plain = DuckWalkEnv(target_speed=EVALUATION_TARGET_SPEED, roughness=0.0)
    assert packed.model.body("torso").subtreemass[0] == pytest.approx(plain.model.body("torso").subtreemass[0] + 0.2)
    assert np.array_equal(packed.reset(seed=1)[0], plain.reset(seed=1)[0])


# A test floor can be taller than the training limit, and its stones reach the asked height
def test_tall_stones_reach_their_height():
    env = TallStoneWorld(0.05)
    env.reset(seed=0)
    assert env.floor.max() == pytest.approx(0.05)


# One row sums up its attempts; the rock rows also count hits and give the rock's size
def test_rows_sum_up_their_attempts():
    row = challenge_row(CHALLENGES["speed"], 0.4, stand_still, attempts=1)
    assert row["asked m/s"] == 0.4 and row["attempts"] == 1 and 0 <= row["walked"] <= 1
    rock_row = challenge_row(CHALLENGES["rocks"], 100, stand_still, attempts=1)
    assert rock_row["rock diameter cm"] == pytest.approx(4.2, abs=0.1) and rock_row["hits"] >= 0


# The README shows every saved table with its description, then the notes written by hand
def test_readme_collects_tables_and_notes(tmp_path):
    write_table([{"asked m/s": 0.4, "walked": 10}], tmp_path / "challenges" / "speed" / "results.csv")
    (tmp_path / "challenges" / "notes.md").write_text("It never runs.", encoding="utf-8")
    text = write_readme(tmp_path, "Duck 9").read_text(encoding="utf-8")
    assert "# Duck 9: challenges" in text and "## Speeds above training" in text
    assert "| asked m/s | walked |" in text and "| 0.4 | 10 |" in text
    assert "## Notes" in text and "It never runs." in text
    assert "Thrown rocks" not in text  # no table, no section
