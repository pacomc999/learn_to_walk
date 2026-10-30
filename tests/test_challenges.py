"""Tests for the challenges."""
import numpy as np
import pytest

from duck.challenges import (CHALLENGES, BackpackWorld, TallStoneWorld, _fall_seconds, challenge_row, film_value,
                             write_film_page, write_readme, write_table)
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


# A fall shows up as the first fallen picture; the very last picture is only the end of the attempt
def test_fall_seconds():
    assert _fall_seconds([False] * 30 + [True] * 10) == 1.0
    assert _fall_seconds([False] * 599 + [True]) is None
    assert _fall_seconds([False] * 10) is None


# A short film of one value is saved, and the challenge's page lists it with its label and outcome
def test_film_value_and_page(tmp_path):
    folder = tmp_path / "challenges" / "backpack" / "films"
    film_value("backpack", 200, stand_still, folder / "200.mp4", "Duck 9", seconds=0.2)
    assert (folder / "200.mp4").stat().st_size > 0
    (folder / "outcomes.json").write_text('{"200": null, "800": 5.0}')
    page = write_film_page(tmp_path, "backpack").read_text(encoding="utf-8")
    assert '"200.mp4"' in page and "200 g backpack" in page and "walks all 20 s" in page
    assert '"800.mp4"' in page and "falls after 5.0 s" in page
    write_table([{"backpack g": 200, "walked": 10}], tmp_path / "challenges" / "backpack" / "results.csv")
    assert "`backpack/films/index.html`" in write_readme(tmp_path, "Duck 9").read_text(encoding="utf-8")
