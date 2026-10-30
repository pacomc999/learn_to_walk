"""Tests for the rough floor height maps and the height field model."""
import time

import mujoco
import numpy as np
import pytest

from duck.floor import (
    FLOOR_COLS, FLOOR_ROWS, MAX_ROUGHNESS, floor_heights, floor_index,
    highest_floor_near, load_floor_model, set_floor,
)


def test_zero_roughness_is_flat():
    assert np.all(floor_heights(np.random.default_rng(0), 0.0) == 0.0)


def test_highest_point_is_the_roughness():
    heights = floor_heights(np.random.default_rng(0), 0.006)
    assert heights.min() == pytest.approx(0.0) and heights.max() == pytest.approx(0.006)


def test_same_seed_same_layout_and_new_seed_new_layout():
    a = floor_heights(np.random.default_rng(1), MAX_ROUGHNESS)
    assert np.array_equal(a, floor_heights(np.random.default_rng(1), MAX_ROUGHNESS))
    assert not np.array_equal(a, floor_heights(np.random.default_rng(2), MAX_ROUGHNESS))


def test_floor_model_has_a_height_field_named_floor():
    model = load_floor_model()
    assert model.geom("floor").type == mujoco.mjtGeom.mjGEOM_HFIELD
    assert model.hfield_size[0][2] == pytest.approx(MAX_ROUGHNESS)


def test_set_floor_round_trips():
    model = load_floor_model()
    heights = floor_heights(np.random.default_rng(3), MAX_ROUGHNESS)
    set_floor(model, heights)
    assert model.hfield_data.max() == pytest.approx(1.0)


def test_highest_floor_near():
    heights = np.zeros((FLOOR_ROWS, FLOOR_COLS))
    heights[floor_index(1.0, 0.2)] = 0.008
    assert highest_floor_near(heights, 1.03, 0.2, radius=0.05) == pytest.approx(0.008)
    assert highest_floor_near(heights, 1.5, 0.2, radius=0.05) == 0.0


def test_floor_index_matches_the_mujoco_layout():
    # The cell index must land on a ray hit at the same height MuJoCo reports
    model = load_floor_model()
    heights = np.zeros((FLOOR_ROWS, FLOOR_COLS))
    heights[floor_index(3.0, -0.4)] = MAX_ROUGHNESS
    set_floor(model, heights)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    point = np.array([3.0, -0.4, 1.0])
    distance = mujoco.mj_ray(model, data, point, np.array([0.0, 0.0, -1.0]), None, 1, -1, np.zeros(1, dtype=np.int32))
    assert 1.0 - distance == pytest.approx(MAX_ROUGHNESS, abs=1e-4)


def test_one_layout_is_fast():
    start = time.perf_counter()
    floor_heights(np.random.default_rng(4), MAX_ROUGHNESS)
    assert time.perf_counter() - start < 0.2


# A roughness outside 0 to MAX_ROUGHNESS is refused with a plain message
def test_floor_heights_refuse_a_roughness_out_of_range():
    for roughness in (-0.001, MAX_ROUGHNESS * 2):
        with pytest.raises(ValueError, match="Roughness must be from 0"):
            floor_heights(np.random.default_rng(0), roughness)
    assert floor_heights(np.random.default_rng(0), MAX_ROUGHNESS).max() == pytest.approx(MAX_ROUGHNESS)


# The command line type accepts values in range and turns the rest into an argparse error
def test_roughness_argument_for_the_command_line():
    import argparse
    from duck.floor import roughness_argument
    assert roughness_argument("0.005") == 0.005
    for text in ("-1", "0.5", "abc"):
        with pytest.raises(argparse.ArgumentTypeError):
            roughness_argument(text)
    parser = argparse.ArgumentParser()
    parser.add_argument("--roughness", type=roughness_argument)
    with pytest.raises(SystemExit):
        parser.parse_args(["--roughness", "0.5"])
