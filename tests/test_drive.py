"""Tests for driving a duck (the parts without windows)."""
import numpy as np
import pytest

from duck.drive import Driver, newest_brain


# Hold every motor at the standing pose
def stand_still(observation):
    return np.zeros(12)


# A new speed reaches the duck's senses at its next step
def test_new_speed_reaches_the_senses():
    driver = Driver(stand_still, 0.4)
    driver.set_speed(1.2)
    driver.step()
    assert driver.observation[-1] == pytest.approx(1.2)


# Reset stands the duck up again, keeping the speed set on the slider
def test_reset_keeps_the_speed_and_stands_up():
    driver = Driver(stand_still, 0.4)
    driver.set_speed(0.9)
    driver.fallen = True
    driver.reset()
    assert not driver.fallen and driver.env.target_speed == pytest.approx(0.9)
    assert driver.observation[-1] == pytest.approx(0.9)
    assert "Walking for 0.0 s" in driver.status()


# The duck keeps going past the 20 s end of a training attempt
def test_no_twenty_second_end():
    driver = Driver(stand_still, 0.4)
    for _ in range(1100):  # 22 s
        driver.step()
    assert driver.env.data.time > 20.0


# With no folder given, the newest saved brain is driven, from the runs or the catalogue
def test_newest_brain(tmp_path):
    import os
    assert newest_brain(tmp_path) is None
    for folder, age in (("runs/duck_09/a/best", 100), ("duck_catalogue/duck_09/brain", 50)):
        snapshots = tmp_path / folder / "snapshots"
        snapshots.mkdir(parents=True)
        for name in ("snapshot_000000001000.zip", "snapshot_000000001000_norm.pkl"):
            (snapshots / name).write_text("x")
            os.utime(snapshots / name, (age, age))
    assert newest_brain(tmp_path) == tmp_path / "runs/duck_09/a/best"
