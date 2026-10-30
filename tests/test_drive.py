"""Tests for driving a duck (the parts without windows)."""
import numpy as np
import pytest

from duck.drive import Driver


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
