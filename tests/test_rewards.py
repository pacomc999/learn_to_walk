"""Tests for the reward and fall rules."""
import numpy as np
import pytest
from duck.rewards import compute_reward, has_fallen, RewardWeights


# Moving forward, upright, with no effort: forward 1.0 plus alive 0.5
def test_moving_forward_upright_with_no_effort():
    total, parts = compute_reward(1.0, 0.0, np.zeros(12))
    assert total == pytest.approx(1.5)


# Effort is the sum of squared actions times 0.01: twelve 1s give 0.12
def test_effort_penalty_uses_squared_actions():
    total, parts = compute_reward(1.0, 0.0, np.ones(12))
    assert parts["effort"] == pytest.approx(-0.12)
    assert total == pytest.approx(1.38)


# Drifting left or right costs the same amount
def test_sideways_penalty_same_on_both_sides():
    left, _ = compute_reward(0.0, -0.4, np.zeros(12))
    right, _ = compute_reward(0.0, 0.4, np.zeros(12))
    assert left == right == pytest.approx(0.3)


# Walking backwards gives a negative forward score that cancels the alive bonus
def test_moving_backward_loses_points():
    total, _ = compute_reward(-0.5, 0.0, np.zeros(12))
    assert total == pytest.approx(0.0)


# The four parts must add up to the total reward
def test_parts_add_up_to_total():
    total, parts = compute_reward(0.7, 0.2, np.full(12, 0.5), RewardWeights())
    assert sum(parts.values()) == pytest.approx(total)


# Falling means too low, or too tilted (up_z below the limit)
def test_fall_rules():
    assert has_fallen(0.3, 1.0, min_height=0.4)
    assert has_fallen(0.6, 0.5, min_height=0.4)
    assert not has_fallen(0.6, 0.9, min_height=0.4)
