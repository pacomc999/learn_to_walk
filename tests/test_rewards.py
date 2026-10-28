"""Tests for the reward and fall rules."""
import pytest
from duck.rewards import compute_reward, has_fallen, RewardWeights


# Moving forward, upright, using no energy: forward 3.0 plus alive 0.2
def test_moving_forward_upright_with_no_energy():
    total, parts = compute_reward(1.0, 0.0, 0.0)
    assert total == pytest.approx(3.2)


# Energy costs 1.5 points per joule of motor work: 0.8 J costs 1.2 points
def test_energy_penalty_per_joule():
    total, parts = compute_reward(1.0, 0.0, 0.8)
    assert parts["energy"] == pytest.approx(-1.2)
    assert total == pytest.approx(2.0)


# Going faster than the target speed earns no extra forward points
def test_forward_points_stop_at_target_speed():
    _, slow = compute_reward(0.3, 0.0, 0.0, target_speed=0.4)
    _, fast = compute_reward(0.9, 0.0, 0.0, target_speed=0.4)
    assert slow["forward"] == pytest.approx(0.9)
    assert fast["forward"] == pytest.approx(1.2)


# Walking backwards still loses points when there is a target speed
def test_backwards_loses_points_with_target_speed():
    _, parts = compute_reward(-0.5, 0.0, 0.0, target_speed=0.4)
    assert parts["forward"] == pytest.approx(-1.5)


# Drifting left or right costs the same amount
def test_sideways_penalty_same_on_both_sides():
    left, _ = compute_reward(0.0, -0.4, 0.0)
    right, _ = compute_reward(0.0, 0.4, 0.0)
    assert left == right == pytest.approx(0.0)


# Walking backwards gives a negative forward score bigger than the alive bonus
def test_moving_backward_loses_points():
    total, _ = compute_reward(-0.5, 0.0, 0.0)
    assert total == pytest.approx(-1.3)


# The four parts must add up to the total reward
def test_parts_add_up_to_total():
    total, parts = compute_reward(0.7, 0.2, 0.3, RewardWeights())
    assert sum(parts.values()) == pytest.approx(total)


# Falling means too low, or too tilted (up_z below the limit)
def test_fall_rules():
    assert has_fallen(0.3, 1.0, min_height=0.4)
    assert has_fallen(0.6, 0.5, min_height=0.4)
    assert not has_fallen(0.6, 0.9, min_height=0.4)
