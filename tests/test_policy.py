"""Tests for loading a snapshot as a policy."""
import numpy as np
import pytest

from duck.duck_env import DuckWalkEnv
from duck.policy import load_policy
from duck.snapshots import latest_snapshot
from duck.train import train
from tests.test_train import save_plain_run


# An untrained symmetric snapshot is enough: the checks hold whatever the weights
@pytest.fixture(scope="module")
def symmetric_snapshot(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    train(run_dir, total_steps=0, snapshot_every=1024, n_envs=2,
          use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})
    return latest_snapshot(run_dir)


# An untrained old style (plain brain) snapshot, built the same way as in the train tests
@pytest.fixture(scope="module")
def plain_snapshot(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("plain_run")
    save_plain_run(run_dir)
    return latest_snapshot(run_dir)


# The policy gives 12 motor commands within the motor limits, and the same ones every time
def check_policy(snapshot):
    observation, _ = DuckWalkEnv().reset(seed=0)
    policy = load_policy(snapshot)
    action = policy(observation)
    assert action.shape == (12,) and np.all(np.abs(action) <= 1)
    assert np.array_equal(action, policy(observation))


# A symmetric brain loads as a policy
def test_symmetric_brain_loads_as_policy(symmetric_snapshot):
    check_policy(symmetric_snapshot)


# An old plain brain still loads as a policy
def test_plain_brain_loads_as_policy(plain_snapshot):
    check_policy(plain_snapshot)
