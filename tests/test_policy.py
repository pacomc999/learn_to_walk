"""Tests for loading a snapshot as a policy and reading its brain."""
import numpy as np
import pytest

from duck.duck_env import DuckWalkEnv
from duck.policy import load_brain, load_policy
from duck.snapshots import latest_snapshot
from duck.train import train
from tests.test_train import save_plain_run


# An untrained symmetric snapshot is enough: the readout must agree with the policy whatever the weights
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


# The readout picks exactly the action the policy picks, and shows both layers of 256 cells
def check_brain_matches_policy(snapshot):
    observation, _ = DuckWalkEnv().reset(seed=0)
    reading = load_brain(snapshot)(observation)
    assert reading.action == pytest.approx(load_policy(snapshot)(observation), abs=1e-5)
    assert reading.layer1.shape == (256,) and reading.layer2.shape == (256,)
    assert np.all(np.abs(reading.layer1) <= 1) and np.all(np.abs(reading.layer2) <= 1)


# The symmetric brain's readout matches its policy
def test_symmetric_brain_matches_policy(symmetric_snapshot):
    check_brain_matches_policy(symmetric_snapshot)


# An old plain brain's readout still matches its policy
def test_plain_brain_matches_policy(plain_snapshot):
    check_brain_matches_policy(plain_snapshot)
