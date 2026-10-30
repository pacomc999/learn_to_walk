import subprocess
import sys

from tests.test_train import FLAT_SETTINGS
from tests.worlds_for_tests import BODY, FLAT
from duck.evaluate import judge_attempt, evaluate_snapshot
from duck.snapshots import latest_snapshot
from duck.train import train


def test_judge_attempt():
    assert judge_attempt(False, 20.0, 5.0, 0.1)
    assert not judge_attempt(True, 20.0, 5.0, 0.1)    # fell
    assert not judge_attempt(False, 20.0, 3.0, 0.1)   # 0.15 m/s is too slow
    assert not judge_attempt(False, 20.0, 5.0, -0.6)  # drifted off the line


def test_evaluate_a_real_snapshot(tmp_path):
    train(tmp_path, total_steps=512, snapshot_every=512, n_envs=2, settings=FLAT_SETTINGS,
          use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})
    result = evaluate_snapshot(latest_snapshot(tmp_path), FLAT, BODY, attempts=2)
    assert result.attempts == 2 and 0 <= result.successes <= 2


def test_cli_without_snapshots_exits_with_message(tmp_path):
    for module in ("duck.evaluate", "duck.view_snapshot"):
        run = subprocess.run([sys.executable, "-m", module, str(tmp_path)],
                             capture_output=True, text=True)
        assert run.returncode == 1
        assert f"No snapshots found in {tmp_path}" in run.stdout
        assert "Traceback" not in run.stderr


def test_cli_with_unknown_snapshot_lists_available_steps(tmp_path):
    train(tmp_path, total_steps=512, snapshot_every=512, n_envs=2, settings=FLAT_SETTINGS,
          use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})
    run = subprocess.run([sys.executable, "-m", "duck.evaluate", str(tmp_path), "--snapshot", "123"],
                         capture_output=True, text=True)
    assert run.returncode == 1
    assert "No snapshot with 123 steps" in run.stdout
    assert "512" in run.stdout
    assert "Traceback" not in run.stderr


# evaluate_snapshot is flat by default and passes a chosen roughness to the environment
def test_evaluate_passes_the_roughness(monkeypatch):
    import duck.evaluate
    import pytest
    seen = {}

    def fake_env(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(duck.evaluate, "DuckWalkEnv", fake_env)
    monkeypatch.setattr(duck.evaluate, "load_policy", lambda snapshot: None)
    with pytest.raises(RuntimeError):
        evaluate_snapshot(None, FLAT, BODY)
    assert seen["roughness"] == 0.0 and seen["world"] == FLAT and seen["body"] == BODY
    with pytest.raises(RuntimeError):
        evaluate_snapshot(None, FLAT, BODY, roughness=0.01)
    assert seen["roughness"] == 0.01
