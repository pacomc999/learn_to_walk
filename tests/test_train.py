import pytest
import torch

from duck.train import load_latest_snapshot, train
from duck.snapshots import list_snapshots
from duck.progress import read_progress

FAST = dict(n_envs=2, use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})

def test_new_run_saves_untrained_regular_and_final(tmp_path):
    train(tmp_path, total_steps=2048, snapshot_every=1024, **FAST)
    steps = [s.steps for s in list_snapshots(tmp_path)]
    assert steps[0] == 0
    assert 1024 in steps
    assert steps[-1] >= 2048

def test_resume_continues_with_more_curiosity(tmp_path):
    train(tmp_path, total_steps=1024, snapshot_every=1024, **FAST)
    model = train(tmp_path, total_steps=2048, snapshot_every=1024, ent_coef=0.02, resume=True, **FAST)
    assert model.num_timesteps >= 2048
    assert model.ent_coef == 0.02
    assert list_snapshots(tmp_path)[-1].steps >= 2048

def test_zero_steps_saves_only_untrained_snapshot(tmp_path):
    train(tmp_path, total_steps=0, **FAST)
    assert [s.steps for s in list_snapshots(tmp_path)] == [0]

def test_brain_has_two_layers_of_256(tmp_path):
    model = train(tmp_path, total_steps=0, **FAST)
    assert model.policy.net_arch == {"pi": [256, 256], "vf": [256, 256]}

def test_progress_is_logged(tmp_path):
    train(tmp_path, total_steps=4096, snapshot_every=4096, **FAST)
    logged = read_progress(tmp_path)
    assert len(logged) >= 1
    assert all(abs(row.mean_distance) < 10 for row in logged)  # a real distance in metres

def test_new_run_in_used_folder_is_refused(tmp_path):
    train(tmp_path, total_steps=0, **FAST)
    first = list_snapshots(tmp_path)[0]
    saved_time = first.model_path.stat().st_mtime_ns
    with pytest.raises(FileExistsError, match="Use --resume"):
        train(tmp_path, total_steps=0, **FAST)
    assert first.model_path.stat().st_mtime_ns == saved_time

def test_resumed_brain_keeps_its_learning(tmp_path):
    trained = train(tmp_path, total_steps=1024, snapshot_every=1024, **FAST)
    loaded, vec_normalize = load_latest_snapshot(tmp_path, n_envs=2, use_subprocess=False)
    vec_normalize.close()
    assert loaded.num_timesteps == list_snapshots(tmp_path)[-1].steps
    trained_weights = trained.policy.state_dict()
    for name, weights in loaded.policy.state_dict().items():
        assert torch.equal(weights, trained_weights[name]), name


# A new brain starts with calmer random moves (log_std of -1 for every motor)
def test_new_brain_starts_with_log_std_minus_one(tmp_path):
    model = train(tmp_path, total_steps=0, **FAST)
    assert torch.allclose(model.policy.log_std, torch.full((12,), -1.0))

# A broken newest snapshot is skipped with a warning, and the one before it is used
def test_corrupt_newest_snapshot_falls_back(tmp_path, capsys):
    train(tmp_path, total_steps=1024, snapshot_every=1024, **FAST)
    previous, newest = list_snapshots(tmp_path)[-2:]
    newest.model_path.write_bytes(b"not a real zip file")
    loaded, vec_normalize = load_latest_snapshot(tmp_path, n_envs=2, use_subprocess=False)
    vec_normalize.close()
    assert loaded.num_timesteps == previous.steps
    assert newest.model_path.name in capsys.readouterr().out
    assert list_snapshots(tmp_path)[-1] == previous  # the broken one no longer counts
