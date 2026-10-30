import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from duck.mirror import MirroredRunningMeanStd
from duck.symmetric_policy import SymmetricPolicy
from duck.train import (DEFAULT_PPO_SETTINGS, NET_ARCH, START_LOG_STD, _save_snapshot_once,
                        build_vec_env, load_latest_snapshot, train)
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


# New runs get the left/right symmetric brain
def test_new_run_uses_the_symmetric_brain(tmp_path):
    model = train(tmp_path, total_steps=0, **FAST)
    assert isinstance(model.policy, SymmetricPolicy)


# A symmetric run comes back as a symmetric brain with the same weights and mirrored scaling, and keeps training
def test_symmetric_run_resumes_with_the_same_brain(tmp_path):
    trained = train(tmp_path, total_steps=1024, snapshot_every=1024, **FAST)
    loaded, vec_normalize = load_latest_snapshot(tmp_path, n_envs=2, use_subprocess=False)
    vec_normalize.close()
    assert isinstance(loaded.policy, SymmetricPolicy)
    assert isinstance(vec_normalize.obs_rms, MirroredRunningMeanStd)
    trained_weights = trained.policy.state_dict()
    for name, weights in loaded.policy.state_dict().items():
        assert torch.equal(weights, trained_weights[name]), name
    resumed = train(tmp_path, total_steps=2048, snapshot_every=1024, resume=True, **FAST)
    assert resumed.num_timesteps >= 2048
    assert isinstance(resumed.policy, SymmetricPolicy)


# Build an old style run (plain MlpPolicy, plain-class brain) and save it as the step 0 snapshot.
# It uses build_vec_env, so its statistics are mirrored ones, which is fine for this test
def save_plain_run(run_dir):
    vec_normalize = build_vec_env(2, seed=0, use_subprocess=False)
    settings = {**DEFAULT_PPO_SETTINGS, "n_steps": 256, "batch_size": 64}
    model = PPO("MlpPolicy", vec_normalize, device="cpu", seed=0, verbose=0,
                policy_kwargs={"net_arch": NET_ARCH, "log_std_init": START_LOG_STD}, **settings)
    _save_snapshot_once(model, vec_normalize, run_dir, steps=0)
    vec_normalize.close()


# An old plain snapshot keeps its plain brain when resumed
def test_plain_snapshot_still_resumes(tmp_path):
    save_plain_run(tmp_path)
    model = train(tmp_path, total_steps=1024, snapshot_every=1024, resume=True, **FAST)
    assert model.num_timesteps >= 1024
    assert isinstance(model.policy, ActorCriticPolicy)
    assert not isinstance(model.policy, SymmetricPolicy)


# The step size shrinks after a too big change, grows after a tiny one, and stays within its limits
def test_adjusted_learning_rate_follows_the_rule():
    from duck.train import STEP_SIZE_RULE, adjusted_learning_rate
    target, factor = STEP_SIZE_RULE["target_change"], STEP_SIZE_RULE["factor"]
    assert adjusted_learning_rate(3e-4, change=3 * target) == pytest.approx(3e-4 / factor)
    assert adjusted_learning_rate(3e-4, change=target / 3) == pytest.approx(3e-4 * factor)
    assert adjusted_learning_rate(3e-4, change=target) == 3e-4  # close enough: unchanged
    assert adjusted_learning_rate(1e-5, change=1.0) == STEP_SIZE_RULE["min_learning_rate"]
    assert adjusted_learning_rate(1e-2, change=0.0) == STEP_SIZE_RULE["max_learning_rate"]


# Training changes the learning rate, the optimizer uses it, and a resumed run carries on from it
def test_adapted_learning_rate_is_used_and_resumed(tmp_path):
    import csv
    model = train(tmp_path, total_steps=1536, snapshot_every=1536, exam_every=10_000, **FAST)
    with (tmp_path / "training_stats.csv").open(newline="") as log_file:
        rates = [float(row["learning_rate"]) for row in csv.DictReader(log_file)]
    assert rates[0] == DEFAULT_PPO_SETTINGS["learning_rate"]
    assert len(set(rates)) > 1  # the rate moved between updates
    saved_rate = model.learning_rate
    resumed = train(tmp_path, total_steps=1536, resume=True, **FAST)  # nothing left to learn: just loads
    assert resumed.learning_rate == pytest.approx(saved_rate)


# The files of a folder with their modification times, to prove nothing in it changed
def folder_fingerprint(folder):
    return {str(path): path.stat().st_mtime_ns for path in sorted(folder.rglob("*")) if path.is_file()}


# A new run starts from another run's newest brain, counts from 0 and leaves the source alone
def test_start_from_another_runs_brain(tmp_path):
    import csv
    source, target = tmp_path / "source", tmp_path / "target"
    train(source, total_steps=512, snapshot_every=512, exam_every=10_000, **FAST)
    source_newest = list_snapshots(source)[-1]
    before = folder_fingerprint(source)

    train(target, total_steps=0, start_from=source, **FAST)
    assert [s.steps for s in list_snapshots(target)] == [0]
    started = PPO.load(str(list_snapshots(target)[0].model_path), device="cpu")
    original = PPO.load(str(source_newest.model_path), device="cpu")
    for name, weights in started.policy.state_dict().items():
        assert torch.equal(weights, original.policy.state_dict()[name]), name
    assert started.learning_rate == pytest.approx(original.learning_rate)

    model = train(target, total_steps=256, snapshot_every=256, resume=True, **FAST)  # carries on from 0
    assert 256 <= model.num_timesteps <= 1024  # counted from 0, not from the source steps
    with (target / "exam.csv").open(newline="") as exam_file:
        assert int(next(csv.DictReader(exam_file))["timesteps"]) == 0
    assert folder_fingerprint(source) == before


# The first exam row of a started run is at 0 steps even when learning follows
def test_start_from_counts_steps_from_zero(tmp_path, capsys):
    import csv
    source, target = tmp_path / "source", tmp_path / "target"
    train(source, total_steps=0, **FAST)
    model = train(target, total_steps=512, snapshot_every=512, exam_every=10_000, start_from=source, **FAST)
    assert 512 <= model.num_timesteps < 1024
    with (target / "exam.csv").open(newline="") as exam_file:
        assert int(next(csv.DictReader(exam_file))["timesteps"]) == 0
    assert "Starting from the brain at 0 steps" in capsys.readouterr().out


def test_start_from_refuses_a_used_run_folder(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    train(source, total_steps=0, **FAST)
    train(target, total_steps=0, **FAST)
    with pytest.raises(FileExistsError, match="Use --resume"):
        train(target, total_steps=0, start_from=source, **FAST)


def test_start_from_and_resume_together_is_refused(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    train(source, total_steps=0, **FAST)
    with pytest.raises(ValueError, match="--start-from"):
        train(target, total_steps=0, start_from=source, resume=True, **FAST)
    assert not target.exists()


def test_start_from_an_empty_folder_is_refused(tmp_path):
    empty, target = tmp_path / "empty", tmp_path / "target"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No complete snapshot"):
        train(target, total_steps=0, start_from=empty, **FAST)
    assert list_snapshots(target) == []
