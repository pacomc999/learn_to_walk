import json

import pytest

from duck import catalogue
from duck.catalogue import keep_brain, new_duck, write_index
from duck.snapshots import list_snapshots
from duck.train import DEFAULT_PPO_SETTINGS, train
from tests.test_train import FAST


# Short GIFs and few gait attempts keep these tests quick
@pytest.fixture(autouse=True)
def quick_measurements(monkeypatch):
    monkeypatch.setattr(catalogue, "GIF_SECONDS", 0.4)
    monkeypatch.setattr(catalogue, "GAIT_SEEDS", (0,))


# A tiny trained run with snapshots at 0 and 512 steps
@pytest.fixture
def small_run(tmp_path):
    run_dir = tmp_path / "runs" / "small"
    train(run_dir, total_steps=512, snapshot_every=512, exam_every=10_000, **FAST)
    return run_dir


# A new duck freezes the current recipe and gets a card
def test_new_duck_freezes_the_recipe(tmp_path):
    duck_dir = new_duck(tmp_path / "ducks" / "duck_01", "Duck 1")
    story = json.loads((duck_dir / "duck.json").read_text())
    assert story["name"] == "Duck 1"
    assert story["recipe"]["brain"] == "SymmetricPolicy"
    assert story["recipe"]["ppo"] == DEFAULT_PPO_SETTINGS
    assert story["recipe"]["reward"]["energy"] == 1.5
    assert "# Duck 1" in (duck_dir / "card.md").read_text()


# Two ducks can not share a folder
def test_new_duck_refuses_a_used_folder(tmp_path):
    new_duck(tmp_path / "duck_01", "Duck 1")
    with pytest.raises(FileExistsError):
        new_duck(tmp_path / "duck_01", "Duck 1 again")


# Keeping a brain copies it, measures it, makes a GIF and fills the card
def test_keep_brain_fills_the_duck_folder(tmp_path, small_run):
    duck_dir = new_duck(tmp_path / "ducks" / "duck_01", "Duck 1")
    snapshot = list_snapshots(small_run)[-1]
    assert keep_brain(duck_dir, small_run, snapshot)
    kept = list_snapshots(duck_dir / "brain")
    assert [snap.steps for snap in kept] == [snapshot.steps]
    results = json.loads((duck_dir / "results.json").read_text())
    assert results["steps"] == snapshot.steps and 0 <= results["exam_falls"] <= 12
    assert (duck_dir / "duck_01.gif").stat().st_size > 0
    assert "Exam (12 fixed attempts" in (duck_dir / "card.md").read_text()


# A brain that does not beat the kept one is not kept, unless forced
def test_keep_brain_only_keeps_a_better_one(tmp_path, small_run):
    duck_dir = new_duck(tmp_path / "ducks" / "duck_01", "Duck 1")
    first, second = list_snapshots(small_run)[0], list_snapshots(small_run)[-1]
    keep_brain(duck_dir, small_run, first)
    results = json.loads((duck_dir / "results.json").read_text())
    results["exam_score"] = 1e9  # pretend the kept brain is unbeatable
    (duck_dir / "results.json").write_text(json.dumps(results))
    assert not keep_brain(duck_dir, small_run, second)
    assert [snap.steps for snap in list_snapshots(duck_dir / "brain")] == [first.steps]
    assert keep_brain(duck_dir, small_run, second, force=True)
    assert [snap.steps for snap in list_snapshots(duck_dir / "brain")] == [second.steps]


# The list page has one row per duck, oldest folder first
def test_index_lists_every_duck(tmp_path):
    ducks = tmp_path / "ducks"
    new_duck(ducks / "duck_02", "Duck 2")
    new_duck(ducks / "duck_01", "Duck 1")
    index = write_index(ducks).read_text()
    assert index.index("[Duck 1]") < index.index("[Duck 2]")
    assert "none yet" in index
