"""A whole (tiny) training session: it practises, takes exams, keeps its best brain and can carry on."""
import json

import pytest

from studio import train
from studio.learning import clean_learning
from studio.replay import WalkerBrain
from studio.worlds import builtin_worlds


@pytest.fixture()
def tiny(monkeypatch):
    def tiny_learning(values):
        cleaned = clean_learning(values)
        cleaned.update(million_steps=0.002, parallel_worlds=1, steps_per_update=256, batch_size=64, epochs=2)
        return cleaned
    monkeypatch.setattr(train, "clean_learning", tiny_learning)


def write_setup(folder, animal="puppy", world_index=1, **brain):
    folder.mkdir(parents=True)
    setup = {"id": folder.name, "name": "tiny", "creature": {"animal": animal, "values": {}},
             "world": builtin_worlds()[world_index].to_dict(), "brain": {"layers": [16, 16], "values": brain},
             "learning": {}}
    (folder / "setup.json").write_text(json.dumps(setup))


def test_a_tiny_session_writes_everything_the_page_needs(tmp_path, tiny):
    folder = tmp_path / "run"
    write_setup(folder, see_ground=True)
    train.run(folder, resume=False)
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "finished"
    assert status["steps"] >= 2000
    for name in ("progress.csv", "exams.csv", "brain/best.zip", "brain/best_norm.npz", "brain/latest.zip",
                 "replays/best.json"):
        assert (folder / name).exists(), name
    replay = json.loads((folder / "replays/best.json").read_text())
    assert replay["shapes"] and replay["frames"] and replay["ground"]["kind"] == "heights"
    brain = WalkerBrain(folder / "brain/best.zip", folder / "brain/best_norm.npz")
    assert brain.act(__import__("numpy").zeros(brain.mean.shape)).shape == (14,)

    # Carrying on adds to the same session
    status["total_steps"] = 0
    train.run(folder, resume=True)
    assert json.loads((folder / "status.json").read_text())["state"] == "finished"


def test_a_stop_request_pauses_the_session(tmp_path, tiny):
    folder = tmp_path / "run"
    write_setup(folder, animal="duck", world_index=0)
    (folder / "stop.request").write_text("stop")
    original = train.Folder.stop_requested
    calls = {"count": 0}

    def stop_after_a_while(self):
        calls["count"] += 1
        return calls["count"] > 3

    train.Folder.stop_requested = property(stop_after_a_while)
    try:
        train.run(folder, resume=False)
    finally:
        train.Folder.stop_requested = original
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "stopped"
    assert (folder / "brain/latest.zip").exists()


def test_a_new_session_can_start_from_another_walkers_brain(tmp_path, tiny, capsys):
    first = tmp_path / "first"
    write_setup(first, animal="duck", world_index=0)
    train.run(first, resume=False)
    second = tmp_path / "second"
    write_setup(second, animal="duck", world_index=3)  # rolling hills
    setup = json.loads((second / "setup.json").read_text())
    setup["start_from"] = "first"
    (second / "setup.json").write_text(json.dumps(setup))
    capsys.readouterr()
    train.run(second, resume=False)
    assert "Starting from the best brain of first" in capsys.readouterr().out
    first_brain = WalkerBrain(first / "brain/best.zip", first / "brain/best_norm.npz")
    second_brain = WalkerBrain(second / "brain/latest.zip", second / "brain/latest_norm.npz")
    assert first_brain.mean.shape == second_brain.mean.shape
    assert json.loads((second / "status.json").read_text())["state"] == "finished"


def test_a_trained_walker_can_take_a_challenge_level(tmp_path, tiny, monkeypatch):
    from studio import challenges, library
    monkeypatch.setattr(library, "LIBRARY", tmp_path)
    folder = tmp_path / "trainings" / "walker"
    write_setup(folder, animal="duck", world_index=0)
    train.run(folder, resume=False)
    for challenge in ("speed", "stones", "stairs", "ice", "backpack", "wind", "gravity"):
        level = next(c for c in challenges.CHALLENGES if c["key"] == challenge)["levels"][0]
        result = challenges.run_level("walker", challenge, level, keep_replay=True)
        assert result["replay"]["frames"] and isinstance(result["passed"], bool)
