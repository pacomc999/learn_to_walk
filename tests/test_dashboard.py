"""Tests for the training dashboard."""
import json
import threading
import urllib.request
from datetime import datetime, timedelta

import imageio.v3 as iio
import numpy as np
import pytest

from duck.dashboard import (CLIP_NAME, CLIP_STEPS_NAME, brain_to_film, make_server, recent_steps_per_second,
                            run_status, save_mp4)
from duck.exam import ExamResult, FullExamResult, append_exam
from duck.progress import ProgressRow, append_progress

START = datetime(2026, 10, 28, 20, 0)


# 1,000 steps every 10 seconds is 100 steps per second
def test_recent_steps_per_second():
    rows = [ProgressRow(START + timedelta(seconds=10 * i), 1000 * i, 0.0, 0.0) for i in range(5)]
    assert recent_steps_per_second(rows) == pytest.approx(100)


# Speed needs at least two rows
def test_no_speed_from_one_row():
    assert recent_steps_per_second([ProgressRow(START, 1000, 0.0, 0.0)]) is None


# A run with no log yet still gives a status the page can show
def test_status_of_an_empty_run(tmp_path):
    status = run_status(tmp_path, total_steps=3_000_000)
    assert status["points"] == []
    assert status["latest_steps"] is None
    assert status["minutes_left"] is None
    assert status["clip_exam"] is None


# Halfway at 100 steps per second, 1,500 steps remain: 15 seconds, so 0.25 minutes
def test_status_with_progress(tmp_path):
    for i in range(4):
        append_progress(tmp_path, 500 * i, 10.0 * i, 0.5 * i, START + timedelta(seconds=5 * i))
    status = run_status(tmp_path, total_steps=3000)
    assert status["latest_steps"] == 1500
    assert status["percent_done"] == pytest.approx(50)
    assert status["minutes_left"] == pytest.approx(0.25)
    assert status["points"][-1] == {"steps": 1500, "score": 30.0, "distance": 1.5}


# Without a total step count there is no percentage or time left
def test_status_without_total(tmp_path):
    append_progress(tmp_path, 500, 1.0, 0.1, START)
    append_progress(tmp_path, 1000, 2.0, 0.2, START + timedelta(seconds=5))
    status = run_status(tmp_path, total_steps=None)
    assert status["percent_done"] is None and status["minutes_left"] is None


# A list of pictures becomes a video with one frame per picture
def test_save_mp4(tmp_path):
    frames = [np.full((48, 64, 3), shade, dtype=np.uint8) for shade in (0, 120, 240)]
    path = tmp_path / CLIP_NAME
    save_mp4(frames, path, fps=10)
    assert len(iio.imread(path)) == 3


# The server answers the page, the status, a missing video and pieces of a video
def test_server_routes(tmp_path):
    append_progress(tmp_path, 500, 1.0, 0.1, START)
    server = make_server(tmp_path, total_steps=None, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as page:
            assert b"<html" in page.read()
        with urllib.request.urlopen(base + "/status.json") as answer:
            assert json.load(answer)["latest_steps"] == 500
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/latest.mp4")
        # A video player asks for pieces of the file; the server sends just that piece
        (tmp_path / CLIP_NAME).write_bytes(b"0123456789")
        request = urllib.request.Request(base + "/latest.mp4", headers={"Range": "bytes=2-5"})
        with urllib.request.urlopen(request) as piece:
            assert piece.status == 206 and piece.read() == b"2345"
    finally:
        server.shutdown()
        server.server_close()


# Empty brain and statistics files, enough for the snapshot list to count a snapshot
def _fake_snapshot(folder, steps):
    snapshots = folder / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    (snapshots / f"snapshot_{steps:012d}.zip").touch()
    (snapshots / f"snapshot_{steps:012d}_norm.pkl").touch()


# Without exams yet, the film shows the newest snapshot
def test_films_newest_snapshot_without_a_best(tmp_path):
    _fake_snapshot(tmp_path, 0)
    _fake_snapshot(tmp_path, 1_000_000)
    assert brain_to_film(tmp_path).steps == 1_000_000


# Once the exams have kept a best brain, the film shows it, even if a newer snapshot exists
def test_films_the_best_brain(tmp_path):
    _fake_snapshot(tmp_path, 2_000_000)
    _fake_snapshot(tmp_path / "best", 1_750_000)
    assert brain_to_film(tmp_path).steps == 1_750_000


# The status carries the exam result of the brain in the film
def test_status_has_the_exam_of_the_filmed_brain(tmp_path):
    half = ExamResult(916.0, 0, 9.2)
    append_exam(tmp_path, 1_750_000, FullExamResult(916.0, 0, 9.2, 24, {"lower": half, "taller": half}), START)
    (tmp_path / CLIP_STEPS_NAME).write_text("1750000")
    assert run_status(tmp_path, total_steps=None)["clip_exam"]["score"] == 916.0


# A new style exam row gives the status both parts and the attempt count
def test_status_has_the_parts_of_a_new_exam(tmp_path):
    full = FullExamResult(900.0, 1, 9.0, 24, {"lower": ExamResult(950.0, 0, 9.0), "taller": ExamResult(850.0, 1, 9.0)})
    append_exam(tmp_path, 2_500_000, full, START)
    (tmp_path / CLIP_STEPS_NAME).write_text("2500000")
    exam = run_status(tmp_path, total_steps=None)["clip_exam"]
    assert (exam["attempts"], exam["lower_score"], exam["taller_score"]) == (24, 950.0, 850.0)


# An old row (no parts in the log) gives None parts and 12 attempts
def test_status_has_no_parts_for_an_old_exam(tmp_path):
    (tmp_path / "exam.csv").write_text("time,timesteps,score,falls,distance\n2026-10-28 20:00:00,1000000,500.0,3,8.0\n")
    (tmp_path / CLIP_STEPS_NAME).write_text("1000000")
    exam = run_status(tmp_path, total_steps=None)["clip_exam"]
    assert (exam["attempts"], exam["flat_score"], exam["rough_score"]) == (12, None, None)


# The dashboard films on the training floor (flat when training is flat), on a fixed layout from seed 1000
def test_dashboard_film_uses_the_training_floor(tmp_path, monkeypatch):
    import duck.dashboard
    from duck.exam import EXAM_FIRST_SEED
    from duck.floor import TRAINING_ROUGHNESS
    seen = {}

    def fake_record_clip(policy, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(duck.dashboard, "record_clip", fake_record_clip)
    monkeypatch.setattr(duck.dashboard, "load_policy", lambda snapshot: None)
    with pytest.raises(RuntimeError):
        duck.dashboard.make_snapshot_clip(tmp_path, None)
    assert seen["roughness"] == TRAINING_ROUGHNESS and seen["seed"] == EXAM_FIRST_SEED
