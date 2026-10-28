"""Tests for the training dashboard."""
import json
import threading
import urllib.request
from datetime import datetime, timedelta

import imageio.v3 as iio
import numpy as np
import pytest

from duck.dashboard import (CLIP_NAME, frame_reading, lean_degrees, make_server,
                            recent_steps_per_second, run_status, save_mp4)
from duck.policy import BrainReading
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


# Upright means no lean; a turn of 10 degrees about the sideways axis tips the nose down 10 degrees
def test_lean_degrees():
    assert lean_degrees(np.array([1.0, 0, 0, 0])) == pytest.approx(0)
    half = np.radians(10) / 2
    assert lean_degrees(np.array([np.cos(half), 0, np.sin(half), 0])) == pytest.approx(10)


# One frame of brain data: senses in plain units, brain cells as whole numbers from -100 to 100
def test_frame_reading():
    observation = np.zeros(39, dtype=np.float32)
    observation[2] = 1.0                      # upright quaternion
    observation[6] = 0.3                      # forward speed
    observation[36:39] = [1.0, 0.0, 0.4]      # left foot down, right foot up, target 0.4
    reading = BrainReading(action=np.full(12, 0.5), layer1=np.full(256, -0.5), layer2=np.ones(256))
    frame = frame_reading(lambda obs: reading, observation)
    assert frame["senses"] == {"left_foot": True, "right_foot": False, "speed": 0.3,
                               "target": 0.4, "lean": 0.0}
    assert frame["layer1"][0] == -50 and frame["layer2"][0] == 100
    assert frame["motors"] == [0.5] * 12


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
