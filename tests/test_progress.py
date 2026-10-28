from datetime import datetime, timedelta
from pathlib import Path
import pytest
from duck.progress import (LOG_NAME, ProgressRow, append_progress, read_progress,
                           minutes_since_improvement, plot_progress, latest_distance_message)

START = datetime(2026, 10, 28, 20, 0)

def rows(rewards, minutes_apart=10):
    return [ProgressRow(START + timedelta(minutes=i * minutes_apart), i * 1000, r, r / 10)
            for i, r in enumerate(rewards)]

def test_still_improving():
    assert minutes_since_improvement(rows([1, 2, 3])) == 0.0

def test_stalled_for_90_minutes():
    assert minutes_since_improvement(rows([5, 4, 4, 3, 4, 4, 4, 4, 4, 4])) == pytest.approx(90)

def test_no_rows_means_no_stall():
    assert minutes_since_improvement([]) == 0.0

def test_log_round_trip(tmp_path):
    append_progress(tmp_path, 2048, 12.5, 0.25, START)
    append_progress(tmp_path, 4096, 15.0, 0.75, START + timedelta(minutes=5))
    assert read_progress(tmp_path) == [
        ProgressRow(START, 2048, 12.5, 0.25),
        ProgressRow(START + timedelta(minutes=5), 4096, 15.0, 0.75)]

# The log file starts with a header that names the distance column
def test_log_header_has_distance(tmp_path):
    append_progress(tmp_path, 2048, 12.5, 0.25, START)
    first_line = (tmp_path / LOG_NAME).read_text().splitlines()[0]
    assert first_line == "time,timesteps,mean_reward,mean_distance"

def test_chart_is_saved(tmp_path):
    for row in rows([1, 2, 3]):
        append_progress(tmp_path, row.timesteps, row.mean_reward, row.mean_distance, row.time)
    assert plot_progress(tmp_path).exists()

# A half written last line (for example after a crash) is skipped, not a crash
def test_half_written_row_is_skipped(tmp_path):
    append_progress(tmp_path, 2048, 12.5, 0.25, START)
    with (tmp_path / LOG_NAME).open("a", newline="") as log_file:
        log_file.write("2026-10-28T20:10:00,40")
    assert read_progress(tmp_path) == [ProgressRow(START, 2048, 12.5, 0.25)]

# A locked log file (open in another program) gives a warning, and training goes on
def test_locked_log_only_warns(tmp_path, monkeypatch, capsys):
    def locked_open(*args, **kwargs):
        raise PermissionError("locked")
    monkeypatch.setattr(Path, "open", locked_open)
    append_progress(tmp_path, 2048, 12.5, 0.25, START)
    assert "Could not write progress.csv" in capsys.readouterr().out

# The command line tells the latest average distance, or that nothing is logged yet
def test_latest_distance_message():
    assert latest_distance_message(rows([1, 2, 3])) == "Latest average distance per attempt: 0.30 metres"
    assert latest_distance_message([]) == "No progress logged yet."
