from datetime import datetime, timedelta
import pytest
from duck.progress import (ProgressRow, append_progress, read_progress,
                           minutes_since_improvement, plot_progress)

START = datetime(2026, 10, 28, 20, 0)

def rows(rewards, minutes_apart=10):
    return [ProgressRow(START + timedelta(minutes=i * minutes_apart), i * 1000, r)
            for i, r in enumerate(rewards)]

def test_still_improving():
    assert minutes_since_improvement(rows([1, 2, 3])) == 0.0

def test_stalled_for_90_minutes():
    assert minutes_since_improvement(rows([5, 4, 4, 3, 4, 4, 4, 4, 4, 4])) == pytest.approx(90)

def test_no_rows_means_no_stall():
    assert minutes_since_improvement([]) == 0.0

def test_log_round_trip(tmp_path):
    append_progress(tmp_path, 2048, 12.5, START)
    append_progress(tmp_path, 4096, 15.0, START + timedelta(minutes=5))
    assert read_progress(tmp_path) == [
        ProgressRow(START, 2048, 12.5),
        ProgressRow(START + timedelta(minutes=5), 4096, 15.0)]

def test_chart_is_saved(tmp_path):
    for row in rows([1, 2, 3]):
        append_progress(tmp_path, row.timesteps, row.mean_reward, row.time)
    assert plot_progress(tmp_path).exists()
