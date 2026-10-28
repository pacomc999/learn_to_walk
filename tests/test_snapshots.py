"""Tests for saving, finding and choosing snapshot files."""
import os
from pathlib import Path
import pytest
import duck.snapshots as snapshots_module
from duck.snapshots import (snapshot_paths, save_snapshot, list_snapshots,
                            latest_snapshot, choose_for_video)


# A fake brain or statistics object that just writes a small file
class FakeSaver:
    def save(self, path):
        Path(path).write_text("x")


# Once saving is done, only the final files should be left in the folder
def test_saving_leaves_only_final_files(tmp_path):
    snap = save_snapshot(FakeSaver(), FakeSaver(), tmp_path, 2048)
    assert snap.model_path.exists() and snap.norm_path.exists()
    assert not list((tmp_path / "snapshots").glob("*.tmp*"))


# A snapshot with only the brain saved (like a crash mid-save) must be skipped
def test_half_written_snapshot_is_ignored(tmp_path):
    save_snapshot(FakeSaver(), FakeSaver(), tmp_path, 1000)
    broken = snapshot_paths(tmp_path, 2000)
    broken.model_path.write_text("x")  # brain saved, statistics missing
    assert [s.steps for s in list_snapshots(tmp_path)] == [1000]
    assert latest_snapshot(tmp_path).steps == 1000


# Snapshots come back in training order, not in the order they were saved
def test_snapshots_sorted_by_steps(tmp_path):
    for steps in (3000, 0, 1000):
        save_snapshot(FakeSaver(), FakeSaver(), tmp_path, steps)
    assert [s.steps for s in list_snapshots(tmp_path)] == [0, 1000, 3000]


# An empty run has no snapshots and no latest snapshot
def test_no_snapshots(tmp_path):
    assert list_snapshots(tmp_path) == []
    assert latest_snapshot(tmp_path) is None


# Long runs are cut to 25 clips spread evenly, keeping the first and last
def test_choose_for_video_spreads_evenly():
    snaps = [snapshot_paths(Path("run"), i * 1000) for i in range(100)]
    chosen = choose_for_video(snaps)
    assert len(chosen) == 25
    assert chosen[0] == snaps[0] and chosen[-1] == snaps[-1]
    assert choose_for_video(snaps[:3]) == snaps[:3]
    assert choose_for_video(snaps[:1]) == snaps[:1]


# A file briefly locked by Windows (for example by a virus scanner) is retried, not fatal
def test_locked_file_is_retried(tmp_path, monkeypatch):
    real_replace = os.replace
    failures_left = [2]
    def flaky_replace(source, target):
        if failures_left[0] > 0:
            failures_left[0] -= 1
            raise PermissionError("locked")
        real_replace(source, target)
    monkeypatch.setattr(snapshots_module.os, "replace", flaky_replace)
    monkeypatch.setattr(snapshots_module.time, "sleep", lambda seconds: None)
    snap = save_snapshot(FakeSaver(), FakeSaver(), tmp_path, 2048)
    assert snap.model_path.exists() and snap.norm_path.exists()


# A file that stays locked still raises an error after a few tries
def test_file_locked_for_good_raises(tmp_path, monkeypatch):
    tries = []
    def locked_replace(source, target):
        tries.append(1)
        raise PermissionError("locked")
    monkeypatch.setattr(snapshots_module.os, "replace", locked_replace)
    monkeypatch.setattr(snapshots_module.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        save_snapshot(FakeSaver(), FakeSaver(), tmp_path, 2048)
    assert len(tries) == 5
