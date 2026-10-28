"""Save, find and pick snapshot files of the duck's brain during training.

A snapshot has two files: the brain (.zip) and the statistics (_norm.pkl).
A snapshot only counts as complete when both final files exist.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path


# One saved point in training: its step count and the two file paths
@dataclass(frozen=True)
class Snapshot:
    steps: int
    model_path: Path
    norm_path: Path


# The folder that holds all snapshot files of one training run
def _snapshot_folder(run_dir: Path) -> Path:
    return run_dir / "snapshots"


# The final file names for a snapshot at a given step count
def snapshot_paths(run_dir: Path, steps: int) -> Snapshot:
    folder = _snapshot_folder(run_dir)
    model_path = folder / f"snapshot_{steps:012d}.zip"
    norm_path = folder / f"snapshot_{steps:012d}_norm.pkl"
    return Snapshot(steps, model_path, norm_path)


# Save one file under a temp name, then move it onto its final name
def _save_via_temp(save_function, temp_path: Path, final_path: Path) -> None:
    save_function(str(temp_path))
    os.replace(temp_path, final_path)


# Save the brain and then the statistics, so the statistics file comes last
def save_snapshot(model, vec_normalize, run_dir: Path, steps: int) -> Snapshot:
    snap = snapshot_paths(run_dir, steps)
    snap.model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model = snap.model_path.with_name(f"snapshot_{steps:012d}.tmp.zip")
    temp_norm = snap.norm_path.with_name(f"snapshot_{steps:012d}_norm.tmp.pkl")
    _save_via_temp(model.save, temp_model, snap.model_path)
    _save_via_temp(vec_normalize.save, temp_norm, snap.norm_path)
    return snap


# Find the step count from a brain file name, or None if the name does not match
def _steps_from_model_name(file_name: str) -> int | None:
    match = re.fullmatch(r"snapshot_(\d{12})\.zip", file_name)
    if match is None:
        return None
    return int(match.group(1))


# All complete snapshots in a run, oldest first
def list_snapshots(run_dir: Path) -> list[Snapshot]:
    folder = _snapshot_folder(run_dir)
    if not folder.is_dir():
        return []
    snapshots = []
    for file in folder.iterdir():
        steps = _steps_from_model_name(file.name)
        if steps is None:
            continue
        snap = snapshot_paths(run_dir, steps)
        if snap.norm_path.exists():  # only complete snapshots count
            snapshots.append(snap)
    return sorted(snapshots, key=lambda snap: snap.steps)


# The newest complete snapshot, or None if there are none yet
def latest_snapshot(run_dir: Path) -> Snapshot | None:
    snapshots = list_snapshots(run_dir)
    return snapshots[-1] if snapshots else None


# Pick at most max_count snapshots spread evenly, always with the first and last
def choose_for_video(snapshots: list[Snapshot], max_count: int = 25) -> list[Snapshot]:
    if len(snapshots) <= max_count:
        return list(snapshots)
    last_index = len(snapshots) - 1
    step_indexes = [i * last_index // (max_count - 1) for i in range(max_count)]
    return [snapshots[i] for i in step_indexes]
