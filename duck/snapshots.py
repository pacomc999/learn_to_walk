"""Save, find and pick snapshot files of the duck's brain during training.

A snapshot has two files: the brain (.zip) and the statistics (_norm.pkl).
A snapshot only counts as complete when both final files exist.

Slimming a finished brain empties its optimizer memory (two thirds of the file), for example:
    python -m duck.snapshots slim runs/duck_03
"""
import io
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Windows can lock a file for a moment (for example a virus scanner), so renames are retried
RENAME_TRIES = 5
RENAME_WAIT_SECONDS = 0.2

# Inside a brain file: the optimizer's memory of past updates. Training needs it to carry on
# exactly where it stopped; watching, testing or starting a new duck does not
OPTIMIZER_FILE = "policy.optimizer.pth"


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


# Move a file onto a new name, trying again a few times if Windows has it locked
def replace_with_retry(source: Path, target: Path) -> None:
    for attempt in range(1, RENAME_TRIES + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == RENAME_TRIES:
                raise  # still locked after every try
            time.sleep(RENAME_WAIT_SECONDS)


# Save one file under a temp name, then move it onto its final name
def _save_via_temp(save_function, temp_path: Path, final_path: Path) -> None:
    save_function(str(temp_path))
    replace_with_retry(temp_path, final_path)


# Save the brain and then the statistics, so the statistics file comes last
def save_snapshot(model, vec_normalize, run_dir: Path, steps: int) -> Snapshot:
    snap = snapshot_paths(run_dir, steps)
    snap.model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model = snap.model_path.with_name(f"snapshot_{steps:012d}.tmp.zip")
    temp_norm = snap.norm_path.with_name(f"snapshot_{steps:012d}_norm.tmp.pkl")
    _save_via_temp(model.save, temp_model, snap.model_path)
    _save_via_temp(vec_normalize.save, temp_norm, snap.norm_path)
    return snap


# Rename a broken snapshot's files with a ".broken" ending, so it no longer counts
# as complete and a new snapshot can be saved at the same step count later
def set_aside_broken_snapshot(snap: Snapshot) -> None:
    for path in (snap.model_path, snap.norm_path):
        if path.exists():
            replace_with_retry(path, path.with_name(path.name + ".broken"))


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


# Delete every snapshot in a run folder except the one at keep_steps (used for the best brain)
def remove_other_snapshots(run_dir: Path, keep_steps: int) -> None:
    for snap in list_snapshots(run_dir):
        if snap.steps != keep_steps:
            snap.model_path.unlink()
            snap.norm_path.unlink()


# The optimizer memory stored in a brain file, as PyTorch saved it
def _read_optimizer(archive: zipfile.ZipFile) -> dict:
    import torch  # only needed here, and slow to import
    return torch.load(io.BytesIO(archive.read(OPTIMIZER_FILE)), map_location="cpu", weights_only=False)


# The same optimizer settings with an empty memory, as bytes ready to store
def _empty_optimizer_bytes(optimizer: dict) -> bytes:
    import torch
    buffer = io.BytesIO()
    torch.save({**optimizer, "state": {}}, buffer)
    return buffer.getvalue()


# Empty a brain file's optimizer memory, keeping everything else exactly as it was.
# The brain still loads everywhere; training from it starts with a fresh memory.
# Returns False when there was nothing to empty
def slim_brain_file(model_path: Path) -> bool:
    with zipfile.ZipFile(model_path) as archive:
        if OPTIMIZER_FILE not in archive.namelist():
            return False
        optimizer = _read_optimizer(archive)
        if not optimizer["state"]:
            return False
        files = [(info, archive.read(info.filename)) for info in archive.infolist()]
    temp_path = model_path.with_name(model_path.stem + ".tmp.zip")
    with zipfile.ZipFile(temp_path, "w") as slim:
        for info, data in files:
            slim.writestr(info, _empty_optimizer_bytes(optimizer) if info.filename == OPTIMIZER_FILE else data)
    replace_with_retry(temp_path, model_path)
    return True


# Slim every brain file in the given folders (and their subfolders); prints what it saved
def slim_folders(folders: list[Path]) -> None:
    before = after = slimmed = 0
    for folder in folders:
        for model_path in sorted(folder.rglob("snapshot_*.zip")):
            if _steps_from_model_name(model_path.name) is None:
                continue  # temp or broken files
            size = model_path.stat().st_size
            if slim_brain_file(model_path):
                slimmed += 1
                before += size
                after += model_path.stat().st_size
    print(f"Slimmed {slimmed} brains: {before / 1e6:.1f} MB became {after / 1e6:.1f} MB.")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "slim":
        print("Usage: python -m duck.snapshots slim FOLDER [FOLDER ...]")
        sys.exit(1)
    slim_folders([Path(folder) for folder in sys.argv[2:]])
