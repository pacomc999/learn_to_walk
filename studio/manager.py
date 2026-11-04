"""Starting, pausing and following training sessions.

Each session runs as its own process (python -m studio.train <folder>), so the page stays
quick while the computer works hard. This module starts those processes, asks them to stop,
and reads back what they wrote, in the form the page wants.
"""
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from studio import library
from studio.brain import clean_design, design_senses
from studio.creatures import animal, build_creature
from studio.env import observation_size
from studio.learning import clean_learning
from studio.worlds import world_from_dict

# Processes this server started, by training id
_processes: dict[str, subprocess.Popen] = {}

# A session whose status has not changed for this long, and whose process is gone, was interrupted
SILENT_SECONDS = 600

ACTIVE_STATES = ("starting", "training", "examining")


# ---------- Starting ----------

def create_training(request: dict) -> dict:
    creature = request.get("creature") or {}
    kind = animal(creature.get("animal", "duck"))
    creature_values = creature.get("values") or {}
    world = world_from_dict(request.get("world") or {})
    design = clean_design(request.get("brain"))
    learning = clean_learning(request.get("learning"))
    name = str(request.get("name") or f"{creature.get('name') or kind.NAME} in {world.name}").strip()[:80]
    training_id = library.new_id(name)
    setup = {
        "id": training_id, "name": name, "created": library.now(),
        "creature": {"animal": kind.KEY, "name": creature.get("name") or kind.NAME, "values": creature_values,
                     "id": creature.get("id")},
        "world": world.to_dict(), "brain": design, "learning": learning,
        "start_from": request.get("start_from") or None,
    }
    if setup["start_from"]:
        _check_start_from(setup)
    folder = library.training_folder(training_id)
    library.write_json(folder / "setup.json", setup)
    library.write_json(folder / "status.json", {"state": "starting", "steps": 0,
                                                 "total_steps": int(learning["million_steps"] * 1e6),
                                                 "message": "Starting", "updated": library.now()})
    start_process(training_id, resume=False)
    return setup


# A brain can only carry on in a body with the same motors and senses
def _check_start_from(setup: dict) -> None:
    source = library.read_json(library.training_folder(setup["start_from"]) / "setup.json")
    if not source:
        raise ValueError("The brain to start from was not found.")
    if not (library.training_folder(setup["start_from"]) / "brain" / "best.zip").exists():
        raise ValueError("That training has no best brain yet.")
    old, new = _brain_shape(source), _brain_shape(setup)
    if old != new:
        raise ValueError("That brain does not fit: the animal needs the same motors, senses and brain layers.")


def _brain_shape(setup: dict) -> tuple:
    design = clean_design(setup["brain"])
    build = build_creature(setup["creature"]["animal"], setup["creature"]["values"])
    senses = design_senses(design)
    return (build.motor_names, observation_size(build, senses), design["layers"], design["judge_layers"],
            design["values"]["symmetric"], design["values"]["activation"])


def start_process(training_id: str, resume: bool) -> None:
    if is_running(training_id):
        return
    folder = library.training_folder(training_id)
    (folder / "stop.request").unlink(missing_ok=True)
    command = [sys.executable, "-m", "studio.train", str(folder)]
    if resume:
        command.append("--resume")
    log = open(folder / "log.txt", "a", encoding="utf-8")
    log.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {'Carrying on' if resume else 'Starting'} ===\n")
    log.flush()
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    _processes[training_id] = subprocess.Popen(command, cwd=library.PROJECT_ROOT, stdout=log,
                                               stderr=subprocess.STDOUT, creationflags=flags)
    if resume:
        status = library.read_json(folder / "status.json", {}) or {}
        status.update(state="starting", message="Carrying on", updated=library.now())
        library.write_json(folder / "status.json", status)


def request_stop(training_id: str) -> None:
    folder = library.training_folder(training_id)
    (folder / "stop.request").write_text("stop", encoding="utf-8")


def is_running(training_id: str) -> bool:
    process = _processes.get(training_id)
    return process is not None and process.poll() is None


def running_ids() -> list[str]:
    return [training_id for training_id in list(_processes) if is_running(training_id)]


# Stop every session this server started (used when the studio closes)
def stop_all(wait_seconds: float = 20.0) -> None:
    ids = running_ids()
    for training_id in ids:
        request_stop(training_id)
    deadline = time.time() + wait_seconds
    for training_id in ids:
        process = _processes[training_id]
        try:
            process.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            process.terminate()


# ---------- Reading back ----------

def status_of(training_id: str) -> dict:
    folder = library.training_folder(training_id)
    status = library.read_json(folder / "status.json", {}) or {}
    state = status.get("state", "starting")
    if state in ACTIVE_STATES and not is_running(training_id):
        updated = status.get("updated")
        silent = _seconds_since(updated) if updated else SILENT_SECONDS + 1
        gone = status.get("pid") is not None and not process_alive(int(status["pid"]))
        if training_id in _processes or silent > SILENT_SECONDS or gone:
            status["state"] = "interrupted"
            status["message"] = "It stopped unexpectedly (was the studio closed?). Press Continue to carry on."
    status["running"] = is_running(training_id)
    return status


# Whether a process with this id still runs (used for sessions started by an earlier studio window)
def process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _seconds_since(text: str) -> float:
    try:
        return (datetime.now() - datetime.fromisoformat(text)).total_seconds()
    except ValueError:
        return 0.0


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({key: _number(value) for key, value in row.items()})
    except OSError:
        return []
    return rows


def _number(text):
    if text is None or text == "":
        return None
    try:
        number = float(text)
        return int(number) if number.is_integer() and "." not in text else number
    except ValueError:
        return text


def summary(training_id: str) -> dict | None:
    folder = library.training_folder(training_id)
    setup = library.read_json(folder / "setup.json")
    if not setup:
        return None
    return {"id": training_id, "name": setup.get("name"), "created": setup.get("created"),
            "creature": {key: setup["creature"].get(key) for key in ("animal", "name")},
            "creature_values": setup["creature"].get("values", {}),
            "world": {"name": setup["world"].get("name"), "id": setup["world"].get("id"),
                      "style": setup["world"].get("values", {}).get("style")},
            "status": status_of(training_id), "has_brain": (folder / "brain" / "best.zip").exists()}


def list_trainings() -> list[dict]:
    items = [summary(path.name) for path in library.folder("trainings").iterdir() if path.is_dir()]
    items = [item for item in items if item]
    return sorted(items, key=lambda item: item.get("created") or "", reverse=True)


def details(training_id: str) -> dict | None:
    folder = library.training_folder(training_id)
    setup = library.read_json(folder / "setup.json")
    if not setup:
        return None
    replays = sorted(path.stem for path in (folder / "replays").glob("exam_*.json"))
    # Every value filled in, so the page never meets a missing one
    setup["brain"] = clean_design(setup.get("brain"))
    setup["learning"] = clean_learning(setup.get("learning"))
    setup["world"] = world_from_dict(setup.get("world") or {}).to_dict()
    return {"setup": setup, "status": status_of(training_id), "progress": read_rows(folder / "progress.csv"),
            "exams": read_rows(folder / "exams.csv"), "replays": replays,
            "has_best": (folder / "replays" / "best.json").exists(),
            "has_brain": (folder / "brain" / "best.zip").exists(), "log": _log_tail(folder / "log.txt")}


def _log_tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def replay_path(training_id: str, name: str) -> Path:
    name = library.safe_id(name)
    return library.training_folder(training_id) / "replays" / f"{name}.json"
