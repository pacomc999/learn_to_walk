"""The library: everything the user makes, kept as plain files in the library folder.

    library/creatures/<id>.json   creatures designed on the Creature screen
    library/worlds/<id>.json      worlds made on the World screen
    library/brains/<id>.json      brain designs saved on the Brain screen
    library/trainings/<id>/       one folder per training session (see studio/train.py)

The folder sits next to the studio code and is not kept in git, so it is the user's own.
"""
import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LIBRARY = Path(os.environ.get("STUDIO_LIBRARY", PROJECT_ROOT / "library"))
KINDS = ("creatures", "worlds", "brains", "trainings")


def folder(kind: str) -> Path:
    path = LIBRARY / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


# A short id made from a name, unique in its folder, for example "big_legs_duck_3f2a"
def new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:32] or "item"
    return f"{slug}_{uuid.uuid4().hex[:4]}"


def safe_id(item_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", item_id or ""):
        raise ValueError("That is not a valid name for a library item.")
    return item_id


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path, fallback=None):
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return fallback
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)  # it may be half written right now
    return fallback


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


# ---------- Creatures, worlds and brain designs: one JSON file each ----------

def list_items(kind: str) -> list[dict]:
    items = [read_json(path) for path in folder(kind).glob("*.json")]
    items = [item for item in items if isinstance(item, dict)]
    return sorted(items, key=lambda item: item.get("updated", ""), reverse=True)


def get_item(kind: str, item_id: str) -> dict | None:
    return read_json(folder(kind) / f"{safe_id(item_id)}.json")


def save_item(kind: str, data: dict) -> dict:
    data = dict(data)
    name = str(data.get("name") or "Untitled").strip()[:80] or "Untitled"
    data["name"] = name
    item_id = data.get("id")
    # Saving over an existing item keeps its id; anything else (new, or a built-in template) gets a new one
    existing = None
    if item_id and not data.get("builtin"):
        try:
            existing = get_item(kind, item_id)
        except ValueError:
            existing = None
    if existing is None:
        data["id"] = new_id(name)
        data["created"] = now()
    data.pop("builtin", None)
    data.setdefault("created", now())
    data["updated"] = now()
    write_json(folder(kind) / f"{data['id']}.json", data)
    return data


def delete_item(kind: str, item_id: str) -> None:
    path = folder(kind) / f"{safe_id(item_id)}.json"
    path.unlink(missing_ok=True)


# ---------- Training folders ----------

def training_folder(training_id: str) -> Path:
    return folder("trainings") / safe_id(training_id)


def delete_training(training_id: str) -> None:
    path = training_folder(training_id)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


# Measured practice speed of this computer, so the page can estimate how long training takes
def system_facts() -> dict:
    return read_json(LIBRARY / "system.json", {}) or {}


def remember_speed(worlds: int, steps_per_second: float) -> None:
    facts = system_facts()
    speeds = facts.get("speeds", {})
    speeds[str(worlds)] = round(steps_per_second)
    facts["speeds"] = speeds
    write_json(LIBRARY / "system.json", facts)
