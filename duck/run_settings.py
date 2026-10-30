"""Run settings: which duck a run belongs to, and the world and body it trains in.

A new run writes run_settings.json next to its snapshots, with frozen copies of its world and
body, so it always continues in its own world, even if the duck's files change later.
A duck keeps the same frozen copies in its duck.json (world_settings and body_settings).
"""
import json
from dataclasses import dataclass
from pathlib import Path

from duck.bodies import STANDARD_BODY, Body, body_from_dict, body_to_dict, load_body
from duck.worlds import World, load_world, world_from_dict, world_to_dict

# The settings file of a run, and the story file of a duck
RUN_SETTINGS_NAME = "run_settings.json"
STORY_NAME = "duck.json"

# Folders inside a run or a duck that hold brains: their settings are the parent's
BRAIN_FOLDER_NAMES = ("best", "brain")


# The world and body of one run (or one duck), and the duck folder it belongs to (None for a scratch run)
@dataclass(frozen=True)
class RunSettings:
    duck_dir: Path | None
    world: World
    body: Body


# The file contents of settings
def _settings_to_dict(settings: RunSettings) -> dict:
    return {
        "duck": settings.duck_dir.as_posix() if settings.duck_dir else None,
        "world": world_to_dict(settings.world),
        "body": body_to_dict(settings.body),
    }


# Write a run's settings next to its snapshots
def write_run_settings(run_dir: Path, settings: RunSettings) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_SETTINGS_NAME).write_text(json.dumps(_settings_to_dict(settings), indent=2) + "\n")


# A run's settings, or None if it has none (runs from before worlds)
def read_run_settings(run_dir: Path) -> RunSettings | None:
    path = run_dir / RUN_SETTINGS_NAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    duck_dir = Path(data["duck"]) if data.get("duck") else None
    source = path.as_posix()
    return RunSettings(duck_dir, world_from_dict(data["world"], source), body_from_dict(data["body"], source))


# A duck's settings, from the frozen copies in its duck.json
def duck_settings(duck_dir: Path) -> RunSettings:
    story = json.loads((duck_dir / STORY_NAME).read_text())
    if "world_settings" not in story or "body_settings" not in story:
        raise ValueError(f"{duck_dir.as_posix()} has no world yet (it is older than worlds).")
    source = (duck_dir / STORY_NAME).as_posix()
    return RunSettings(duck_dir, world_from_dict(story["world_settings"], source),
                       body_from_dict(story["body_settings"], source))


# The settings of any folder a tool is given: a duck, its brain/, a run, or its best/. None if it has none
def settings_for_folder(folder: Path) -> RunSettings | None:
    if folder.name in BRAIN_FOLDER_NAMES:
        folder = folder.parent
    if (folder / RUN_SETTINGS_NAME).exists():
        return read_run_settings(folder)
    if (folder / STORY_NAME).exists():
        return duck_settings(folder)
    return None


# ---------- For the tools ----------

# The world and body a tool should use for a folder: the folder's own, each replaced by a named file when
# given. A folder from before worlds needs --world (the body falls back to the standard duck)
def resolve_settings(folder: Path, world_name: str | None, body_name: str | None) -> tuple[World, Body]:
    own = settings_for_folder(folder)
    if own is None and world_name is None:
        raise ValueError(f"{folder.as_posix()} has no world yet (it is older than worlds). "
                         "Add --world NAME, for example --world flat_walking.")
    world = load_world(world_name) if world_name else own.world
    body = load_body(body_name) if body_name else (own.body if own else STANDARD_BODY)
    return world, body


# Add --world and --body to a tool's command line
def add_settings_arguments(parser) -> None:
    parser.add_argument("--world", default=None,
                        help="try another world (a file name in worlds/); default: the folder's own world")
    parser.add_argument("--body", default=None,
                        help="try another body (a file name in bodies/); default: the folder's own body")


# resolve_settings for a command line: on a problem, print it as one plain sentence and stop with code 1
def settings_or_exit(folder: Path, world_name: str | None, body_name: str | None) -> tuple[World, Body]:
    try:
        return resolve_settings(folder, world_name, body_name)
    except ValueError as error:
        print(error)
        raise SystemExit(1)
