"""Sharing a walker: everything about one training packed into a single .walker file (a zip), and
opened again on another computer as a walker of its own."""
import io
import json
import zipfile
from pathlib import PurePosixPath

from studio import library

# What goes into a .walker file: the recipe, the results and the brains (never the log or stop files)
SHARED_FILES = ["setup.json", "status.json", "progress.csv", "exams.csv", "challenges.json", "thumbnail.jpg"]
SHARED_FOLDERS = ["brain", "replays"]
MOST_BYTES = 300 * 1024 * 1024


def pack(training_id: str) -> tuple[bytes, str]:
    folder = library.training_folder(training_id)
    setup = library.read_json(folder / "setup.json")
    if not setup:
        raise ValueError("That walker was not found.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SHARED_FILES:
            if (folder / name).exists():
                archive.write(folder / name, name)
        for sub in SHARED_FOLDERS:
            for path in sorted((folder / sub).glob("*")):
                if path.is_file() and ".tmp" not in path.name:
                    archive.write(path, f"{sub}/{path.name}")
    name = "".join(c if c.isalnum() or c in " -_.()" else "_" for c in setup.get("name", "walker")).strip() or "walker"
    return buffer.getvalue(), f"{name}.walker"


def unpack(data: bytes) -> str:
    if len(data) > MOST_BYTES:
        raise ValueError("That file is too big to be a walker.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("That is not a .walker file.") from None
    with archive:
        names = archive.namelist()
        if "setup.json" not in names:
            raise ValueError("That is not a .walker file (it has no recipe).")
        setup = json.loads(archive.read("setup.json"))
        training_id = library.new_id(setup.get("name", "walker"))
        folder = library.training_folder(training_id)
        for name in names:
            path = PurePosixPath(name)
            allowed = (name in SHARED_FILES) or (len(path.parts) == 2 and path.parts[0] in SHARED_FOLDERS
                                                and not path.parts[1].startswith("."))
            if not allowed or ".." in path.parts:
                continue  # never write anything else, and never outside the walker's folder
            target = folder.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    setup.update(id=training_id, start_from=None, shared=True)
    library.write_json(folder / "setup.json", setup)
    status = library.read_json(folder / "status.json", {}) or {}
    if status.get("state") in ("starting", "training", "examining"):
        status["state"] = "stopped"
    status.update(message="Opened from a shared walker file.", pid=None)
    library.write_json(folder / "status.json", status)
    return training_id
