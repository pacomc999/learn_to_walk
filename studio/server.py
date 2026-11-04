"""The studio's web server: the page and the small API behind it.

It only listens on this computer (127.0.0.1), uses Python's own http.server, and answers
JSON. Start it with `python -m studio` (or the Start Studio launcher), which also opens the page.
"""
import json
import mimetypes
import os
import platform
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from studio import library, manager, previews
from studio.brain import SETTING_GROUPS as BRAIN_GROUPS
from studio.brain import default_design
from studio.creatures import ANIMALS, custom
from studio.creatures.kit import default_values
from studio.learning import SETTING_GROUPS as LEARNING_GROUPS
from studio.learning import default_learning
from studio.worlds import SETTING_GROUPS as WORLD_GROUPS
from studio.worlds import builtin_worlds

WEB = Path(__file__).parent / "web"
DEFAULT_PORT = 8770

# Ready-made brain designs for the Brain screen
BRAIN_PRESETS = [
    {"id": "standard", "name": "Standard", "description": "Two layers of 256 cells. The brain of the "
     "learn_to_walk ducks.", "layers": [256, 256]},
    {"id": "small", "name": "Small and quick", "description": "Two layers of 64 cells. Fast to train, simpler "
     "gaits.", "layers": [64, 64]},
    {"id": "deep", "name": "Deep thinker", "description": "Three layers of 256 cells. More room for tricky "
     "ground, slower to train.", "layers": [256, 256, 256]},
    {"id": "wide", "name": "Wide", "description": "Two layers of 512 cells. Lots of capacity.",
     "layers": [512, 512]},
    {"id": "tiny", "name": "Tiny", "description": "One layer of 32 cells. Can it still walk?", "layers": [32]},
]

# Ready-made training lengths for the Training plan screen
LEARNING_PRESETS = [
    {"id": "taste", "name": "Quick taste", "description": "About 10 to 20 minutes. Enough to see the first "
     "steps, not a good walk.", "values": {"million_steps": 1.0, "patience": 0}},
    {"id": "evening", "name": "An evening", "description": "About an hour. Usually a steady walk on easy ground.",
     "values": {"million_steps": 4.0, "patience": 12}},
    {"id": "overnight", "name": "Overnight", "description": "Several hours. For hard worlds and polished gaits.",
     "values": {"million_steps": 20.0, "patience": 20}},
]


def catalog() -> dict:
    animals = []
    for module in ANIMALS.values():
        entry = {"key": module.KEY, "name": module.NAME, "description": module.DESCRIPTION,
                 "groups": [group.to_dict() for group in module.SETTING_GROUPS],
                 "defaults": default_values(module.SETTING_GROUPS)}
        if module is custom:
            entry["defaults"]["body"] = custom.default_body()
            entry["templates"] = [{"key": key, **template} for key, template in custom.TEMPLATES.items()]
        animals.append(entry)
    return {
        "animals": animals,
        "world_groups": [group.to_dict() for group in WORLD_GROUPS],
        "world_templates": [world.to_dict() for world in builtin_worlds()],
        "brain_groups": [group.to_dict() for group in BRAIN_GROUPS],
        "brain_default": default_design(), "brain_presets": BRAIN_PRESETS,
        "learning_groups": [group.to_dict() for group in LEARNING_GROUPS],
        "learning_default": default_learning(), "learning_presets": LEARNING_PRESETS,
        "system": {"cores": os.cpu_count() or 1, "platform": platform.system(), **library.system_facts()},
    }


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "WalkStudio/1.0"

    # Quiet: the console only shows problems
    def log_message(self, format, *args):
        return

    # ---------- Answers ----------

    def send_json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_json({"error": "Not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # ---------- Routing ----------

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")

    def _handle(self, method: str) -> None:
        url = urlparse(self.path)
        parts = [part for part in url.path.split("/") if part]
        try:
            if not parts or parts[0] != "api":
                self._static(url.path)
                return
            result = self._api(method, parts[1:], parse_qs(url.query))
            if result is not None:
                self.send_json(result)
        except ValueError as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:  # a plain message for the page, the details for the console
            traceback.print_exc()
            self.send_json({"error": f"Something went wrong: {error}"}, 500)

    def _static(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"
        target = (WEB / path.lstrip("/")).resolve()
        if WEB.resolve() not in target.parents:
            self.send_json({"error": "Not found"}, 404)
            return
        types = {".js": "text/javascript", ".css": "text/css", ".html": "text/html; charset=utf-8",
                 ".svg": "image/svg+xml"}
        self.send_file(target, types.get(target.suffix))

    def _api(self, method: str, parts: list[str], query: dict):
        route = parts[0] if parts else ""
        if route == "catalog" and method == "GET":
            return catalog()
        if route == "preview":
            return self._previews(parts[1:], self.read_body())
        if route in ("creatures", "worlds", "brains"):
            return self._library(route, method, parts[1:])
        if route == "trainings":
            return self._trainings(method, parts[1:])
        if route == "jobs" and len(parts) > 1:
            from studio import challenges
            return challenges.job_status(parts[1])
        if route == "import":
            return self._import(method, parts[1:])
        if route == "race" and method == "POST":
            body = self.read_body()
            return previews.race(body.get("ids") or [], body.get("world") or {}, body.get("speed"), body.get("height"),
                                 int(body.get("seed", 1000)), float(body.get("seconds", 20)))
        raise ValueError("Unknown request.")

    def _previews(self, parts: list[str], body: dict):
        kind = parts[0] if parts else ""
        if kind == "creature":
            return previews.creature_preview(body.get("animal", "duck"), body.get("values") or {})
        if kind == "drop":
            return previews.drop_test(body.get("animal", "duck"), body.get("values") or {}, body.get("world"),
                                      float(body.get("seconds", 4.0)))
        if kind == "brain":
            return previews.brain_facts(body.get("animal", "duck"), body.get("values") or {}, body.get("design"))
        raise ValueError("Unknown preview.")

    def _import(self, method: str, parts: list[str]):
        action = parts[0] if parts else ""
        if action == "guess" and method == "GET":
            found = [path for path in library.likely_learn_to_walk_folders() if (path / "duck_catalogue").is_dir()]
            return {"folder": str(found[0]) if found else ""}
        from studio import importer  # loads PyTorch, so only when needed
        body = self.read_body()
        if action == "list" and method == "POST":
            return importer.find_ducks(str(body.get("folder") or ""))
        if action == "duck" and method == "POST":
            return {"id": importer.import_duck(str(body.get("folder") or ""))}
        raise ValueError("Unknown request.")

    def _library(self, kind: str, method: str, parts: list[str]):
        if method == "GET" and not parts:
            return library.list_items(kind)
        if method == "POST" and not parts:
            return library.save_item(kind, self.read_body())
        if method == "DELETE" and parts:
            library.delete_item(kind, parts[0])
            return {"ok": True}
        if method == "GET" and parts:
            item = library.get_item(kind, parts[0])
            if item is None:
                raise ValueError("Not found.")
            return item
        raise ValueError("Unknown request.")

    def _trainings(self, method: str, parts: list[str]):
        if not parts:
            if method == "GET":
                return manager.list_trainings()
            if method == "POST":
                return manager.create_training(self.read_body())
            raise ValueError("Unknown request.")
        training_id = library.safe_id(parts[0])
        action = parts[1] if len(parts) > 1 else ""
        if method == "DELETE" and not action:
            if manager.is_running(training_id):
                raise ValueError("Pause this training before deleting it.")
            library.delete_training(training_id)
            return {"ok": True}
        if method == "GET" and not action:
            found = manager.details(training_id)
            if found is None:
                raise ValueError("That training was not found.")
            return found
        if method == "GET" and action == "replay" and len(parts) > 2:
            self.send_file(manager.replay_path(training_id, parts[2]), "application/json")
            return None
        if method == "POST" and action == "stop":
            manager.request_stop(training_id)
            return {"ok": True}
        if method == "POST" and action == "extend":
            manager.extend(training_id, float(self.read_body().get("million", 2.0)))
            return {"ok": True}
        if method == "POST" and action == "continue":
            manager.start_process(training_id, resume=True)
            return {"ok": True}
        if method == "POST" and action == "rename":
            folder = library.training_folder(training_id)
            setup = library.read_json(folder / "setup.json")
            setup["name"] = str(self.read_body().get("name") or setup["name"]).strip()[:80]
            library.write_json(folder / "setup.json", setup)
            return {"ok": True}
        if action == "challenges":
            from studio import challenges
            if method == "GET":
                return challenges.saved_results(training_id) or {}
            if method == "POST" and len(parts) > 2:  # one level, with its replay
                body = self.read_body()
                return challenges.run_level(training_id, parts[2], float(body.get("level")), keep_replay=True)
            if method == "POST":
                return {"job": challenges.start(training_id)}
        if method == "POST" and action == "watch":
            body = self.read_body()
            return previews.watch(training_id, body.get("world"), body.get("speed"), body.get("height"),
                                  int(body.get("seed", 1000)), body.get("seconds"))
        raise ValueError("Unknown request.")


# Whether a Walk Studio already answers on this port (started in another window)
def _studio_answers(port: int) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/catalog", timeout=2) as response:
            return b"world_templates" in response.read(200000)
    except OSError:
        return False


def serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    server = None
    for candidate in range(port, port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), StudioHandler)
            port = candidate
            break
        except OSError:
            if _studio_answers(candidate):
                print(f"Walk Studio is already running at http://127.0.0.1:{candidate}/ (opening it).", flush=True)
                if open_browser:
                    webbrowser.open(f"http://127.0.0.1:{candidate}/")
                return
    if server is None:
        raise SystemExit("No free port for Walk Studio (tried 10). Close other programs and try again.")
    address = f"http://127.0.0.1:{port}/"
    print(f"Walk Studio is running at {address}", flush=True)
    print("Keep this window open while you use the studio. Press Ctrl+C to close it and pause any training;",
          flush=True)
    print("closing the window instead leaves a running training going in the background.", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPausing any training that is running, then closing.", flush=True)
        manager.stop_all()
    finally:
        server.server_close()
