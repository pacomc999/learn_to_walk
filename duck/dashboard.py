"""A live training dashboard in the browser.

Run it from the project root in a second terminal while training runs, for example:
    python -m duck.dashboard runs/duck_v1 --steps 30000000
It opens a page that refreshes itself: score and distance charts, how far the run is,
and a GIF of the newest snapshot walking. Press Ctrl+C to stop it; training is not affected.
"""
import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from duck.policy import load_policy
from duck.progress import ProgressRow, minutes_since_improvement, read_progress
from duck.snapshots import Snapshot, latest_snapshot, replace_with_retry
from duck.video import record_clip

# The page itself, kept next to this file
PAGE_PATH = Path(__file__).with_name("dashboard.html")

# The GIF of the newest snapshot, and a note of which snapshot it shows
GIF_NAME = "dashboard_latest.gif"
GIF_STEPS_NAME = "dashboard_latest_steps.txt"

# GIF size and length: small enough to make in a few seconds while training runs
GIF_SECONDS = 8.0
GIF_FPS = 15
GIF_WIDTH = 400
GIF_HEIGHT = 300

# How often to look for a new snapshot, in seconds
CHECK_SECONDS = 20

# Training speed is measured over this many of the newest log rows
SPEED_ROWS = 20

DEFAULT_PORT = 8765


# Steps per second over the newest log rows, or None if there are too few rows
def recent_steps_per_second(rows: list[ProgressRow]) -> float | None:
    recent = rows[-SPEED_ROWS:]
    if len(recent) < 2:
        return None
    seconds = (recent[-1].time - recent[0].time).total_seconds()
    if seconds <= 0:
        return None
    return (recent[-1].timesteps - recent[0].timesteps) / seconds


# Which snapshot the current GIF shows, or None if there is no GIF yet
def _gif_steps(run_dir: Path) -> int | None:
    try:
        return int((run_dir / GIF_STEPS_NAME).read_text())
    except (OSError, ValueError):
        return None


# Percent done and minutes left, or None for each when they cannot be worked out
def _time_left(rows: list[ProgressRow], total_steps: int | None) -> tuple[float | None, float | None]:
    if not rows or not total_steps:
        return None, None
    percent = 100 * rows[-1].timesteps / total_steps
    speed = recent_steps_per_second(rows)
    if speed is None or speed <= 0:
        return percent, None
    minutes = max(0, total_steps - rows[-1].timesteps) / speed / 60
    return percent, minutes


# Everything the page shows, as a dictionary that becomes JSON
def run_status(run_dir: Path, total_steps: int | None) -> dict:
    rows = read_progress(run_dir)
    percent, minutes_left = _time_left(rows, total_steps)
    return {
        "run": run_dir.name,
        "total_steps": total_steps,
        "latest_steps": rows[-1].timesteps if rows else None,
        "percent_done": percent,
        "minutes_left": minutes_left,
        "steps_per_second": recent_steps_per_second(rows),
        "minutes_since_improvement": minutes_since_improvement(rows),
        "gif_steps": _gif_steps(run_dir),
        "points": [
            {"steps": row.timesteps, "score": row.mean_reward, "distance": row.mean_distance}
            for row in rows
        ],
    }


# Save pictures as a looping GIF, under a temp name first so the page never sees half a file
def save_gif(frames: list[np.ndarray], path: Path, fps: int = GIF_FPS) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    temp_path = path.with_suffix(".tmp.gif")
    images[0].save(temp_path, save_all=True, append_images=images[1:],
                   duration=round(1000 / fps), loop=0)
    replace_with_retry(temp_path, path)


# Film a snapshot walking at the judging speed and save it as the dashboard GIF
def make_snapshot_gif(run_dir: Path, snapshot: Snapshot) -> None:
    frames = record_clip(load_policy(snapshot), seconds=GIF_SECONDS, fps=GIF_FPS,
                         width=GIF_WIDTH, height=GIF_HEIGHT)
    save_gif(frames, run_dir / GIF_NAME)
    (run_dir / GIF_STEPS_NAME).write_text(str(snapshot.steps))


# Make a new GIF if there is a snapshot newer than the one the GIF shows
def refresh_gif(run_dir: Path) -> None:
    snapshot = latest_snapshot(run_dir)
    if snapshot is None or snapshot.steps == _gif_steps(run_dir):
        return
    print(f"Filming the snapshot from {snapshot.steps:,} steps for the dashboard", flush=True)
    try:
        make_snapshot_gif(run_dir, snapshot)
    except Exception as error:  # a snapshot being written right now can fail to load
        print(f"Could not film that snapshot yet ({error}). Trying again soon.", flush=True)


# A web server that answers the page, the status and the GIF for one run
def make_server(run_dir: Path, total_steps: int | None, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:

    class DashboardHandler(BaseHTTPRequestHandler):
        # Send bytes back with the right content type, never cached
        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # The three things the page asks for
        def do_GET(self) -> None:
            route = self.path.split("?")[0]
            if route == "/":
                self._send(PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif route == "/status.json":
                body = json.dumps(run_status(run_dir, total_steps)).encode()
                self._send(body, "application/json")
            elif route == "/latest.gif" and (run_dir / GIF_NAME).exists():
                self._send((run_dir / GIF_NAME).read_bytes(), "image/gif")
            else:
                self.send_error(404)

        # Keep the terminal quiet: no line for every request
        def log_message(self, *args) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch training progress in the browser.")
    parser.add_argument("run_dir", type=Path, help="run folder, for example runs/duck_v1")
    parser.add_argument("--steps", type=int, default=None, help="total steps of the run, for the time left")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="web page port")
    arguments = parser.parse_args()

    # The web page runs in the background; filming GIFs happens here in the main thread
    server = make_server(arguments.run_dir, arguments.steps, arguments.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{arguments.port}/"
    print(f"Dashboard at {address} . Press Ctrl+C to stop it.", flush=True)
    webbrowser.open(address)
    try:
        while True:
            refresh_gif(arguments.run_dir)
            time.sleep(CHECK_SECONDS)
    except KeyboardInterrupt:
        print("Dashboard stopped.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
