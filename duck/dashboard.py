"""A live training dashboard in the browser.

Run it from the project root in a second terminal while training runs, for example:
    python -m duck.dashboard runs/duck_v1 --steps 30000000
It opens a page that refreshes itself: score and distance charts, how far the run is,
and a film of the best brain so far (the newest snapshot before the first exam)
walking.
Press Ctrl+C to stop it; training is not affected.
"""
import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import imageio
import numpy as np

from duck.duck_env import FILM_HEIGHT, FILM_WIDTH
from duck.exam import EXAM_FIRST_SEED, best_folder, exam_at
from duck.floor import MAX_ROUGHNESS
from duck.policy import load_policy
from duck.progress import ProgressRow, minutes_since_improvement, read_progress
from duck.snapshots import Snapshot, latest_snapshot, replace_with_retry
from duck.video import record_clip

# The page itself, kept next to this file
PAGE_PATH = Path(__file__).with_name("dashboard.html")

# The film of the newest snapshot and a note of which snapshot it shows (written last, once the film is ready)
CLIP_NAME = "dashboard_latest.mp4"
CLIP_STEPS_NAME = "dashboard_latest_steps.txt"

# Film size and length: a whole 20 s attempt, like the exam, at the full film size.
# Measured on the rough floor: about 45 s to make, loading included, and about 6 MB.
# Both sides are multiples of 16, which video files need
CLIP_SECONDS = 20.0
CLIP_FPS = 15
CLIP_WIDTH = FILM_WIDTH
CLIP_HEIGHT = FILM_HEIGHT

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


# Which snapshot the current film shows, or None if there is no film yet
def _clip_steps(run_dir: Path) -> int | None:
    try:
        return int((run_dir / CLIP_STEPS_NAME).read_text())
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
    clip_steps = _clip_steps(run_dir)
    return {
        "run": run_dir.name,
        "total_steps": total_steps,
        "latest_steps": rows[-1].timesteps if rows else None,
        "percent_done": percent,
        "minutes_left": minutes_left,
        "steps_per_second": recent_steps_per_second(rows),
        "minutes_since_improvement": minutes_since_improvement(rows),
        "clip_steps": clip_steps,
        "clip_exam": exam_at(run_dir, clip_steps) if clip_steps is not None else None,
        "points": [
            {"steps": row.timesteps, "score": row.mean_reward, "distance": row.mean_distance}
            for row in rows
        ],
    }


# Save pictures as a video, under a temp name first so the page never sees half a file
def save_mp4(frames: list[np.ndarray], path: Path, fps: int = CLIP_FPS) -> None:
    temp_path = path.with_suffix(".tmp.mp4")
    writer = imageio.get_writer(temp_path, fps=fps, codec="libx264", pixelformat="yuv420p")
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    replace_with_retry(temp_path, path)


# Film a snapshot walking at the judging speed on the roughest floor, on a fixed rough layout
# built from seed 1000. It is not the exam's first rough layout, because the film fixes the
# speed at 0.4 and so draws no target speed before the floor
def make_snapshot_clip(run_dir: Path, snapshot: Snapshot) -> None:
    frames = record_clip(load_policy(snapshot), seconds=CLIP_SECONDS, fps=CLIP_FPS,
                         width=CLIP_WIDTH, height=CLIP_HEIGHT, seed=EXAM_FIRST_SEED, roughness=MAX_ROUGHNESS)
    save_mp4(frames, run_dir / CLIP_NAME)
    (run_dir / CLIP_STEPS_NAME).write_text(str(snapshot.steps))


# The brain to film: the best one the exams have kept, or the newest snapshot if there is none yet.
# Regular snapshots come only once a million steps, so the newest one can be far behind the best
def brain_to_film(run_dir: Path) -> Snapshot | None:
    return latest_snapshot(best_folder(run_dir)) or latest_snapshot(run_dir)


# Make a new film if the brain to film is not the one the film shows
def refresh_clip(run_dir: Path) -> None:
    snapshot = brain_to_film(run_dir)
    if snapshot is None or snapshot.steps == _clip_steps(run_dir):
        return
    print(f"Filming the brain from {snapshot.steps:,} steps for the dashboard", flush=True)
    try:
        make_snapshot_clip(run_dir, snapshot)
    except Exception as error:  # a snapshot being written right now can fail to load
        print(f"Could not film that snapshot yet ({error}). Trying again soon.", flush=True)


# The first and last byte asked for in a header like "bytes=200-999" or "bytes=200-".
# None when there is no usable header, so the whole file is sent
def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    first_text, _, last_text = header[len("bytes="):].partition("-")
    try:
        first = int(first_text) if first_text else 0
        last = min(int(last_text), size - 1) if last_text else size - 1
    except ValueError:
        return None
    return (first, last) if first <= last else None


# A web server that answers the page, the status, the film and its brain data for one run
def make_server(run_dir: Path, total_steps: int | None, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:

    class DashboardHandler(BaseHTTPRequestHandler):
        # Send bytes back with the right content type, never cached
        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # Send the film, or just the piece of it the video player asks for
        def _send_video(self, path: Path) -> None:
            body = path.read_bytes()
            wanted = byte_range(self.headers.get("Range"), len(body))
            if wanted is None:
                self._send(body, "video/mp4")
                return
            first, last = wanted
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body[first:last + 1])

        # The three things the page asks for
        def do_GET(self) -> None:
            route = self.path.split("?")[0]
            if route == "/":
                self._send(PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif route == "/status.json":
                body = json.dumps(run_status(run_dir, total_steps)).encode()
                self._send(body, "application/json")
            elif route == "/latest.mp4" and (run_dir / CLIP_NAME).exists():
                self._send_video(run_dir / CLIP_NAME)
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

    # The web page runs in the background; filming happens here in the main thread
    server = make_server(arguments.run_dir, arguments.steps, arguments.port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{arguments.port}/"
    print(f"Dashboard at {address} . Press Ctrl+C to stop it.", flush=True)
    webbrowser.open(address)
    try:
        while True:
            refresh_clip(arguments.run_dir)
            time.sleep(CHECK_SECONDS)
    except KeyboardInterrupt:
        print("Dashboard stopped.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
