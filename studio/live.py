"""Live mode: a trained walker runs in real time and answers the page as it goes.

The page changes the asked speed, gives the walker a push or starts it again, and keeps asking for
the frames made since its last visit. The simulation runs in its own thread at the speed of real
life (50 brain decisions a second). A session that nobody asks about for a while stops by itself.
This is learn_to_walk's drive tool (duck/drive.py) in the browser.
"""
import threading
import time
import uuid

import numpy as np

from studio import library
from studio.brain import clean_design, design_senses
from studio.creatures import build_creature
from studio.env import WalkEnv
from studio.replay import REPLAY_FPS, _frame, ground_patch, scene_shapes
from studio.worlds import world_from_dict

# A push: this share of the walker's weight, for this long, from the side
PUSH_SHARE = 0.6
PUSH_SECONDS = 0.15

# Stop a session nobody has asked about for this long; keep at most this many frames in memory
IDLE_SECONDS = 20.0
MOST_FRAMES = 25 * 60

# A fallen walker is put back on its feet after this long
STAND_UP_AFTER = 2.0

_sessions: dict[str, "LiveSession"] = {}
_lock = threading.Lock()


class LiveSession:
    def __init__(self, training_id: str, world_data: dict | None, speed: float):
        from studio.previews import _brain
        setup = library.read_json(library.training_folder(training_id) / "setup.json")
        if not setup:
            raise ValueError("That walker was not found.")
        self.build = build_creature(setup["creature"]["animal"], setup["creature"]["values"])
        world = dict(world_data or setup["world"])
        world["values"] = {**(world.get("values") or {}), "attempt_seconds": 60}
        self.world = world_from_dict(world)
        self.env = WalkEnv(self.build, self.world, design_senses(clean_design(setup.get("brain"))),
                           target_speed=float(speed), terrain_height=self.world.tallest * 0.75)
        self.brain = _brain(training_id)
        self.id = uuid.uuid4().hex[:10]
        self.speed = float(speed)
        self.frames: list[list[float]] = []
        self.first_frame = 0               # the number of the oldest frame still kept
        self.events: list[str] = []
        self.push_until = -1.0
        self.push = np.zeros(3)
        self.fallen_at = None
        self.seed = 11
        self.last_asked = time.time()
        self.running = True
        self.lock = threading.Lock()
        self.observation, _ = self.env.reset(seed=self.seed)
        self.frames.append(_frame(self.env))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # What the page needs once: the shapes and the whole floor
    def header(self) -> dict:
        bodies, shapes = scene_shapes(self.env)
        layout = self.env.scene.layout
        ground = ground_patch(self.env, layout.x_min, layout.x_max) if layout else {"kind": "plane"}
        return {"id": self.id, "fps": REPLAY_FPS, "bodies": bodies, "shapes": shapes, "ground": ground,
                "style": self.world.values["style"], "size": self.build.size,
                "stand_height": self.env.scene.stand_height, "frames": [self.frames[-1]],
                "floor_end": layout.x_max - 1.0 if layout else None}

    def _run(self) -> None:
        step_seconds = self.env.control_seconds
        steps_per_frame = round(50 / REPLAY_FPS)
        next_time = time.perf_counter()
        count = 0
        while self.running:
            if time.time() - self.last_asked > IDLE_SECONDS:
                break
            with self.lock:
                self._step()
                count += 1
                if count % steps_per_frame == 0:
                    self.frames.append(_frame(self.env))
                    if len(self.frames) > MOST_FRAMES:
                        drop = len(self.frames) - MOST_FRAMES
                        del self.frames[:drop]
                        self.first_frame += drop
            next_time += step_seconds
            pause = next_time - time.perf_counter()
            if pause > 0:
                time.sleep(pause)
            else:
                next_time = time.perf_counter()  # running behind: do not try to catch up
        self.running = False

    def _step(self) -> None:
        env = self.env
        env.target_speed = self.speed
        action = self.brain.act(self.observation)
        if env.data.time < self.push_until:
            env.data.xfrc_applied[env.torso, :3] = self.push
        else:
            env.data.xfrc_applied[env.torso, :3] = 0.0
        self.observation, _, fell, _, _ = env.step(action)
        layout = env.scene.layout
        off_the_floor = layout is not None and env.data.qpos[0] > layout.x_max - 1.0
        if (fell or off_the_floor) and self.fallen_at is None:
            self.fallen_at = env.data.time
            self.events.append("fell" if fell else "end")
        if self.fallen_at is not None and env.data.time - self.fallen_at > STAND_UP_AFTER:
            self._restart()

    def _restart(self) -> None:
        self.seed += 1
        self.observation, _ = self.env.reset(seed=self.seed)
        self.fallen_at = None
        self.push_until = -1.0
        self.events.append("restart")

    # ---------- What the page asks for ----------

    def frames_after(self, number: int) -> dict:
        self.last_asked = time.time()
        with self.lock:
            start = max(number + 1, self.first_frame)
            frames = self.frames[start - self.first_frame:]
            events, self.events = self.events, []
            data = self.env.data
            state = {"x": float(data.qpos[0]) - self.env.start_x, "speed": float(data.qvel[0]),
                     "asked": self.speed, "fallen": self.fallen_at is not None, "time": float(data.time)}
        return {"first": start, "frames": frames, "events": events, "state": state, "running": self.running}

    def command(self, command: dict) -> None:
        self.last_asked = time.time()
        with self.lock:
            if "speed" in command:
                self.speed = float(min(max(float(command["speed"]), 0.0), 4.0))
            if command.get("push") in ("left", "right", "front", "back"):
                # The side the push comes from: from the left (+y) it shoves the walker toward -y
                direction = {"left": (0, -1), "right": (0, 1), "front": (-1, 0), "back": (1, 0)}[command["push"]]
                strength = PUSH_SHARE * self.env.scene.total_mass * self.world.values["gravity"]
                self.push = strength * np.array([direction[0], direction[1], 0.0])
                self.push_until = self.env.data.time + PUSH_SECONDS
            if command.get("restart"):
                self._restart()
            if command.get("stop"):
                self.running = False


def start(training_id: str, world_data: dict | None, speed: float) -> dict:
    folder = library.training_folder(training_id)
    if not (folder / "brain" / "best.zip").exists():
        raise ValueError("This walker has no trained brain yet.")
    with _lock:
        for old in list(_sessions.values()):  # one live walker at a time is plenty for a laptop
            old.running = False
        _sessions.clear()
        session = LiveSession(training_id, world_data, speed)
        _sessions[session.id] = session
    return session.header()


def session(session_id: str) -> LiveSession:
    found = _sessions.get(session_id)
    if found is None:
        raise ValueError("That live session has ended. Start it again.")
    return found
