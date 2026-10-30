"""Drive a duck: watch a brain walk live and change the speed it is asked for with a slider.

Double-click "docs/Drive the duck.bat", or run from the project root, for example:
    python -m duck.drive
    python -m duck.drive runs/duck_03_2/from_duck_03_1/best
    python -m duck.drive duck_catalogue/duck_03_1/brain --speed 0.4
Two windows open: the MuJoCo viewer, whose camera follows the duck, and a small control window
with the speed slider, the Reset button and what the duck is doing. The duck walks on the flat
floor for as long as you like (no 20 s limit). After a fall it lies still until you press Reset.
Close either window to stop. With no folder given it drives the newest brain: the most recently
saved best brain of any run, or kept brain in the catalogue.
"""
import argparse
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv
from duck.policy import choose_snapshot_or_exit, load_policy
from duck.snapshots import latest_snapshot

# The slider's range, in metres per second (training asked for 0.2 to 1.5 from Duck 3.2 on)
SLIDER_RANGE = (0.1, 2.0)
SLIDER_STEP = 0.05

# Real time seconds one control step should take (the control step is 0.02 s)
SECONDS_PER_STEP = 0.02

# The actual speed shown is the average over the last second (50 control steps)
SPEED_WINDOW_STEPS = 50


# ---------- The duck being driven (no windows, so it can be tested) ----------

class Driver:
    def __init__(self, policy, speed: float):
        self.policy = policy
        self.env = DuckWalkEnv(target_speed=speed, roughness=0.0)
        self.attempt = 0
        self.reset()

    # Stand the duck up again at the start, keeping the speed it is asked for now
    def reset(self) -> None:
        self.env.fixed_target_speed = self.env.target_speed
        self.observation, _ = self.env.reset(seed=self.attempt)
        self.attempt += 1
        self.fallen = False
        self.started_at = self.env.data.time
        self.recent_x = deque([float(self.env.data.qpos[0])], maxlen=SPEED_WINDOW_STEPS + 1)

    # Change the speed the duck is asked for; it senses the new speed at its next step
    def set_speed(self, speed: float) -> None:
        self.env.target_speed = speed

    # One control step: the brain moves the duck, or after a fall the motors just hold still
    def step(self) -> None:
        action = np.zeros(self.env.motor_count) if self.fallen else self.policy(self.observation)
        self.observation, _, fell, _, _ = self.env.step(action)  # the 20 s end is ignored here
        self.fallen = self.fallen or fell
        self.recent_x.append(float(self.env.data.qpos[0]))

    # Average forward speed over the last second, in metres per second
    def actual_speed(self) -> float:
        steps = len(self.recent_x) - 1
        return 0.0 if steps == 0 else (self.recent_x[-1] - self.recent_x[0]) / (steps * SECONDS_PER_STEP)

    # A line saying what the duck is doing
    def status(self) -> str:
        seconds = self.env.data.time - self.started_at
        if self.fallen:
            return f"Fallen after {seconds:.1f} s. Press Reset to stand it up again."
        return f"Walking for {seconds:.1f} s at {self.actual_speed():.2f} m/s (asked {self.env.target_speed:.2f})"


# The folder of the newest brain worth driving: the most recently saved best brain of any run,
# or kept brain in the catalogue; None if there is none
def newest_brain(project: Path = Path(".")) -> Path | None:
    folders = list(project.glob("runs/duck_*/*/best")) + list(project.glob("duck_catalogue/*/brain"))
    saved = [(latest_snapshot(folder).model_path.stat().st_mtime, folder) for folder in folders if latest_snapshot(folder)]
    return max(saved)[1] if saved else None


# ---------- The windows ----------

# The control window: the speed slider, the Reset button and the status line
def control_window(driver: Driver) -> tuple[tk.Tk, tk.StringVar]:
    root = tk.Tk()
    root.title("Drive the duck")
    root.attributes("-topmost", True)
    tk.Label(root, text="Asked speed (m/s)", font=("Segoe UI", 11)).pack(padx=16, pady=(12, 0))
    slider = tk.Scale(root, from_=SLIDER_RANGE[0], to=SLIDER_RANGE[1], resolution=SLIDER_STEP, orient="horizontal",
                      length=360, font=("Segoe UI", 10), command=lambda value: driver.set_speed(float(value)))
    slider.set(driver.env.target_speed)
    slider.pack(padx=16)
    tk.Label(root, text="Training asked for 0.2 to 1.5 m/s", fg="#666", font=("Segoe UI", 9)).pack()
    tk.Button(root, text="Reset", width=12, font=("Segoe UI", 11), command=driver.reset).pack(pady=10)
    status = tk.StringVar()
    tk.Label(root, textvariable=status, width=58, font=("Segoe UI", 10)).pack(padx=16, pady=(0, 12))
    return root, status


# Point the viewer's camera at the duck and keep following it, seen from its right side
def follow_the_duck(viewer, driver: Driver) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = driver.env.model.body("torso").id
    viewer.cam.distance = 1.2
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -12


# Run both windows until one is closed, keeping the duck in step with the clock
def drive(driver: Driver) -> None:
    root, status = control_window(driver)
    window_open = [True]
    root.protocol("WM_DELETE_WINDOW", lambda: window_open.__setitem__(0, False))
    with mujoco.viewer.launch_passive(driver.env.model, driver.env.data) as viewer:
        follow_the_duck(viewer, driver)
        while viewer.is_running() and window_open[0]:
            step_started = time.perf_counter()
            driver.step()
            viewer.sync()
            status.set(driver.status())
            root.update()  # the slider and the button react here
            time.sleep(max(0.0, SECONDS_PER_STEP - (time.perf_counter() - step_started)))
    root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a brain walk live and change its speed with a slider.")
    parser.add_argument("brain_dir", type=Path, nargs="?", default=None,
                        help="a folder with snapshots, for example runs/duck_03_2/from_duck_03_1/best (default: the newest brain)")
    parser.add_argument("--speed", type=float, default=EVALUATION_TARGET_SPEED, help="starting speed in m/s")
    arguments = parser.parse_args()
    brain_dir = arguments.brain_dir or newest_brain()
    if brain_dir is None:
        print("No brain found in runs/ or duck_catalogue/.")
        return
    snapshot = choose_snapshot_or_exit(brain_dir, None)
    print(f"Driving the brain in {brain_dir} from {snapshot.steps:,} steps. Close a window to stop.")
    driver = Driver(load_policy(snapshot), arguments.speed)
    drive(driver)
    driver.env.close()


if __name__ == "__main__":
    main()
