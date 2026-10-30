"""Make one video that shows every chosen snapshot walking, one after another.

Run from the project root, for example:
    python -m duck.video runs/duck_v1
    python -m duck.video runs/duck_v1 --seconds 6
The video is saved as evolution.mp4 inside the run folder.
"""
import argparse
from pathlib import Path
from typing import Callable

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv
from duck.policy import load_policy
from duck.snapshots import choose_for_video, list_snapshots

# The caption band covers this share of the picture height, and is this dark (0 clear, 1 black)
BAND_HEIGHT_SHARE = 0.1
BAND_DARKNESS = 0.65


# Step counts in friendly words: "untrained", "512 steps", "250 thousand steps", "1.2 million steps"
def _steps_in_words(steps: int) -> str:
    if steps == 0:
        return "untrained"
    if steps >= 1_000_000:
        millions = f"{steps / 1_000_000:.1f}".removesuffix(".0")
        return f"after {millions} million steps"
    if steps >= 1_000:
        thousands = round(steps / 1_000)
        if thousands == 1_000:  # 999,500 and above round up to a whole million
            return "after 1 million steps"
        return f"after {thousands} thousand steps"
    return f"after {steps} steps"


# The caption for one clip, with the position counted from 1
def caption_text(position: int, count: int, steps: int) -> str:
    return f"Snapshot {position} of {count}, {_steps_in_words(steps)}"


# A font scaled to the picture height (falls back to the plain default font on old Pillow)
def _caption_font(frame_height: int):
    font_size = max(10, int(frame_height * BAND_HEIGHT_SHARE * 0.6))
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


# Darken the bottom band of the picture
def _darken_bottom_band(frame: np.ndarray, band_height: int) -> np.ndarray:
    darkened = frame.copy()
    band = darkened[-band_height:].astype(np.float32) * (1.0 - BAND_DARKNESS)
    darkened[-band_height:] = band.astype(np.uint8)
    return darkened


# White text, centered, on a dark semi-transparent band at the bottom of the picture
def add_caption(frame: np.ndarray, text: str) -> np.ndarray:
    height, width = frame.shape[:2]
    band_height = max(16, int(height * BAND_HEIGHT_SHARE))
    image = Image.fromarray(_darken_bottom_band(frame, band_height))
    draw = ImageDraw.Draw(image)
    font = _caption_font(height)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_x = (width - (right - left)) / 2 - left
    text_y = height - band_height + (band_height - (bottom - top)) / 2 - top
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


# Walk one attempt and keep a picture each time simulated time reaches the next 1/fps mark.
# After a fall the duck keeps being simulated with zero actions, so it stays on screen lying down.
# on_frame, if given, is told the observation and whether the duck has fallen at every picture.
# roughness is the height of the cobblestones in metres (0.0 is the flat floor)
def record_clip(policy, seconds: float = 6.0, fps: int = 30, width: int = 640,
                height: int = 480, seed: int = 0,
                on_frame: Callable[[np.ndarray, bool], None] | None = None,
                roughness: float = 0.0) -> list[np.ndarray]:
    frame_count = round(seconds * fps)
    env = DuckWalkEnv(render_mode="rgb_array", target_speed=EVALUATION_TARGET_SPEED, width=width, height=height,
                     roughness=roughness)
    try:
        observation, _ = env.reset(seed=seed)
        frames = [env.render()]  # the first picture comes right after the reset
        if on_frame:
            on_frame(observation, False)
        fallen = False
        while len(frames) < frame_count:
            action = np.zeros(env.motor_count) if fallen else policy(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            fallen = fallen or terminated or truncated
            # Simulated time, not the clock, decides when a picture is taken
            if env.data.time >= len(frames) / fps - 1e-9:
                frames.append(env.render())
                if on_frame:
                    on_frame(observation, fallen)
        return frames
    finally:
        env.close()


# Record every chosen snapshot with its caption, one clip after another, and save one MP4
def make_video(run_dir: Path, out_path: Path | None = None, max_snapshots: int = 25,
               seconds_per_clip: float = 6.0, fps: int = 30, roughness: float = 0.0) -> Path:
    snapshots = list_snapshots(run_dir)
    if not snapshots:
        raise ValueError(f"No snapshots found in {run_dir}")
    chosen = choose_for_video(snapshots, max_snapshots)
    out_path = out_path or run_dir / "evolution.mp4"

    # Write each clip to disk as soon as it is made, so only one clip is in memory
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
    try:
        for position, snapshot in enumerate(chosen, start=1):
            print(f"Recording snapshot {position} of {len(chosen)}")
            clip = record_clip(load_policy(snapshot), seconds_per_clip, fps, roughness=roughness)
            caption = caption_text(position, len(chosen), snapshot.steps)
            for frame in clip:
                writer.append_data(add_caption(frame, caption))
    finally:
        writer.close()
    print(f"Saved the video to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a video of the duck learning to walk.")
    parser.add_argument("run_dir", type=Path, help="run folder, for example runs/duck_v1")
    parser.add_argument("--seconds", type=float, default=6.0, help="seconds per snapshot clip (default 6)")
    parser.add_argument("--roughness", type=float, default=0.0, help="height of the cobblestones in metres (default 0, a flat floor)")
    arguments = parser.parse_args()
    try:
        make_video(arguments.run_dir, seconds_per_clip=arguments.seconds, roughness=arguments.roughness)
    except ValueError as error:
        print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
