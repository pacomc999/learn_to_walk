"""Replays: a walk recorded as 3D frames that the web page plays back with three.js.

A replay holds the walker's shapes (and the rock's), the ground, and for every frame the
position and turn of every part. The page draws the shapes once and moves them each frame,
so a replay is small and plays smoothly in any browser, with the camera wherever the viewer
likes. This is also how the Creature screen shows a new body standing (or falling over).
"""
from pathlib import Path

import mujoco
import numpy as np

from studio.env import WalkEnv

# Frames per second of a replay: every second control step (the walker is controlled 50 times a second)
REPLAY_FPS = 25

GEOM_KINDS = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
}


def _round(values, places: int = 4) -> list[float]:
    return [round(float(value), places) for value in values]


# ---------- A trained brain, ready to act ----------

class WalkerBrain:
    """A saved brain and its sense scaling, ready to pick motor commands."""

    def __init__(self, model_path: Path, norm_path: Path):
        from stable_baselines3 import PPO  # imported here so the page server starts quickly
        self.model = PPO.load(str(model_path), device="cpu")
        scaling = np.load(norm_path)
        self.mean, self.var = scaling["mean"], scaling["var"]
        self.clip = float(scaling["clip"])

    def act(self, observation: np.ndarray) -> np.ndarray:
        scaled = np.clip((observation - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
        action, _ = self.model.predict(scaled.astype(np.float32)[None, :], deterministic=True)
        return action[0]


# Save the sense scaling of a VecNormalize next to a brain
def save_scaling(vec_normalize, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(temporary, mean=vec_normalize.obs_rms.mean, var=vec_normalize.obs_rms.var, clip=vec_normalize.clip_obs)
    temporary.replace(path)


# Motor commands that hold the standing pose. Springy muscles simply aim for it; direct muscles push back
# toward it like a firm spring, as hard as they can, which shows whether they are strong enough to carry the body
def holding_action(env: WalkEnv) -> np.ndarray:
    low = env.model.actuator_ctrlrange[:, 0]
    high = env.model.actuator_ctrlrange[:, 1]
    target = np.zeros(env.model.nu)
    for motor_id, motor in enumerate(env.build.motors):
        if env.model.actuator_biastype[motor_id] != mujoco.mjtBias.mjBIAS_NONE:
            target[motor_id] = env.build.stand_pose[motor.name]
    return np.clip((target - low) / (high - low) * 2 - 1, -1, 1)


def holder(env: WalkEnv):
    springy = np.array([env.model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_NONE for i in range(env.model.nu)])
    if springy.all():
        hold = holding_action(env)
        return lambda observation: hold
    targets = np.array([env.build.stand_pose[motor.name] for motor in env.build.motors])

    # A spring of the force limit per radian with a little damping (tuned so both animals stand;
    # more damping shakes, since the push only changes 50 times a second)
    def act(observation):
        angles = env.data.qpos[env.joint_qpos]
        speeds = env.data.qvel[env.joint_qvel]
        push = -(angles - targets) - speeds / 200
        return np.where(springy, holding_action(env), np.clip(push, -1, 1))
    return act


# ---------- Shapes and ground ----------

# Every shape of the walker (and the rock) with its body, size, place and colour
def scene_shapes(env: WalkEnv) -> tuple[list[str], list[dict]]:
    model = env.model
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    shapes = []
    for geom_id in range(model.ngeom):
        kind = GEOM_KINDS.get(int(model.geom_type[geom_id]))
        body = model.geom_bodyid[geom_id]
        if kind is None or body == 0:
            continue
        material = model.geom_matid[geom_id]
        colour = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom_id]
        shapes.append({"body": int(body - 1), "type": kind, "size": _round(model.geom_size[geom_id]),
                       "pos": _round(model.geom_pos[geom_id]), "quat": _round(model.geom_quat[geom_id]),
                       "rgba": _round(colour, 3), "name": model.geom(geom_id).name or ""})
    return bodies, shapes


# The ground near the walk, as a coarse grid of heights (tenths of a millimetre), or a plain plane
def ground_patch(env: WalkEnv, x_from: float, x_to: float) -> dict:
    layout = env.scene.layout
    if layout is None or env.heights is None:
        return {"kind": "plane"}
    step = 0.015 if env.world.terrain in ("cobblestones",) else 0.03
    x_from, x_to = max(layout.x_min, x_from), min(layout.x_max, x_to)
    y_half = min(layout.y_half, 0.9 * max(env.build.size, 1.0))
    xs = np.arange(x_from, x_to + step / 2, step)
    ys = np.arange(-y_half, y_half + step / 2, step)
    columns = np.clip(np.round((xs - layout.x_min) / layout.cell).astype(int), 0, layout.columns - 1)
    rows = np.clip(np.round((ys + layout.y_half) / layout.cell).astype(int), 0, layout.rows - 1)
    patch = env.heights[np.ix_(rows, columns)]
    return {"kind": "heights", "x0": round(float(xs[0]), 4), "y0": round(float(ys[0]), 4), "step": step,
            "rows": len(ys), "cols": len(xs), "data": np.round(patch * 10000).astype(int).ravel().tolist()}


# ---------- Recording ----------

# Run one attempt and record it. act(observation) picks the motor commands; None holds the pose
def record(env: WalkEnv, act=None, seed: int = 1000, seconds: float | None = None,
           keep_going_after_fall: float = 1.0) -> dict:
    observation, _ = env.reset(seed=seed)
    bodies, shapes = scene_shapes(env)
    steps_per_frame = round(50 / REPLAY_FPS)
    limit = env.max_steps if seconds is None else round(seconds / env.control_seconds)
    frames, total, fell_at = [], 0.0, None
    hold = holder(env)
    extra_steps = round(keep_going_after_fall / env.control_seconds)
    step = 0
    frames.append(_frame(env))
    while step < limit:
        action = hold(observation) if act is None else act(observation)
        observation, reward, fell, _, info = env.step(action)
        step += 1
        if fell_at is None:
            total += reward
        if step % steps_per_frame == 0:
            frames.append(_frame(env))
        if fell and fell_at is None:
            fell_at = step
            if act is not None:
                limit = min(limit, step + extra_steps)
        if not env._finite():
            break
    end = fell_at or step
    distance = _distance_at(env, frames, end, steps_per_frame)
    seconds_walked = end * env.control_seconds
    xs = [frame[0] for frame in frames]
    return {
        "fps": REPLAY_FPS, "bodies": bodies, "shapes": shapes, "frames": frames,
        "ground": ground_patch(env, min(xs) - 1.5, max(xs) + 2.5),
        "style": env.world.values["style"], "world": env.world.name, "animal": env.build.animal,
        "size": env.build.size, "stand_height": env.scene.stand_height,
        "stats": {"distance": round(distance, 3), "seconds": round(seconds_walked, 2),
                  "fell": fell_at is not None, "target_speed": round(env.target_speed, 3),
                  "average_speed": round(distance / max(seconds_walked, 1e-9), 3), "score": round(total, 1),
                  "terrain_height_cm": round(env.attempt_height * 100, 1), "rock_hits": env.rock_hits},
    }


# The forward distance at the moment the walker fell (or at the end)
def _distance_at(env: WalkEnv, frames: list, end_step: int, steps_per_frame: int) -> float:
    index = min(len(frames) - 1, end_step // steps_per_frame)
    return float(frames[index][0] - frames[0][0])


# The place and turn of every body except the world, flattened: x y z qw qx qy qz for each
def _frame(env: WalkEnv) -> list[float]:
    positions = env.data.xpos[1:]
    turns = env.data.xquat[1:]
    return _round(np.concatenate([positions, turns], axis=1).ravel())
