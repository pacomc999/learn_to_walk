"""Quick looks for the page: a creature standing still, a drop test, a world, a trained walker.

All of them come back as replays (see studio/replay.py), so the page shows them in the
same 3D viewer.
"""
import threading

import mujoco
import numpy as np

from studio import library
from studio.brain import clean_design, count_weights, design_for_body, design_senses
from studio.creatures import animal, build_creature
from studio.env import WalkEnv, observation_size
from studio.replay import WalkerBrain, ground_patch, record, scene_shapes
from studio.scene import build_scene, set_stand_pose
from studio.worlds import DEFAULTS, World, world_from_dict

PLAIN_WORLD = World("preview", "Preview", "", dict(DEFAULTS))

# Loaded brains, by training id, with the time their file was written
_brains: dict[str, tuple[float, WalkerBrain]] = {}
_brain_lock = threading.Lock()


# The creature standing in its starting pose, with a few facts about it
def creature_preview(animal_key: str, values: dict) -> dict:
    build = build_creature(animal_key, values)
    scene = build_scene(build, PLAIN_WORLD)
    data = mujoco.MjData(scene.model)
    set_stand_pose(scene.model, data, build, scene.stand_height)
    mujoco.mj_forward(scene.model, data)
    env = _shell(build, scene)
    bodies, shapes = scene_shapes(env)
    frame = np.concatenate([data.xpos[1:], data.xquat[1:]], axis=1).ravel()
    centre = data.subtree_com[scene.model.body("torso").id]
    return {"bodies": bodies, "shapes": shapes, "frames": [[round(float(v), 4) for v in frame]], "fps": 25,
            "ground": {"kind": "plane"}, "style": "studio", "size": build.size, "stand_height": scene.stand_height,
            "facts": creature_facts(build, scene), "centre_of_mass": [round(float(v), 4) for v in centre],
            "support": _support_area(scene.model, data, build), "joints": joint_marks(scene.model, build),
            "view_size": _view_size(scene.model, data, build)}


# How big the creature looks, for framing the camera: its reach from the torso, where a
# 0.3 m reach counts as size 1. Only the camera uses it (build.size also scales the physics)
def _view_size(model, data, build) -> float:
    torso = data.xpos[model.body("torso").id]
    reach = 0.0
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        away = float(np.linalg.norm(data.geom_xpos[geom_id][:2] - torso[:2]) + np.max(model.geom_size[geom_id][:3]))
        reach = max(reach, away)
    return round(max(build.size, reach / 0.3), 3)


# Where every joint sits, for the motor rings in the 3D view. Each hinge with a motor gets a ring
# around its axis (kind: swing, side or twist, from the axis); a part with no hinge is "glued" on.
# Places and axes are in the part's own frame, so the marks move with the part.
JOINT_KINDS = ("side", "swing", "twist")  # axis x, y, z


def joint_marks(model, build) -> list[dict]:
    motors = set(build.motor_names)
    marks, torso = [], model.body("torso").id
    for body in range(1, model.nbody):
        hinges = [j for j in range(model.njnt)
                  if model.jnt_bodyid[j] == body and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        for joint in hinges:
            axis = model.jnt_axis[joint]
            name = model.joint(joint).name
            marks.append({"body": body - 1, "pos": _rounded(model.jnt_pos[joint]), "axis": _rounded(axis),
                          "kind": JOINT_KINDS[int(np.argmax(np.abs(axis)))], "name": name,
                          "motor": name in motors})
        has_shape = any(model.geom_bodyid[g] == body for g in range(model.ngeom))
        if not hinges and body != torso and has_shape and model.body_parentid[body] != 0:
            marks.append({"body": body - 1, "pos": [0.0, 0.0, 0.0], "kind": "glued",
                          "name": model.body(body).name, "motor": False})
    return marks


def _rounded(values) -> list[float]:
    return [round(float(v), 4) for v in values]


# Where the feet stand: the outline (convex hull) of the lowest points of the parts that feel the ground,
# or of every part that touches the floor at the start when none is marked
def _support_area(model, data, build) -> list[list[float]]:
    names = [name for foot in build.feet for name in foot.geoms]
    geoms = [model.geom(name).id for name in names] if names else []
    points = []
    for geom_id in geoms:
        centre = data.geom_xpos[geom_id]
        size = model.geom_size[geom_id]
        reach = float(np.max(size[:3])) if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_CAPSULE else float(size[0])
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            points.append((centre[0] + reach * np.cos(angle) * 0.5, centre[1] + reach * np.sin(angle) * 0.5))
    if len(points) < 3:
        return []
    return [[round(x, 4), round(y, 4)] for x, y in _hull(points)]


def _hull(points):
    points = sorted(set(points))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _shell(build, scene):
    """A minimal environment around a scene, only for listing shapes."""
    env = WalkEnv.__new__(WalkEnv)
    env.model, env.build, env.scene = scene.model, build, scene
    return env


def creature_facts(build, scene) -> dict:
    model = scene.model
    joints = [model.joint(motor.name) for motor in build.motors]
    strongest = max((float(model.actuator_forcerange[i][1]) for i in range(model.nu)), default=0.0)
    return {"mass_kg": round(scene.total_mass, 3), "height_cm": round(_top(scene) * 100, 1),
            "hip_height_cm": round(scene.stand_height * 100, 1), "motors": len(build.motors),
            "feet": len(build.feet), "motor_names": build.motor_names,
            "motor_groups": sorted({motor.group for motor in build.motors}),
            "strongest_motor_nm": round(strongest, 2),
            "senses": observation_size(build, {"feet": True, "ground": False}), "symmetric_ok": build.symmetric_ok,
            "senses_with_eyes": observation_size(build, {"feet": True, "ground": True}),
            "joint_ranges_deg": {joint.name: round(float(np.degrees(joint.range[1])), 1) for joint in joints}}


# The highest point of the standing creature above the ground
def _top(scene) -> float:
    model = scene.model
    data = mujoco.MjData(model)
    set_stand_pose(model, data, scene.build, scene.stand_height)
    mujoco.mj_forward(model, data)
    top = 0.0
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        top = max(top, float(data.geom_xpos[geom_id][2] + np.max(model.geom_size[geom_id][:3])))
    return top


# The creature placed in a world with its muscles holding the starting pose for a few seconds:
# does it stand, wobble or fall over?
def drop_test(animal_key: str, values: dict, world_data: dict | None = None, seconds: float = 4.0) -> dict:
    build = build_creature(animal_key, values)
    world = world_from_dict(world_data) if world_data else PLAIN_WORLD
    height = world.tallest if world.terrain != "flat" else 0.0
    env = WalkEnv(build, world, terrain_height=height, target_speed=float(np.mean(world.speed_range)))
    replay = record(env, None, seed=7, seconds=seconds)
    replay["stats"]["stood"] = not replay["stats"]["fell"]
    if world_data:  # show a good stretch of the ground ahead, so stairs, hurdles and slopes can be seen
        replay["ground"] = ground_patch(env, -1.5, 8.0)
        replay["marks_to"] = 8
    return replay


# A trained walker in any world: the best brain of a training session
def watch(training_id: str, world_data: dict | None, speed: float | None, height: float | None, seed: int,
          seconds: float | None, ground_half_width: float | None = None, turn: float = 0.0) -> dict:
    folder = library.training_folder(training_id)
    setup = library.read_json(folder / "setup.json")
    if not setup:
        raise ValueError("That training was not found.")
    if not (folder / "brain" / "best.zip").exists():
        raise ValueError("This walker has no trained brain yet. Wait for its first exam.")
    build = build_creature(setup["creature"]["animal"], setup["creature"]["values"])
    world = world_from_dict(world_data or setup["world"])
    senses = design_senses(clean_design(setup.get("brain")))
    low, high = world.speed_range
    speed = float(np.mean(world.speed_range)) if speed is None else float(speed)
    if height is None:
        height = world.tallest * 0.75 if world.values["mixed_heights"] else world.tallest
    env = WalkEnv(build, world, senses, target_speed=speed, terrain_height=float(height), turn_rate=float(turn or 0.0))
    brain = _brain(training_id)
    replay = record(env, brain.act, seed=int(seed), seconds=seconds, keep_going_after_fall=2.0,
                    ground_half_width=ground_half_width, brain=None if ground_half_width else brain)
    replay["name"] = setup.get("name", "")
    replay["creature_name"] = setup["creature"].get("name") or animal(setup["creature"]["animal"]).NAME
    return replay


# A race: several trained walkers in the same world, side by side in lanes, as one replay.
# Each walks in its own simulation on the same ground (same seed); the lanes only exist in the picture
def race(training_ids: list[str], world_data: dict, speed: float | None, height: float | None, seed: int,
         seconds: float) -> dict:
    ids = list(dict.fromkeys(training_ids))[:6]
    if len(ids) < 1:
        raise ValueError("Pick at least one walker for the race.")
    world_data = dict(world_data or {})
    world_data["values"] = {**(world_data.get("values") or {}), "attempt_seconds": max(
        float((world_data.get("values") or {}).get("attempt_seconds", 20)), float(seconds))}
    world = world_from_dict(world_data)
    entries = []
    for training_id in ids:
        setup = library.read_json(library.training_folder(training_id) / "setup.json") or {}
        size = build_creature(setup["creature"]["animal"], setup["creature"]["values"]).size if setup else 1.0
        entries.append((training_id, size))
    lane = max(0.9, max(size for _, size in entries) * 0.9)
    replays = [watch(training_id, world.to_dict(), speed, height, seed, seconds, ground_half_width=lane / 2)
               for training_id, _ in entries]
    return _merge_lanes(replays, lane)


def _merge_lanes(replays: list[dict], lane: float) -> dict:
    count = len(replays)
    offsets = [(count - 1) / 2 * lane - i * lane for i in range(count)]  # first walker on the left
    bodies, shapes, torsos, labels, grounds = [], [], [], [], []
    for replay, offset in zip(replays, offsets):
        first = len(bodies)
        torsos.append(first)
        bodies += [f"{name}" for name in replay["bodies"]]
        shapes += [{**shape, "body": shape["body"] + first} for shape in replay["shapes"]]
        short = replay["name"].replace(" (from learn_to_walk)", "")
        labels.append({"text": short, "body": first, "z": replay["stand_height"] * 2.4 + 0.12})
        ground = dict(replay["ground"])
        if ground.get("kind") == "heights":
            ground["y0"] = round(ground["y0"] + offset, 4)
        grounds.append(ground)
    length = max(len(replay["frames"]) for replay in replays)
    frames = []
    for index in range(length):
        frame = []
        for replay, offset in zip(replays, offsets):
            values = list(replay["frames"][min(index, len(replay["frames"]) - 1)])
            for start in range(0, len(values), 7):
                values[start + 1] = round(values[start + 1] + offset, 4)
            frame += values
        frames.append(frame)
    results = sorted(({"name": replay["name"], "creature": replay["creature_name"], **replay["stats"]}
                      for replay in replays), key=lambda result: -result["distance"])
    return {"fps": replays[0]["fps"], "bodies": bodies, "shapes": shapes, "frames": frames,
            "ground": {"kind": "lanes", "patches": grounds, "lane": lane, "count": count},
            "style": replays[0]["style"], "size": max(1.0, count * lane / 1.2),
            "stand_height": max(replay["stand_height"] for replay in replays), "torsos": torsos, "labels": labels,
            "results": results, "stats": {"seconds": max(replay["stats"]["seconds"] for replay in replays)}}


def _brain(training_id: str) -> WalkerBrain:
    folder = library.training_folder(training_id) / "brain"
    model_path, norm_path = folder / "best.zip", folder / "best_norm.npz"
    written = model_path.stat().st_mtime
    with _brain_lock:
        cached = _brains.get(training_id)
        if cached and cached[0] == written:
            return cached[1]
        brain = WalkerBrain(model_path, norm_path)
        _brains[training_id] = (written, brain)
        return brain


# Facts about a brain design for one creature: senses, motors and how many numbers it learns
def brain_facts(animal_key: str, values: dict, design: dict) -> dict:
    build = build_creature(animal_key, values)
    design = design_for_body(clean_design(design), build)
    senses = design_senses(design)
    inputs = observation_size(build, senses)
    return {"inputs": inputs, "outputs": len(build.motors), "motor_names": build.motor_names,
            "weights": count_weights(inputs, len(build.motors), design), "design": design,
            "left_motors": sum(1 for m in build.motors if m.side == "left"),
            "middle_motors": sum(1 for m in build.motors if m.side == "middle"),
            "animal_name": animal(animal_key).NAME, "symmetric_ok": build.symmetric_ok}
