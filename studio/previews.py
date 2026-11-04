"""Quick looks for the page: a creature standing still, a drop test, a world, a trained walker.

All of them come back as replays (see studio/replay.py), so the page shows them in the
same 3D viewer.
"""
import threading

import mujoco
import numpy as np

from studio import library
from studio.brain import clean_design, count_weights, design_senses
from studio.creatures import animal, build_creature
from studio.env import WalkEnv, observation_size
from studio.replay import WalkerBrain, record, scene_shapes
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
    return {"bodies": bodies, "shapes": shapes, "frames": [[round(float(v), 4) for v in frame]], "fps": 25,
            "ground": {"kind": "plane"}, "style": "studio", "size": build.size, "stand_height": scene.stand_height,
            "facts": creature_facts(build, scene)}


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
            "senses": observation_size(build, {"feet": True, "ground": False}),
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
    return replay


# A trained walker in any world: the best brain of a training session
def watch(training_id: str, world_data: dict | None, speed: float | None, height: float | None, seed: int,
          seconds: float | None) -> dict:
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
    env = WalkEnv(build, world, senses, target_speed=speed, terrain_height=float(height))
    brain = _brain(training_id)
    return record(env, brain.act, seed=int(seed), seconds=seconds)


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
    design = clean_design(design)
    senses = design_senses(design)
    inputs = observation_size(build, senses)
    return {"inputs": inputs, "outputs": len(build.motors), "motor_names": build.motor_names,
            "weights": count_weights(inputs, len(build.motors), design), "design": design,
            "left_motors": sum(1 for m in build.motors if m.side == "left"),
            "middle_motors": sum(1 for m in build.motors if m.side == "middle"),
            "animal_name": animal(animal_key).NAME}
