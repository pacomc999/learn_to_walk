"""Bring trained ducks over from a learn_to_walk folder.

Every catalogue duck with a kept brain (duck_catalogue/<duck>/brain/snapshots/) becomes a walker in
the studio: its body and world settings are turned into studio settings, its brain's weights go
into the studio's symmetric brain (the networks have the same layers and names), and its sense
scaling comes along. It then takes a studio exam, so its page has a best walk to show. The
learn_to_walk folder is only read, never changed.
"""
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch

from studio import library
from studio.brain import clean_design, design_senses, policy_for
from studio.creatures import build_creature
from studio.creatures.duck import FORCE, FORCE_MOTOR_BOOST, STIFFNESS
from studio.env import WalkEnv
from studio.learning import clean_learning, ppo_settings
from studio.mirror import MirroredRunningMeanStd, mirror_tables
from studio.worlds import world_from_dict

# learn_to_walk's motor groups (bodies.py) and the studio duck's
GROUPS = {"neck": "neck", "hips": "hip", "knees": "knee", "ankles": "ankle", "toes": "toes"}
SETTING = {"neck": "neck_strength", "hip": "hip_strength", "knee": "knee_strength", "ankle": "ankle_strength",
           "toes": "toe_strength"}


# ---------- Finding the ducks ----------


def find_ducks(folder: str) -> list[dict]:
    root = Path(folder).expanduser()
    catalogue = root / "duck_catalogue" if (root / "duck_catalogue").is_dir() else root
    if not catalogue.is_dir():
        raise ValueError(f"No learn_to_walk folder at {folder} (looked for its duck_catalogue folder).")
    ducks = []
    for duck_dir in sorted(catalogue.iterdir()):
        recipe = duck_dir / "duck.json"
        if not recipe.exists():
            continue
        data = json.loads(recipe.read_text(encoding="utf-8"))
        snapshot = _brain_snapshot(duck_dir)
        ducks.append({"folder": str(duck_dir), "key": duck_dir.name, "name": data.get("name", duck_dir.name),
                      "summary": data.get("summary", ""), "world": data.get("world"), "body": data.get("body"),
                      "has_brain": snapshot is not None, "kind": "duck"})
    ducks += _find_runs(root, catalogue)
    if not ducks:
        raise ValueError("That folder has no catalogue ducks.")
    return ducks


# The runs of each catalogue duck (runs/duck_NN/<run>/, only on the computer that trained them): their best
# brain, in the world and body frozen in their run_settings.json
def _find_runs(root: Path, catalogue: Path) -> list[dict]:
    runs = []
    for run_dir in sorted((root / "runs").glob("duck_*/*")):
        settings_path = run_dir / "run_settings.json"
        if not run_dir.is_dir() or not settings_path.exists():
            continue
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        recipe = catalogue / run_dir.parent.name / "duck.json"
        duck_name = json.loads(recipe.read_text(encoding="utf-8")).get("name") if recipe.exists() else run_dir.parent.name
        runs.append({"folder": str(run_dir), "key": f"{run_dir.parent.name}/{run_dir.name}",
                     "name": f"{duck_name}, run {run_dir.name}", "summary": "",
                     "world": (settings.get("world") or {}).get("name"), "body": (settings.get("body") or {}).get("name"),
                     "has_brain": _brain_snapshot(run_dir / "best") is not None, "kind": "run"})
    return runs


def _brain_snapshot(duck_dir: Path) -> Path | None:
    folder = duck_dir / "snapshots" if (duck_dir / "snapshots").is_dir() else duck_dir / "brain" / "snapshots"
    snapshots = sorted(folder.glob("snapshot_*.zip"))
    snapshots = [path for path in snapshots if path.with_name(path.stem + "_norm.pkl").exists()]
    return snapshots[-1] if snapshots else None


# ---------- Settings ----------

# A learn_to_walk body as studio duck settings
def creature_values(body: dict) -> dict:
    values = {"motor_type": body.get("motor_type", "position")}
    boost = FORCE_MOTOR_BOOST if values["motor_type"] == "force" else 1
    for group, limit in (body.get("motor_force_limits") or {}).items():
        studio_group = GROUPS.get(group)
        if studio_group:
            values[SETTING[studio_group]] = round(100 * limit / (FORCE[studio_group] * boost), 3)
    stiffness = body.get("motor_stiffness") or {}
    if stiffness:
        ratios = [value / STIFFNESS[GROUPS[group]] for group, value in stiffness.items() if group in GROUPS]
        if ratios:
            values["stiffness"] = round(100 * float(np.mean(ratios)), 3)
    if body.get("foot_friction"):
        values["foot_grip"] = round(100 * body["foot_friction"], 3)
    return values


# A learn_to_walk world as studio world settings
def world_values(world: dict) -> dict:
    floor = world.get("floor") or {}
    reward = world.get("reward") or {}
    low, high = world.get("target_speed_range") or [0.2, 0.7]
    tallest = float(floor.get("tallest_stones", 0.0))
    values = {"slowest_speed": low, "fastest_speed": high, "speed_points": reward.get("forward", 3.0),
              "alive_points": reward.get("alive", 0.2), "energy_cost": reward.get("energy", 1.5),
              "straight_cost": reward.get("sideways", 0.5), "end_stops": bool(reward.get("end_stops")),
              "end_stop_cost": reward.get("end_stops", 0.05) or 0.05}
    if tallest > 0:
        values.update(terrain="cobblestones", terrain_height=round(tallest * 100, 2), style="street")
    extras = world.get("extras") or {}
    if "rocks" in extras:
        rocks = extras["rocks"]
        values.update(rocks=True, rock_weight=1000 * rocks["mass"], rock_every=rocks.get("every_seconds", 2.0))
    if "backpack" in extras:
        values.update(backpack=True, backpack_weight=1000 * extras["backpack"]["mass"])
    return values


# ---------- Brains ----------

class _Unpickler(pickle.Unpickler):
    """Reads learn_to_walk's sense scaling without learn_to_walk installed."""

    def find_class(self, module, name):
        if module.startswith("duck."):
            if name == "MirroredRunningMeanStd":
                return _LegacyRunningMeanStd
            raise pickle.UnpicklingError(f"Unexpected learn_to_walk class {module}.{name}")
        return super().find_class(module, name)


class _LegacyRunningMeanStd:
    """Stands in for learn_to_walk's MirroredRunningMeanStd: only its numbers are needed."""

    def __setstate__(self, state):
        self.__dict__.update(state)


def _read_scaling(path: Path):
    with path.open("rb") as handle:
        return _Unpickler(handle).load()


# ---------- Importing ----------

def import_duck(duck_folder: str) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from studio.replay import save_scaling
    from studio.train import Folder, Status, best_walk_replay, take_exam, EXAM_HEADER

    duck_dir = Path(duck_folder)
    data, snapshot, world_settings, body_settings = _source(duck_dir)
    if snapshot is None:
        raise ValueError(f"{data.get('name', duck_dir.name)} has no kept brain to bring over.")
    creature = creature_values(body_settings)
    world = world_from_dict({"id": f"ltw_{world_settings.get('name', 'world')}",
                             "name": _title(world_settings.get("name", "learn_to_walk world")),
                             "description": world_settings.get("description", ""),
                             "values": world_values(world_settings)})
    design = clean_design({"layers": [256, 256], "values": {"symmetric": True, "activation": "tanh",
                                                            "feel_feet": True, "see_ground": False}})
    build = build_creature("duck", creature)
    senses = design_senses(design)
    mirror = mirror_tables(build, senses)

    # The studio brain, with learn_to_walk's weights
    venv = DummyVecEnv([lambda: WalkEnv(build, world, senses)])
    policy, kwargs = policy_for(design, mirror.to_dict(), build.motor_type)
    learning = clean_learning({"parallel_worlds": 8})
    model = PPO(policy, venv, policy_kwargs=kwargs, device="cpu", **ppo_settings(learning))
    with zipfile.ZipFile(snapshot) as archive:
        weights = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=False)
        saved = json.loads(archive.read("data"))
    model.policy.load_state_dict(weights, strict=True)
    model.num_timesteps = int(saved.get("num_timesteps", 0))
    # The brain keeps its own step size (learn_to_walk's adaptive rate settles near 1e-5): with its small wobble,
    # the default step size would wreck it within a few lessons
    rate = saved.get("learning_rate")
    if isinstance(rate, (int, float)) and rate > 0:
        model.learning_rate = float(rate)
        model._setup_lr_schedule()

    old = _read_scaling(snapshot.with_name(snapshot.stem + "_norm.pkl"))
    scaling = VecNormalize(venv, norm_obs=True, norm_reward=True)
    scaling.obs_rms = MirroredRunningMeanStd(mirror.observation_order, mirror.observation_signs,
                                             shape=venv.observation_space.shape)
    scaling.obs_rms.mean, scaling.obs_rms.var = np.array(old.obs_rms.mean), np.array(old.obs_rms.var)
    scaling.obs_rms.count = old.obs_rms.count
    scaling.ret_rms.mean, scaling.ret_rms.var = old.ret_rms.mean, old.ret_rms.var
    scaling.clip_obs = getattr(old, "clip_obs", 10.0)
    scaling.training = False

    # The walker's folder, as a finished training
    name = f"{data.get('name', duck_dir.name)} (from learn_to_walk)"
    training_id = library.new_id(data.get("name", duck_dir.name))
    path = library.training_folder(training_id)
    setup = {"id": training_id, "name": name, "created": library.now(),
             "creature": {"animal": "duck", "name": data.get("name", "Duck"), "values": creature},
             "world": world.to_dict(), "brain": design, "learning": learning, "start_from": None,
             "imported_from": str(duck_dir), "summary": data.get("summary", "")}
    library.write_json(path / "setup.json", setup)
    folder = Folder(path)
    for which in ("best", "latest"):
        model.save(folder.brain / f"{which}.zip")
        scaling.save(str(folder.brain / f"{which}_scaling.pkl"))
        save_scaling(scaling, folder.brain / f"{which}_norm.npz")

    # A studio exam and a best walk to show
    from studio.scene import build_scene
    scene = build_scene(build, world)
    result = take_exam(model, scaling, build, world, senses, scene)
    row = {"time": library.now(), "steps": model.num_timesteps,
           **{key: round(value, 3) if isinstance(value, float) else value for key, value in result.items()}}
    folder.append_row("exams.csv", EXAM_HEADER, row)
    replay = best_walk_replay(model, scaling, build, world, senses, scene)
    replay["steps"], replay["exam"] = model.num_timesteps, row
    folder.write_json("replays/best.json", replay)
    folder.write_json(f"replays/exam_{model.num_timesteps:012d}.json", replay)
    status = Status(folder, model.num_timesteps)
    status.update(state="finished", steps=model.num_timesteps, best_score=result["score"],
                  best_steps=model.num_timesteps, best_quality=result["quality"], last_exam=row,
                  message=f"Brought over from learn_to_walk ({duck_dir.name}).")
    (path / "log.txt").write_text(f"Imported from {duck_dir}: {snapshot.name}\n", encoding="utf-8")
    venv.close()
    return training_id


# The world the kept brain was measured in: results.json names it when it differs from the duck's
# current world (Duck 3.3 kept his walking brain while his world moved on to running)
def _kept_brain_world(duck_dir: Path, data: dict) -> dict:
    results_path = duck_dir / "results.json"
    if results_path.exists():
        name = json.loads(results_path.read_text(encoding="utf-8")).get("exam_world")
        world_file = duck_dir.parent.parent / "worlds" / f"{name}.json"
        if name and world_file.exists():
            return json.loads(world_file.read_text(encoding="utf-8"))
    return data.get("world_settings") or {}


# What to bring over from a catalogue duck or from one of its runs: its story, best brain, world and body
def _source(folder: Path) -> tuple[dict, Path | None, dict, dict]:
    if (folder / "run_settings.json").exists():
        settings = json.loads((folder / "run_settings.json").read_text(encoding="utf-8"))
        recipe = folder.parent.parent.parent / "duck_catalogue" / folder.parent.name / "duck.json"
        story = json.loads(recipe.read_text(encoding="utf-8")) if recipe.exists() else {}
        story = {**story, "name": f"{story.get('name', folder.parent.name)}, run {folder.name}"}
        return story, _brain_snapshot(folder / "best"), settings.get("world") or {}, settings.get("body") or {}
    story = json.loads((folder / "duck.json").read_text(encoding="utf-8"))
    return story, _brain_snapshot(folder), _kept_brain_world(folder, story), story.get("body_settings") or {}


def _title(name: str) -> str:
    return name.replace("_", " ").capitalize()
