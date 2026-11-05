"""Writes reference answers from the Python studio for the JavaScript engine's tests.

    python tests/js/make_fixtures.py

The JavaScript tests (node --test tests/js) build the same things and must get exactly the same results.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studio.creatures import ANIMALS, build_creature  # noqa: E402
from studio.creatures import custom  # noqa: E402
from studio.creatures.kit import all_settings  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


# Random settings within each setting's range (and a few broken values the cleaning must fix)
def random_values(groups, rng: random.Random) -> dict:
    values = {}
    for key, setting in all_settings(groups).items():
        if setting.kind == "number":
            values[key] = round(rng.uniform(setting.minimum - 10, setting.maximum + 10), rng.choice([0, 1, 3]))
        elif setting.kind == "bool":
            values[key] = rng.random() < 0.5
        elif setting.kind == "choice":
            values[key] = rng.choice([value for value, _ in setting.choices] + ["nonsense"])
        else:
            values[key] = rng.choice(["#12ab9f", "#FFFFFF", "red", setting.default])
    return values


# A random custom body: parts hanging on random parents, with random shapes, joints and mirrors
def random_body(rng: random.Random) -> dict:
    parts = []
    for n in range(rng.randint(0, 9)):
        parent = rng.choice(["torso"] + [p["id"] for p in parts])
        parts.append({"id": f"p{n}", "name": f"Part {n}", "parent": parent, "shape": rng.choice(["stick", "ball", "plate"]),
                      "length": rng.uniform(0, 0.3), "thickness": rng.uniform(0.005, 0.05), "width": rng.uniform(0.02, 0.2),
                      "x": rng.uniform(-1, 1), "y": rng.choice([0, rng.uniform(-1, 1)]), "z": rng.uniform(-1, 1),
                      "along": rng.uniform(0, 1), "pitch": rng.uniform(-90, 90), "yaw": rng.uniform(-180, 180),
                      "joint": rng.choice(["fixed", "swing", "side", "twist", "hip"]), "motion": rng.uniform(5, 170),
                      "strength": rng.uniform(0.5, 50), "mirror": rng.random() < 0.4, "foot": rng.random() < 0.5,
                      "heaviness": rng.uniform(10, 500), "colour": "#336699"})
    torso = {"shape": rng.choice(["box", "ellipsoid", "ball", "capsule"]), "length": rng.uniform(0.05, 0.6),
             "width": rng.uniform(0.05, 0.4), "height": rng.uniform(0.03, 0.3), "weight": rng.uniform(0.1, 5),
             "colour": "#aa5500"}
    return {"template": "", "torso": torso, "parts": parts}


def build_facts(build) -> dict:
    return {"bodyXml": build.body_xml, "actuatorXml": build.actuator_xml,
            "motors": [{"name": m.name, "group": m.group, "mirror": m.mirror, "mirrorSign": m.mirror_sign, "side": m.side}
                       for m in build.motors],
            "feet": [{"name": f.name, "geoms": list(f.geoms), "mirror": f.mirror} for f in build.feet],
            "standPose": build.stand_pose, "stridePairs": [list(pair) for pair in build.stride_pairs],
            "materials": build.materials, "jointDamping": build.joint_damping, "jointArmature": build.joint_armature,
            "size": build.size, "torsoGeoms": list(build.torso_geoms), "symmetricOk": build.symmetric_ok,
            "motorType": build.motor_type}


def creature_cases() -> list[dict]:
    rng = random.Random(7)
    cases = []
    for key, module in ANIMALS.items():
        cases.append({"animal": key, "values": {}})
        for _ in range(12):
            values = random_values(module.SETTING_GROUPS, rng)
            if key == "custom":
                values["body"] = random_body(rng)
            cases.append({"animal": key, "values": values})
    for template_key, template in custom.TEMPLATES.items():
        cases.append({"animal": "custom", "values": {"body": {"template": template_key, "torso": template["torso"],
                                                              "parts": template["parts"]}}})
    for case in cases:
        case["expected"] = build_facts(build_creature(case["animal"], case["values"]))
    return cases


WORLD_CASES = [
    {},
    {"terrain": "cobblestones", "terrain_height": 3},
    {"terrain": "stairs", "terrain_height": 8, "backpack": True, "backpack_weight": 500},
    {"terrain": "slopes", "terrain_height": 10, "rocks": True, "rock_weight": 300, "fastest_speed": 2.5},
    {"gravity": 3.7, "ground_grip": 40, "rocks": True},
]


def scene_cases() -> list[dict]:
    from studio.scene import build_scene, scene_xml
    from studio.worlds import World, clean_values, SETTING_GROUPS as WORLD_GROUPS
    cases = []
    creatures = [("duck", {}), ("puppy", {"overall_size": 140}), ("custom", {"body": custom.TEMPLATES["bug"]}),
                 ("custom", {"body": custom.TEMPLATES["worm"]})]
    for (animal, values), world_values in zip(creatures * 2, WORLD_CASES * 2):
        world = World("w", "w", "", clean_values(WORLD_GROUPS, world_values))
        build = build_creature(animal, values)
        scene = build_scene(build, world)
        xml = scene_xml(build, world, scene.layout, scene.floor_capacity, scene.torso_top)
        cases.append({"animal": animal, "values": values, "world": world_values, "xml": xml,
                      "standHeight": scene.stand_height, "torsoTop": scene.torso_top, "totalMass": scene.total_mass})
    return cases


ENV_WORLDS = [
    {},
    {"end_stops": True, "level_body": True, "no_crouch": True, "smooth_moves": True, "lift_feet": True,
     "face_forward": True, "steady_ride": True, "fall_cost": 20, "speed_shape": "bell"},
    {"backpack": True, "gravity": 5.0, "speed_shape": "faster", "ground_grip": 60},
    {"fall_cost": 50},
]


# The same start for both sides: the exact standing pose, no random noise, a fixed asked speed
def exact_start(env, speed: float) -> None:
    import mujoco
    from studio.scene import set_stand_pose
    set_stand_pose(env.model, env.data, env.build, env.scene.stand_height)
    mujoco.mj_forward(env.model, env.data)
    env.target_speed = speed


def env_cases() -> list[dict]:
    import math
    from studio.env import WalkEnv
    from studio.worlds import World, clean_values, SETTING_GROUPS as WORLD_GROUPS
    cases = []
    creatures = [("duck", {}), ("puppy", {"motor_type": "position"}), ("custom", {"body": custom.TEMPLATES["table"]}),
                 ("custom", {"body": custom.TEMPLATES["worm"]})]  # the worm crawls: never falls, no points for staying up
    for (animal, values), world_values in zip(creatures, ENV_WORLDS):
        world = World("w", "w", "", clean_values(WORLD_GROUPS, world_values))
        env = WalkEnv(build_creature(animal, values), world, {"feet": True, "ground": True})
        env.reset(seed=1)
        exact_start(env, 0.45)
        steps = []
        for t in range(60):
            action = [math.sin(0.15 * t + m) for m in range(len(env.build.motors))]
            observation, reward, terminated, truncated, info = env.step(action)
            steps.append({"observation": [float(v) for v in observation], "reward": float(reward),
                          "terminated": bool(terminated), "parts": info["reward_parts"]})
        cases.append({"animal": animal, "values": values, "world": world_values, "steps": steps})
    return cases


# Free and symmetric brains with random weights, and what they answer for some random senses
def brain_cases() -> list[dict]:
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from studio.brain import clean_design, design_senses, policy_for
    from studio.env import WalkEnv
    from studio.export_web import brain_to_web
    from studio.mirror import mirror_tables
    from studio.replay import policy_cells
    from studio.worlds import World, clean_values, SETTING_GROUPS as WORLD_GROUPS
    world = World("w", "w", "", clean_values(WORLD_GROUPS, {}))
    cases = []
    for animal, symmetric, activation, layers in (("duck", True, "tanh", [64, 32]), ("puppy", False, "elu", [48]),
                                                  ("puppy", True, "relu", [40, 40, 24])):
        build = build_creature(animal, {})
        design = clean_design({"values": {"symmetric": symmetric, "activation": activation, "see_ground": True},
                               "layers": layers})
        senses = design_senses(design)
        mirror = mirror_tables(build, senses).to_dict() if symmetric else None
        policy, kwargs = policy_for(design, mirror, build.motor_type)
        venv = DummyVecEnv([lambda: WalkEnv(build, world, senses)])
        model = PPO(policy, venv, policy_kwargs=kwargs, device="cpu", seed=3, n_steps=64, batch_size=64)
        torch.manual_seed(5)
        with torch.no_grad():
            for parameter in model.policy.parameters():
                parameter.normal_(0.0, 0.3)
        brain = brain_to_web(model.policy, activation, mirror)
        rng = np.random.default_rng(9)
        observations = rng.normal(0, 1, (5, venv.observation_space.shape[0])).astype(np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(observations)
            actions = model.policy.get_distribution(tensor).distribution.mean.numpy()
            values = model.policy.predict_values(tensor).numpy()[:, 0]
            std = model.policy.get_distribution(tensor).distribution.stddev.numpy()[0]
        cells = [policy_cells(model.policy, obs) for obs in observations]
        cases.append({"brain": brain, "observations": observations.tolist(), "actions": actions.tolist(),
                      "values": values.tolist(), "std": std.tolist(), "cells": cells})
        venv.close()
    return cases


# One PPO study session of Stable-Baselines3 on a made-up batch: the brain before and after, and approx_kl
def learner_cases() -> list[dict]:
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from studio.brain import clean_design, design_senses, policy_for
    from studio.env import WalkEnv
    from studio.export_web import brain_to_web
    from studio.mirror import mirror_tables
    from studio.worlds import World, clean_values, SETTING_GROUPS as WORLD_GROUPS
    world = World("w", "w", "", clean_values(WORLD_GROUPS, {}))
    cases = []
    count = 64
    for animal, symmetric, activation, layers, entropy in (("duck", True, "tanh", [32, 16], 0.0),
                                                          ("puppy", False, "relu", [24], 0.01)):
        build = build_creature(animal, {})
        design = clean_design({"values": {"symmetric": symmetric, "activation": activation}, "layers": layers})
        senses = design_senses(design)
        mirror = mirror_tables(build, senses).to_dict() if symmetric else None
        policy, kwargs = policy_for(design, mirror, build.motor_type)
        venv = DummyVecEnv([lambda: WalkEnv(build, world, senses)])
        model = PPO(policy, venv, policy_kwargs=kwargs, device="cpu", seed=3, n_steps=count, batch_size=count,
                    n_epochs=1, learning_rate=1e-3, ent_coef=entropy)
        before = brain_to_web(model.policy, activation, mirror)
        rng = np.random.default_rng(11)
        size = venv.observation_space.shape[0]
        observations = rng.normal(0, 1, (count, size)).astype(np.float32)
        with torch.no_grad():
            distribution = model.policy.get_distribution(torch.as_tensor(observations))
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions).numpy() + rng.normal(0, 0.05, count)
        advantages = rng.normal(0, 1, count).astype(np.float32)
        returns = rng.normal(0, 1, count).astype(np.float32)
        buffer = model.rollout_buffer
        buffer.reset()
        buffer.observations[:, 0] = observations
        buffer.actions[:, 0] = actions.numpy()
        buffer.log_probs[:, 0] = log_probs
        buffer.advantages[:, 0] = advantages
        buffer.returns[:, 0] = returns
        buffer.values[:, 0] = 0.0
        buffer.full = True
        buffer.pos = count
        model._logger = __import__("stable_baselines3.common.logger", fromlist=["configure"]).configure(None, [])
        model.train()
        after = brain_to_web(model.policy, activation, mirror)
        cases.append({"design": {"symmetric": symmetric, "activation": activation, "layers": layers},
                      "inputSize": size, "motorCount": len(build.motors), "mirror": mirror, "entropy": entropy,
                      "before": before, "after": after, "observations": observations.ravel().tolist(),
                      "actions": actions.numpy().ravel().tolist(), "logProbs": log_probs.tolist(),
                      "advantages": advantages.tolist(), "returns": returns.tolist(),
                      "approxKl": float(model.logger.name_to_value["train/approx_kl"])})
        venv.close()
    return cases


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    cases = creature_cases()
    (FIXTURES / "creatures.json").write_text(json.dumps(cases), encoding="utf-8")
    print(f"Wrote {len(cases)} creature cases")
    scenes = scene_cases()
    (FIXTURES / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")
    print(f"Wrote {len(scenes)} scene cases")
    envs = env_cases()
    (FIXTURES / "envs.json").write_text(json.dumps(envs), encoding="utf-8")
    print(f"Wrote {len(envs)} environment cases")
    brains = brain_cases()
    (FIXTURES / "brains.json").write_text(json.dumps(brains), encoding="utf-8")
    print(f"Wrote {len(brains)} brain cases")
    learners = learner_cases()
    (FIXTURES / "learners.json").write_text(json.dumps(learners), encoding="utf-8")
    print(f"Wrote {len(learners)} learner cases")
