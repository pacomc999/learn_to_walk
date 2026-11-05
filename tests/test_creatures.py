"""The animals: they build, the standard duck is learn_to_walk's duck, and settings change what they should."""
import mujoco
import numpy as np
import pytest

from studio.creatures import ANIMALS, build_creature
from studio.creatures.kit import clean_values, default_values
from studio.scene import build_scene
from studio.worlds import DEFAULTS, World

PLAIN = World("plain", "Plain", "", dict(DEFAULTS))


def scene_for(animal, values=None):
    return build_scene(build_creature(animal, values or {}), PLAIN)


@pytest.mark.parametrize("animal", list(ANIMALS))
def test_every_animal_builds_with_its_defaults(animal):
    scene = scene_for(animal)
    assert scene.model.nu == len(scene.build.motors)
    assert scene.stand_height > 0.1
    assert scene.total_mass > 0.5


def test_standard_duck_matches_learn_to_walk():
    # The numbers of learn_to_walk's duck/assets/duck.xml (springy muscles)
    model = scene_for("duck", {"motor_type": "position"}).model
    assert model.body_subtreemass[model.body("torso").id] == pytest.approx(1.53)
    assert model.body("left_thigh").pos == pytest.approx([-0.075, 0.05, -0.0875])
    assert model.body("left_toes").pos == pytest.approx([0.065, 0, -0.016])
    assert model.joint("left_hip_side").range == pytest.approx([-0.4, 0.4])
    assert model.actuator("left_ankle").forcerange == pytest.approx([-3, 3])
    assert model.actuator("neck_base").gainprm[0] == pytest.approx(4)
    names = [model.actuator(i).name for i in range(model.nu)]
    assert names == ["neck_base", "neck_top", "left_hip_swing", "left_hip_side", "left_knee", "left_ankle", "left_toes",
                     "right_hip_swing", "right_hip_side", "right_knee", "right_ankle", "right_toes"]
    assert scene_for("duck").stand_height == pytest.approx(0.251)


def test_longer_legs_stand_taller_and_weigh_more():
    standard, long_legs = scene_for("duck"), scene_for("duck", {"leg_length": 150})
    assert long_legs.stand_height > standard.stand_height + 0.05
    assert long_legs.total_mass > standard.total_mass


def test_switching_off_toes_and_neck_removes_their_motors():
    build = build_creature("duck", {"has_toes": False, "neck_moves": False})
    assert len(build.motors) == 8
    assert "left_toes" not in build.motor_names


def test_puppy_without_a_tail_has_no_tail_motor():
    assert "tail" not in build_creature("puppy", {"tail_length": 0}).motor_names


def test_direct_muscles_are_the_default_and_match_the_force_motors_body():
    # learn_to_walk's bodies/force_motors.json: neck 10, hips 40, knees 40, ankles 30, toes 5, no spring
    model = scene_for("duck").model
    assert all(model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE for i in range(model.nu))
    for motor, limit in (("neck_base", 10), ("left_hip_side", 40), ("right_knee", 40), ("left_ankle", 30),
                         ("right_toes", 5)):
        assert model.actuator(motor).ctrlrange == pytest.approx([-limit, limit])


def test_bigger_animals_get_stronger_muscles():
    small = scene_for("puppy", {"overall_size": 50}).model
    big = scene_for("puppy", {"overall_size": 200}).model
    assert big.actuator("left_front_knee").forcerange[1] > 100 * small.actuator("left_front_knee").forcerange[1]


def test_every_custom_template_builds_symmetric():
    from studio.creatures.custom import TEMPLATES
    for key, template in TEMPLATES.items():
        build = build_creature("custom", {"body": {"torso": template["torso"], "parts": template["parts"]}})
        assert build.symmetric_ok, key
        assert build.motors and build.feet, key
        names = build.motor_names
        assert all(motor.mirror in names for motor in build.motors), key


def test_a_lopsided_custom_creature_cannot_use_the_symmetric_brain():
    from studio.brain import clean_design, design_for_body
    body = {"parts": [{"id": "arm", "parent": "torso", "y": 1, "mirror": False}]}
    build = build_creature("custom", {"body": body})
    assert not build.symmetric_ok
    assert design_for_body(clean_design({}), build)["values"]["symmetric"] is False


@pytest.mark.parametrize("animal", ["duck", "puppy"])
def test_settings_are_kept_in_range(animal):
    groups = ANIMALS[animal].SETTING_GROUPS
    values = clean_values(groups, {"overall_size": 99999, "body_length": "nonsense"})
    defaults = default_values(groups)
    assert values["body_length"] == defaults["body_length"]
    assert values["overall_size"] <= 300


@pytest.mark.parametrize("animal", list(ANIMALS))
def test_every_extreme_setting_still_builds(animal):
    groups = ANIMALS[animal].SETTING_GROUPS
    for group in groups:
        for setting in group.settings:
            if setting.kind != "number":
                continue
            for value in (setting.minimum, setting.maximum):
                scene = scene_for(animal, {setting.key: value})
                data = mujoco.MjData(scene.model)
                mujoco.mj_step(scene.model, data)
                assert np.all(np.isfinite(data.qpos)), (setting.key, value)


def test_a_mirrored_pair_can_hang_on_a_middle_part():
    from studio.creatures.custom import TEMPLATES
    body = {"torso": TEMPLATES["hopper"]["torso"], "parts": [dict(part) for part in TEMPLATES["hopper"]["parts"]]}
    body["parts"].append({"id": "toe", "name": "Toe", "parent": "foot", "x": 1, "y": 1, "z": 0, "pitch": 0,
                          "yaw": 30, "joint": "side", "mirror": True, "foot": True})
    build = build_creature("custom", {"body": body})
    assert {"left_toe_side", "right_toe_side"} <= set(build.motor_names)
    assert build.symmetric_ok
    scene_for("custom", {"body": body})


def test_custom_parts_that_point_anywhere_build_and_stand_on_the_ground():
    rng = __import__("random").Random(4)
    for attempt in range(25):
        parts = []
        for index in range(rng.randint(1, 8)):
            parent = rng.choice(["torso"] + [part["id"] for part in parts])
            parts.append({"id": f"p{index}", "parent": parent, "shape": rng.choice(["stick", "ball", "plate"]),
                          "length": rng.uniform(0, 0.3), "x": rng.uniform(-1, 1), "y": rng.uniform(-1, 1),
                          "z": rng.uniform(-1, 1), "pitch": rng.uniform(-90, 90), "yaw": rng.uniform(-180, 180),
                          "joint": rng.choice(["fixed", "swing", "side", "twist", "hip"]),
                          "mirror": rng.random() < 0.5, "foot": rng.random() < 0.5})
        scene = scene_for("custom", {"body": {"parts": parts}})
        data = mujoco.MjData(scene.model)
        from studio.scene import set_stand_pose
        set_stand_pose(scene.model, data, scene.build, scene.stand_height)
        mujoco.mj_forward(scene.model, data)
        for _ in range(50):
            mujoco.mj_step(scene.model, data)
        assert np.all(np.isfinite(data.qpos)), attempt


def test_every_ready_made_body_stands_in_the_drop_test():
    from studio.creatures.custom import TEMPLATES
    from studio.env import WalkEnv
    from studio.replay import record
    from studio.worlds import builtin_worlds
    for key, template in TEMPLATES.items():
        if key == "starter":
            continue  # one pair of legs side by side: tipping over is the point
        env = WalkEnv(build_creature("custom", {"body": template}), builtin_worlds()[0], target_speed=0.4)
        assert not record(env, None, seed=7, seconds=3)["stats"]["fell"], key


def test_a_creature_without_motors_can_still_be_shown():
    from studio import previews
    assert previews.creature_preview("custom", {"body": {"parts": []}})["facts"]["motors"] == 0
    assert previews.drop_test("custom", {"body": {"parts": []}}, None, 0.5)["frames"]


def test_the_preview_marks_every_motor_and_every_glued_part():
    from studio.creatures.custom import TEMPLATES
    from studio.previews import creature_preview
    bug = TEMPLATES["bug"]
    preview = creature_preview("custom", {"body": {"torso": bug["torso"], "parts": bug["parts"]}})
    rings = [mark for mark in preview["joints"] if mark["motor"]]
    assert len(rings) == preview["facts"]["motors"] == 18  # six hips with two motors, six knees with one
    assert {mark["kind"] for mark in rings} == {"swing", "side"}
    glued = dict(bug["parts"][-1], id="horn", parent="torso", joint="fixed", mirror=False, y=0)
    preview = creature_preview("custom", {"body": {"torso": bug["torso"], "parts": bug["parts"] + [glued]}})
    assert [mark["name"] for mark in preview["joints"] if mark["kind"] == "glued"] == ["horn"]


def test_a_crawler_never_falls_and_earns_no_points_for_staying_up():
    import mujoco
    from studio.creatures.custom import TEMPLATES
    from studio.env import WalkEnv
    worm = TEMPLATES["worm"]
    build = build_creature("custom", {"body": {"torso": worm["torso"], "parts": worm["parts"], "crawler": worm["crawler"]}})
    assert build.crawler
    assert not build_creature("custom", {"body": TEMPLATES["table"]}).crawler and not build_creature("duck", {}).crawler
    env = WalkEnv(build, PLAIN)
    env.reset(seed=1)
    env.data.qpos[3:7] = [0.7071, 0.7071, 0, 0]  # rolled right over onto its side
    mujoco.mj_forward(env.model, env.data)
    _, _, fell, _, info = env.step([0.0] * len(build.motors))
    assert not fell
    assert info["reward_parts"]["alive"] == 0.0
