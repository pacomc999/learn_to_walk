"""Tests for the duck model file and its helper functions."""
import mujoco
from duck.model import (load_duck_model, MOTOR_NAMES, STANDING_HEIGHT,
                        MIN_TORSO_HEIGHT, torso_height, torso_up_z, foot_contacts)


# The XML must be found even when Python runs from another folder
def test_model_loads_from_any_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert load_duck_model().nu == 12


# The motors come in the exact order the brain will use
def test_motor_names_and_order():
    model = load_duck_model()
    assert [model.actuator(i).name for i in range(model.nu)] == MOTOR_NAMES


# Wings are fixed to the body for now. "wing" is checked as a whole word,
# because "hip_swing" also contains the letters "wing"
def test_wings_have_no_joints():
    model = load_duck_model()
    names = [model.joint(i).name for i in range(model.njnt)]
    assert not any("wing" in name.split("_") for name in names)


# Videos use the tracking camera
def test_follow_camera_exists():
    assert load_duck_model().camera("follow") is not None


# A fall is when the torso drops below 60 percent of standing height
def test_min_height_is_60_percent_of_standing():
    assert MIN_TORSO_HEIGHT == 0.6 * STANDING_HEIGHT


# Holding the standing pose for 2 seconds keeps the duck up
def test_duck_stands_when_holding_pose():
    model = load_duck_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    while data.time < 2.0:
        mujoco.mj_step(model, data)
    assert torso_height(data) > 0.8 * STANDING_HEIGHT
    assert torso_up_z(data) > 0.9


# The duck holding its standing pose for a moment, ready for contact checks
def _settled_duck():
    model = load_duck_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    while data.time < 1.0:
        mujoco.mj_step(model, data)
    return model, data


# After settling, both feet are on the ground
def test_both_feet_touch_ground_after_settling():
    model, data = _settled_duck()
    assert foot_contacts(model, data) == (True, True)


# Standing on the toes alone still counts as the foot touching the ground
def test_toes_count_as_foot_contact():
    model, data = _settled_duck()
    for side in ("left_foot", "right_foot"):
        model.geom(side).contype = 0  # the main foot boxes can no longer touch the floor
    mujoco.mj_forward(model, data)
    assert foot_contacts(model, data) == (True, True)
