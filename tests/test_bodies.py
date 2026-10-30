import mujoco
import pytest

from duck.bodies import STANDARD_BODY, body_from_dict, body_to_dict, body_xml, load_body
from duck.model import DUCK_XML_PATH


# The standard body is exactly today's duck, so every result stays the same
def test_standard_body_is_todays_duck():
    assert body_xml(STANDARD_BODY) == DUCK_XML_PATH.read_text()
    assert load_body("standard") == STANDARD_BODY


# A strong legged body doubles the hips and knees and leaves the other motors alone
def test_strong_legs_double_the_hips_and_knees():
    body = body_from_dict({"name": "strong_legs", "description": "x",
                           "motor_force_limits": {"hips": 8, "knees": 8}})
    model = mujoco.MjModel.from_xml_string(body_xml(body))
    assert list(model.actuator("left_hip_side").forcerange) == [-8, 8]
    assert list(model.actuator("right_knee").forcerange) == [-8, 8]
    assert list(model.actuator("left_ankle").forcerange) == [-3, 3]  # untouched


# Stiffness, a part's mass and the feet's grip can all change
def test_stiffness_mass_and_friction():
    body = body_from_dict({"name": "b", "description": "x", "motor_stiffness": {"left_knee": 30},
                           "part_masses": {"torso": 1.0}, "foot_friction": 1.5})
    model = mujoco.MjModel.from_xml_string(body_xml(body))
    assert model.actuator("left_knee").gainprm[0] == 30
    assert model.body("torso").mass[0] == pytest.approx(1.0)
    assert model.geom("left_toes").friction[0] == 1.5


# A body reads back as the same body
def test_body_round_trip():
    body = body_from_dict({"name": "b", "description": "x", "motor_force_limits": {"knees": 6},
                           "part_masses": {"left_thigh": 0.1}, "foot_friction": 0.8})
    assert body_from_dict(body_to_dict(body)) == body


# Mistakes in a body file are refused with a message that names the key
@pytest.mark.parametrize("data, word", [
    ({"motor_force_limits": {"wings": 5}}, "wings"),
    ({"part_masses": {"tail": 1}}, "tail"),
    ({"part_masses": {"torso": -1}}, "torso"),
    ({"size": 2}, "size"),
])
def test_bad_body_is_refused_plainly(data, word):
    with pytest.raises(ValueError, match=word):
        body_from_dict({"name": "b", "description": "x", **data}, source="bodies/test.json")
