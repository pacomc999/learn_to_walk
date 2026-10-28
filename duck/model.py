"""Load the duck model and read simple facts about its state."""
from pathlib import Path

import mujoco

# Where the duck XML lives, found from this file so any working folder works
DUCK_XML_PATH = Path(__file__).parent / "assets" / "duck.xml"

# The 12 motors, in the same order as the actuators in the XML
MOTOR_NAMES = [
    "neck_base", "neck_top",
    "left_hip_swing", "left_hip_side", "left_knee", "left_ankle", "left_toes",
    "right_hip_swing", "right_hip_side", "right_knee", "right_ankle", "right_toes",
]


# Build the MuJoCo model from the XML file
def load_duck_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(DUCK_XML_PATH))


# Height of the torso above the ground, in metres
def torso_height(data: mujoco.MjData) -> float:
    return float(data.body("torso").xpos[2])


# How upright the torso is: 1.0 standing straight, 0.0 lying on its side
def torso_up_z(data: mujoco.MjData) -> float:
    rotation = data.body("torso").xmat  # 3x3 rotation, stored as 9 numbers
    return float(rotation[8])  # z part of the torso's own up axis


# Whether one geom is touching the floor right now
def _touches_floor(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str) -> bool:
    geom_id = model.geom(geom_name).id
    floor_id = model.geom("floor").id
    for contact in data.contact[:data.ncon]:
        if {contact.geom1, contact.geom2} == {geom_id, floor_id}:
            return True
    return False


# Whether the left and right feet touch the floor
def foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool]:
    left = _touches_floor(model, data, "left_foot")
    right = _touches_floor(model, data, "right_foot")
    return left, right


# Torso height in the "stand" keyframe, worked out once when this file loads
def _standing_height() -> float:
    model = load_duck_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    return torso_height(data)


STANDING_HEIGHT = _standing_height()

# Below this torso height the duck counts as fallen
MIN_TORSO_HEIGHT = 0.6 * STANDING_HEIGHT
