"""Bodies: the duck himself, as changes from the standard duck in duck/assets/duck.xml.

A body is a small JSON file in bodies/ that lists only what differs: how strong and how
stiff the motors are, how heavy the parts are, and how well the feet grip. Bodies change
numbers only, so every body has the same 12 motors and senses and every brain fits it.
A duck keeps a frozen copy of its body in its duck.json.
"""
import json
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

from duck.model import DUCK_XML_PATH, MOTOR_NAMES

# Where the body files live (the bodies folder at the project root)
BODIES_DIR = Path(__file__).parent.parent / "bodies"

# Motor groups, so a body file can say "hips" instead of four motor names
MOTOR_GROUPS = {
    "neck": ["neck_base", "neck_top"],
    "hips": ["left_hip_swing", "left_hip_side", "right_hip_swing", "right_hip_side"],
    "knees": ["left_knee", "right_knee"],
    "ankles": ["left_ankle", "right_ankle"],
    "toes": ["left_toes", "right_toes"],
}

# The shapes that touch the ground with the feet
FOOT_GEOMS = ["left_foot", "left_toes", "right_foot", "right_toes"]

# The parts whose mass a body can set: the duck's bodies in duck.xml
PART_NAMES = [element.get("name") for element in ElementTree.parse(DUCK_XML_PATH).iter("body")]

# The keys a body file may have
BODY_KEYS = {"name", "description", "motor_force_limits", "motor_stiffness", "part_masses", "foot_friction", "notes"}


# One body, read from its file. Empty changes mean the standard duck
@dataclass(frozen=True)
class Body:
    name: str
    description: str
    motor_force_limits: dict[str, float] = field(default_factory=dict)  # by group or motor name
    motor_stiffness: dict[str, float] = field(default_factory=dict)     # kp, by group or motor name
    part_masses: dict[str, float] = field(default_factory=dict)         # kg, by part name
    foot_friction: float | None = None                                  # sliding friction of feet and toes
    notes: str = ""


# ---------- Checking a file ----------

# A number above 0, or a plain refusal that names the file and the key
def _positive(value, source: str, key: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{source}: {key} must be a number above 0, not {value!r}")
    return float(value)


# A table of motor settings: every name a group or a motor, every value above 0
def _motor_table(table: dict, source: str, key: str) -> dict[str, float]:
    for name in table:
        if name not in MOTOR_GROUPS and name not in MOTOR_NAMES:
            raise ValueError(f"{source}: {key}.{name} is not a motor or a motor group "
                             f"({', '.join(MOTOR_GROUPS)}, or one motor name)")
    return {name: _positive(value, source, f"{key}.{name}") for name, value in table.items()}


# A table of part masses: every name a part of the duck, every mass above 0
def _mass_table(table: dict, source: str) -> dict[str, float]:
    for name in table:
        if name not in PART_NAMES:
            raise ValueError(f"{source}: part_masses.{name} is not a part of the duck ({', '.join(PART_NAMES)})")
    return {name: _positive(value, source, f"part_masses.{name}") for name, value in table.items()}


# ---------- Reading and writing ----------

# A body from the contents of its file, checked; source names the file in messages
def body_from_dict(data: dict, source: str = "body") -> Body:
    unknown = set(data) - BODY_KEYS
    if unknown:
        raise ValueError(f"{source}: {sorted(unknown)[0]} is not something a body can have")
    friction = data.get("foot_friction")
    return Body(
        name=data["name"],
        description=data["description"],
        motor_force_limits=_motor_table(data.get("motor_force_limits", {}), source, "motor_force_limits"),
        motor_stiffness=_motor_table(data.get("motor_stiffness", {}), source, "motor_stiffness"),
        part_masses=_mass_table(data.get("part_masses", {}), source),
        foot_friction=None if friction is None else _positive(friction, source, "foot_friction"),
        notes=data.get("notes", ""),
    )


# The file layout of a body: only the changes it has
def body_to_dict(body: Body) -> dict:
    data = {"name": body.name, "description": body.description}
    for key in ("motor_force_limits", "motor_stiffness", "part_masses"):
        if getattr(body, key):
            data[key] = dict(getattr(body, key))
    if body.foot_friction is not None:
        data["foot_friction"] = body.foot_friction
    data["notes"] = body.notes
    return data


# Read the body file with this name from the bodies folder
def load_body(name: str, folder: Path = BODIES_DIR) -> Body:
    path = folder / f"{name}.json"
    if not path.exists():
        raise ValueError(f"There is no body called {name} (looked for {path.as_posix()}).")
    return body_from_dict(json.loads(path.read_text()), source=path.as_posix())


# ---------- Building the duck ----------

# Whether the body changes anything at all
def _is_standard(body: Body) -> bool:
    return not (body.motor_force_limits or body.motor_stiffness or body.part_masses or body.foot_friction)


# The motor names a table entry stands for: a group's motors, or the one motor
def _motors(name: str) -> list[str]:
    return MOTOR_GROUPS.get(name, [name])


# Set an attribute on each named motor in the actuator section
def _set_motors(root, table: dict[str, float], attribute: str, text) -> None:
    motors = {element.get("name"): element for element in root.find("actuator")}
    for name, value in table.items():
        for motor in _motors(name):
            motors[motor].set(attribute, text(value))


# Give a part a new mass, shared over its own shapes in proportion to their old masses
def _set_part_mass(root, part: str, mass: float) -> None:
    body = next(element for element in root.iter("body") if element.get("name") == part)
    shapes = [geom for geom in body.findall("geom") if float(geom.get("mass", "0")) > 0]
    old_total = sum(float(geom.get("mass")) for geom in shapes)
    for geom in shapes:
        geom.set("mass", f"{mass * float(geom.get('mass')) / old_total:.6g}")


# Set the sliding friction of the feet and toes (the turning and rolling parts stay as the default)
def _set_foot_friction(root, friction: float) -> None:
    for geom in root.iter("geom"):
        if geom.get("name") in FOOT_GEOMS:
            geom.set("friction", f"{friction:g} 0.05 0.01")


# The duck XML for this body: the standard file untouched, or a copy with the body's numbers
def body_xml(body: Body) -> str:
    text = DUCK_XML_PATH.read_text()
    if _is_standard(body):
        return text
    parser = ElementTree.XMLParser(target=ElementTree.TreeBuilder(insert_comments=True))
    root = ElementTree.fromstring(text, parser=parser)
    _set_motors(root, body.motor_force_limits, "forcerange", lambda value: f"-{value:g} {value:g}")
    _set_motors(root, body.motor_stiffness, "kp", lambda value: f"{value:g}")
    for part, mass in body.part_masses.items():
        _set_part_mass(root, part, mass)
    if body.foot_friction is not None:
        _set_foot_friction(root, body.foot_friction)
    return ElementTree.tostring(root, encoding="unicode")


# The standard duck, as in duck/assets/duck.xml
STANDARD_BODY = load_body("standard")
