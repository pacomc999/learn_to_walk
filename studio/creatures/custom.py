"""Build your own: a creature made of parts (sticks, balls and plates) joined by motors.

The body design is a torso and a list of parts. Each part hangs on a parent (the torso or
another part), points in some direction, and is joined to its parent by a joint: fixed, or a
hinge that swings forward and back, sideways, or twists, or a hip-like joint that swings both
ways. Every hinge gets a motor. A part marked "mirrored" gets a twin on the other side,
together with everything that hangs on it, so the left/right symmetric brain can be used.

The design travels inside the creature's values as values["body"]; the other values are the
ordinary settings below (muscle kind, strength, grip).
"""
import math
import re

from studio.creatures.kit import (Build, Foot, Motor, Setting, SettingGroup, clean_values, geom, indent, motor_line,
                                  num, percent, vec)

KEY = "custom"
NAME = "Build your own"
DESCRIPTION = "Put sticks, balls and plates together with motors, then see if it can learn to move."

SETTING_GROUPS = (
    SettingGroup("muscles", "All muscles", "Settings for every motor at once. Each joint also has its own strength.", (
        Setting("motor_type", "Kind of muscle", "choice", "force",
                "Direct muscles push with the force the brain asks for and let the joint swing freely at zero. "
                "Springy muscles pull each joint toward an angle the brain asks for.",
                choices=(("force", "Direct (pushes with a force)"), ("position", "Springy (aims for an angle)"))),
        percent("strength", "Strength of every muscle", "Multiplies every joint's own strength.", minimum=10,
                maximum=1000, step=10),
        percent("stiffness", "Muscle springiness", "How hard springy muscles pull toward their angle.", minimum=25,
                maximum=1000, step=25),
        percent("foot_grip", "Grip", "How well its parts grip the ground.", minimum=10, maximum=300),
    )),
)

SHAPES = ("stick", "ball", "plate")
JOINTS = ("fixed", "swing", "side", "twist", "hip")
TORSO_SHAPES = ("box", "ellipsoid", "ball", "capsule")
MOST_PARTS = 24
DENSITY = 1000.0  # kilograms per cubic metre: parts weigh about as much as water

# Joint axes in the part's own frame, and the sign its angle gets in the mirror
AXES = {"swing": ((0, 1, 0), 1), "side": ((1, 0, 0), -1), "twist": ((0, 0, 1), -1)}


# ---------- Ready-made bodies ----------

def _part(pid, name, parent, shape="stick", length=0.12, thickness=0.015, x=0.0, y=0.0, z=0.0, along=1.0,
          pitch=90.0, yaw=0.0, joint="swing", motion=60.0, strength=3.0, mirror=False, foot=False, width=0.05,
          heaviness=100.0, colour="#4f46e5"):
    return {"id": pid, "name": name, "parent": parent, "shape": shape, "length": length, "thickness": thickness,
            "width": width, "x": x, "y": y, "z": z, "along": along, "pitch": pitch, "yaw": yaw, "joint": joint,
            "motion": motion, "strength": strength, "mirror": mirror, "foot": foot, "heaviness": heaviness,
            "colour": colour}


def _torso(shape="box", length=0.3, width=0.16, height=0.07, weight=1.0, colour="#f59e0b"):
    return {"shape": shape, "length": length, "width": width, "height": height, "weight": weight, "colour": colour}


TEMPLATES = {
    "table": {"name": "Four-legged table", "description": "A flat body on four two-part legs. A good first build.",
              "torso": _torso(),
              "parts": [
                  _part("front", "Front leg", "torso", x=0.8, y=1, z=-1, joint="hip", motion=50, length=0.12, mirror=True,
                        strength=10),
                  _part("front_low", "Front shin", "front", length=0.11, thickness=0.012, joint="swing", motion=70,
                        foot=True, strength=10),
                  _part("back", "Back leg", "torso", x=-0.8, y=1, z=-1, joint="hip", motion=50, length=0.12, mirror=True,
                        strength=10),
                  _part("back_low", "Back shin", "back", length=0.11, thickness=0.012, joint="swing", motion=70,
                        foot=True, strength=10),
              ]},
    "biped": {"name": "Stick walker", "description": "Two legs with knees and flat feet, like a little robot.",
              "torso": _torso("ellipsoid", 0.16, 0.18, 0.2, 1.2, "#22c55e"),
              "parts": [
                  _part("thigh", "Thigh", "torso", x=0, y=0.55, z=-0.8, joint="hip", motion=60, length=0.14,
                        thickness=0.02, strength=20, mirror=True),
                  _part("shin", "Shin", "thigh", length=0.14, thickness=0.016, joint="swing", motion=70, strength=20),
                  _part("foot", "Foot", "shin", shape="plate", length=0.1, thickness=0.01, width=0.06, pitch=0,
                        joint="swing", motion=40, strength=10, foot=True),
              ]},
    "bug": {"name": "Six-legged bug", "description": "Six legs that splay out to the sides. Very hard to tip over.",
            "torso": _torso("capsule", 0.36, 0.1, 0.1, 0.8, "#0ea5e9"),
            "parts": [
                _part(pid, name, "torso", x=x, y=1, z=-0.2, pitch=20, yaw=yaw, joint="hip", motion=40,
                      length=0.11, thickness=0.012, strength=8, mirror=True)
                for pid, name, x, yaw in (("leg1", "Front leg", 0.7, 50), ("leg2", "Middle leg", 0, 90),
                                          ("leg3", "Back leg", -0.7, 130))
            ] + [
                _part(f"{pid}_low", f"{name} tip", pid, pitch=80, yaw=yaw, joint="swing", motion=50, length=0.12,
                      thickness=0.01, strength=6, foot=True)
                for pid, name, yaw in (("leg1", "Front leg", 50), ("leg2", "Middle leg", 90), ("leg3", "Back leg", 130))
            ]},
    "hopper": {"name": "One-legged hopper", "description": "One springy leg under a round body. Can it learn to hop?",
               "torso": _torso("ball", 0.2, 0.2, 0.2, 1.0, "#ec4899"),
               "parts": [
                   _part("thigh", "Leg", "torso", z=-0.9, joint="swing", motion=60, length=0.16, thickness=0.02,
                         strength=25),
                   _part("shin", "Shin", "thigh", joint="swing", motion=70, length=0.16, thickness=0.016, strength=25),
                   _part("foot", "Foot", "shin", shape="plate", length=0.14, width=0.1, thickness=0.012, pitch=0,
                         joint="swing", motion=40, strength=12, foot=True),
               ]},
    "starter": {"name": "Just a body", "description": "A body with one pair of legs. Add the rest yourself.",
                "torso": _torso(),
                "parts": [_part("leg", "Leg", "torso", x=0, y=1, z=-1, joint="hip", motion=50, length=0.18,
                                foot=True, strength=12, mirror=True)]},
}


def default_body() -> dict:
    template = TEMPLATES["table"]
    return {"template": "table", "torso": dict(template["torso"]), "parts": [dict(part) for part in template["parts"]]}


# ---------- Cleaning a design ----------

def _number(value, low, high, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return min(max(number, low), high)


def _colour(value, fallback):
    text = str(value or "")
    return text if re.fullmatch(r"#[0-9a-fA-F]{6}", text) else fallback


def clean_body(body: dict | None) -> dict:
    body = body if isinstance(body, dict) else default_body()
    torso_in = body.get("torso") if isinstance(body.get("torso"), dict) else {}
    torso = {
        "shape": torso_in.get("shape") if torso_in.get("shape") in TORSO_SHAPES else "box",
        "length": _number(torso_in.get("length"), 0.04, 1.2, 0.3),
        "width": _number(torso_in.get("width"), 0.04, 1.0, 0.16),
        "height": _number(torso_in.get("height"), 0.02, 1.0, 0.07),
        "weight": _number(torso_in.get("weight"), 0.05, 20.0, 1.0),
        "colour": _colour(torso_in.get("colour"), "#f59e0b"),
    }
    parts, known = [], {"torso"}
    for raw in (body.get("parts") or [])[:MOST_PARTS]:
        if not isinstance(raw, dict):
            continue
        pid = re.sub(r"[^a-z0-9_]", "", str(raw.get("id", "")).lower())[:24]
        if not pid or pid in known or pid == "torso" or pid.startswith(("left_", "right_")):
            continue
        parent = raw.get("parent") if raw.get("parent") in known else "torso"
        parts.append({
            "id": pid, "name": str(raw.get("name") or pid)[:40], "parent": parent,
            "shape": raw.get("shape") if raw.get("shape") in SHAPES else "stick",
            "length": _number(raw.get("length"), 0.0, 1.0, 0.12),
            "thickness": _number(raw.get("thickness"), 0.004, 0.2, 0.015),
            "width": _number(raw.get("width"), 0.01, 0.5, 0.05),
            "x": _number(raw.get("x"), -1, 1, 0), "y": _number(raw.get("y"), -1, 1, 0),
            "z": _number(raw.get("z"), -1, 1, 0), "along": _number(raw.get("along"), 0, 1, 1),
            "pitch": _number(raw.get("pitch"), -90, 90, 90), "yaw": _number(raw.get("yaw"), -180, 180, 0),
            "joint": raw.get("joint") if raw.get("joint") in JOINTS else "swing",
            "motion": _number(raw.get("motion"), 5, 170, 60), "strength": _number(raw.get("strength"), 0.1, 100, 3),
            "mirror": bool(raw.get("mirror")), "foot": bool(raw.get("foot")),
            "heaviness": _number(raw.get("heaviness"), 10, 1000, 100), "colour": _colour(raw.get("colour"), "#4f46e5"),
        })
        known.add(pid)
    return {"template": str(body.get("template", "")), "torso": torso, "parts": parts}


# ---------- Building ----------

def build(values: dict | None = None) -> Build:
    values = values or {}
    settings = clean_values(SETTING_GROUPS, values)
    body = clean_body(values.get("body"))
    maker = _Maker(body, settings)
    return maker.build()


def _direction(pitch: float, yaw: float) -> tuple[float, float, float]:
    p, y = math.radians(pitch), math.radians(yaw)
    return (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p))


# The half sizes of a part, used to place what hangs on it
def _half_size(part: dict) -> tuple[float, float, float]:
    if part["shape"] == "ball":
        return (part["thickness"],) * 3
    if part["shape"] == "plate":
        return (part["length"] / 2, part["width"] / 2, part["thickness"] / 2)
    return (part["thickness"],) * 3


class _Maker:
    """Turns a cleaned design into MuJoCo XML, motors and feet."""

    def __init__(self, body: dict, settings: dict):
        self.body = body
        self.settings = settings
        self.parts = body["parts"]
        self.children = {}
        for part in self.parts:
            self.children.setdefault(part["parent"], []).append(part)
        self.motors: list[Motor] = []
        self.actuators: list[str] = []
        self.feet: list[Foot] = []
        self.materials = {"torso": _rgba(body["torso"]["colour"])}
        self.symmetric_ok = True

    def build(self) -> Build:
        torso = self.body["torso"]
        lines = ['<body name="torso" pos="0 0 0">', '  <freejoint name="root"/>', "  " + self._torso_geom(torso)]
        for part in self.children.get("torso", []):
            lines.append(indent(self._part_tree(part, "torso", side="middle"), 2))
            if part["mirror"]:
                lines.append(indent(self._part_tree(part, "torso", side="right"), 2))
        lines.append("</body>")
        self._order_motors()
        size = max(0.5, max(torso["length"], torso["width"]) / 0.3)
        pairs = [(m.name, m.mirror) for m in self.motors if m.side == "left" and m.name.endswith("_swing")
                 and self._top_level(m.name)]
        return Build(animal=KEY, body_xml="\n".join(lines), actuator_xml="\n".join(self.actuators),
                     motors=self.motors, feet=self.feet, stand_pose={m.name: 0.0 for m in self.motors},
                     stride_pairs=pairs[:2], materials=self.materials, size=size, torso_geoms=["torso"],
                     symmetric_ok=self.symmetric_ok, motor_type=self.settings["motor_type"])

    def _top_level(self, motor_name: str) -> bool:
        part_id = motor_name.removeprefix("left_").rsplit("_", 1)[0]
        return any(p["id"] == part_id and p["parent"] == "torso" for p in self.parts)

    def _torso_geom(self, torso: dict) -> str:
        half = (torso["length"] / 2, torso["width"] / 2, torso["height"] / 2)
        mass = torso["weight"]
        if torso["shape"] == "ball":
            return geom("torso", "sphere", (half[0],), mass, "torso")
        if torso["shape"] == "ellipsoid":
            return geom("torso", "ellipsoid", half, mass, "torso")
        if torso["shape"] == "capsule":
            radius = min(half[1], half[2])
            reach = max(half[0] - radius, 0.001)
            return geom("torso", "capsule", (radius,), mass, "torso", fromto=vec(-reach, 0, 0, reach, 0, 0))
        return geom("torso", "box", half, mass, "torso")

    # Where a child joins its parent, in the parent's frame
    def _attach_point(self, part: dict, parent: dict | None) -> tuple[float, float, float]:
        if parent is None:
            torso = self.body["torso"]
            half = (torso["length"] / 2, torso["width"] / 2, torso["height"] / 2)
            if torso["shape"] == "ball":
                half = (torso["length"] / 2,) * 3
            return (part["x"] * half[0], part["y"] * half[1], part["z"] * half[2])
        if parent["shape"] == "stick":
            direction = _direction(parent["pitch"], parent["yaw"])
            return tuple(part["along"] * parent["length"] * d for d in direction)
        half = _half_size(parent)
        centre = self._shape_centre(parent)
        return (centre[0] + part["x"] * half[0], centre[1] + part["y"] * half[1], centre[2] + part["z"] * half[2])

    # The centre of a part's shape, from its joint, in its own (unmirrored) frame
    def _shape_centre(self, part: dict) -> tuple[float, float, float]:
        direction = _direction(part["pitch"], part["yaw"])
        if part["shape"] == "stick":
            return tuple(part["length"] / 2 * d for d in direction)
        if part["shape"] == "plate":
            return tuple(part["length"] / 2 * d for d in direction)
        return tuple(part["length"] * d for d in direction)

    # A part and everything that hangs on it. side: "middle" (or the left copy of a mirrored part) or "right"
    def _part_tree(self, part: dict, parent_id: str, side: str, mirrored: bool = False) -> str:
        mirrored = mirrored or part["mirror"]
        if side == "middle" and mirrored:
            side = "left"
        flip = -1.0 if side == "right" else 1.0
        name = part["id"] if side == "middle" else f"{side}_{part['id']}"
        parent = None if parent_id == "torso" else next(p for p in self.parts if p["id"] == parent_id)
        position = self._mirrored(self._attach_point(part, parent), flip)
        if side == "middle" and not mirrored and abs(position[1]) > 1e-9:
            self.symmetric_ok = False  # an unpaired part off the middle line
        lines = [f'<body name="{name}" pos="{vec(*position)}">']
        for kind in self._joint_kinds(part):
            axis, mirror_sign = AXES[kind]
            joint = f"{name}_{kind}"
            limit = math.radians(part["motion"] / 2)
            lines.append(f'  <joint name="{joint}" type="hinge" axis="{vec(*axis)}" range="{vec(-limit, limit)}" '
                         f'limited="true"/>')
            self._add_motor(joint, part, kind, side, mirror_sign, limit)
        lines.append("  " + self._part_geom(part, name, flip))
        if part["foot"]:
            other = name if side == "middle" else f"{'right' if side == 'left' else 'left'}_{part['id']}"
            self.feet.append(Foot(name, (name,), other))
        for child in self.children.get(part["id"], []):
            lines.append(indent(self._part_tree(child, part["id"], side, mirrored), 2))
        lines.append("</body>")
        return "\n".join(lines)

    @staticmethod
    def _mirrored(point, flip: float):
        return (point[0], flip * point[1], point[2])

    @staticmethod
    def _joint_kinds(part: dict) -> list[str]:
        if part["joint"] == "fixed":
            return []
        if part["joint"] == "hip":
            return ["swing", "side"]
        return [part["joint"]]

    def _part_geom(self, part: dict, name: str, flip: float) -> str:
        material = f"m_{part['id']}"
        self.materials[material] = _rgba(part["colour"])
        direction = _direction(part["pitch"], flip * part["yaw"])
        grip = f'{num(self.settings["foot_grip"] / 100)} 0.05 0.01'
        weight = part["heaviness"] / 100 * DENSITY
        if part["shape"] == "stick":
            radius = part["thickness"]
            length = max(part["length"], 0.001)
            mass = weight * (math.pi * radius ** 2 * length + 4 / 3 * math.pi * radius ** 3)
            end = tuple(length * d for d in direction)
            return geom(name, "capsule", (radius,), mass, material, fromto=vec(0, 0, 0, *end), friction=grip)
        if part["shape"] == "ball":
            radius = part["thickness"]
            mass = weight * 4 / 3 * math.pi * radius ** 3
            centre = tuple(part["length"] * d for d in direction)
            return geom(name, "sphere", (radius,), mass, material, pos=vec(*centre), friction=grip)
        half = (max(part["length"], 0.01) / 2, part["width"] / 2, part["thickness"] / 2)
        mass = weight * 8 * half[0] * half[1] * half[2]
        centre = tuple(half[0] * d for d in direction)
        quat = _turn(part["pitch"], flip * part["yaw"])
        return geom(name, "box", half, mass, material, pos=vec(*centre), quat=vec(*quat), friction=grip)

    def _add_motor(self, joint: str, part: dict, kind: str, side: str, mirror_sign: int, limit: float) -> None:
        if side == "middle":
            mirror_name, sign = joint, (1 if kind == "swing" else -1)
        else:
            other = "right" if side == "left" else "left"
            mirror_name, sign = f"{other}_{joint.split('_', 1)[1]}", mirror_sign
        self.motors.append(Motor(joint, "custom", mirror_name, sign, side))
        strength = part["strength"] * self.settings["strength"] / 100
        stiffness = 5 * strength * self.settings["stiffness"] / 100
        self.actuators.append(motor_line(joint, limit, stiffness, strength, self.settings["motor_type"]))

    # The brain's order: middle motors, then the left side, then the right side in the same order
    def _order_motors(self) -> None:
        lines = dict(zip([m.name for m in self.motors], self.actuators))
        middle = [m for m in self.motors if m.side == "middle"]
        left = [m for m in self.motors if m.side == "left"]
        right_by_name = {m.name: m for m in self.motors if m.side == "right"}
        right = [right_by_name[m.mirror] for m in left]
        self.motors = middle + left + right
        self.actuators = [lines[m.name] for m in self.motors]


# A turn (quaternion w x y z) that points a shape's x axis along pitch (down) and yaw (left)
def _turn(pitch: float, yaw: float) -> tuple[float, float, float, float]:
    half_yaw, half_pitch = math.radians(yaw) / 2, math.radians(pitch) / 2
    yaw_q = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    pitch_q = (math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0)
    return _multiply(yaw_q, pitch_q)


def _multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _rgba(hex_colour: str) -> str:
    red, green, blue = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return vec(red, green, blue, 1.0)
