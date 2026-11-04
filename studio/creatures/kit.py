"""The building kit every animal uses: settings the screens can show, and the finished build.

An animal is a recipe that turns a dictionary of settings (sizes, weights, muscle strengths,
colours) into a Build: the MuJoCo XML of its body, its motors with their mirror partners,
its feet and its standing pose. The settings are described by Setting objects, so the
web page can draw a slider, a switch, a colour picker or a choice for each one without
knowing anything about ducks or puppies.
"""
from dataclasses import dataclass, field


# ---------- Settings the screens can show ----------

# One setting of an animal: a slider (number), a switch (bool), a colour or a choice
@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    kind: str                 # "number", "bool", "colour" or "choice"
    default: object
    help: str = ""
    minimum: float = 0.0      # numbers only
    maximum: float = 1.0      # numbers only
    step: float = 1.0         # numbers only
    unit: str = ""            # numbers only, for example "%" or "kg"
    choices: tuple = ()       # choices only: (value, label) pairs

    # The setting as plain data for the web page
    def to_dict(self) -> dict:
        data = {"key": self.key, "label": self.label, "kind": self.kind, "default": self.default,
                "help": self.help}
        if self.kind == "number":
            data.update(minimum=self.minimum, maximum=self.maximum, step=self.step, unit=self.unit)
        if self.kind == "choice":
            data["choices"] = [{"value": value, "label": label} for value, label in self.choices]
        return data


# A titled group of settings, shown as one card on the creature screen
@dataclass(frozen=True)
class SettingGroup:
    key: str
    title: str
    description: str
    settings: tuple[Setting, ...]

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title, "description": self.description,
                "settings": [setting.to_dict() for setting in self.settings]}


# Short ways to write the common kinds of setting
def percent(key, label, help="", default=100, minimum=25, maximum=300, step=5) -> Setting:
    return Setting(key, label, "number", default, help, minimum, maximum, step, "%")


def switch(key, label, help="", default=True) -> Setting:
    return Setting(key, label, "bool", default, help)


def colour(key, label, default) -> Setting:
    return Setting(key, label, "colour", default)


# Every setting of a list of groups, by key
def all_settings(groups) -> dict[str, Setting]:
    return {setting.key: setting for group in groups for setting in group.settings}


# The default value of every setting
def default_values(groups) -> dict:
    return {key: setting.default for key, setting in all_settings(groups).items()}


# The settings with every missing or broken value replaced by its default, numbers kept in range
def clean_values(groups, values: dict | None) -> dict:
    values = values or {}
    cleaned = {}
    for key, setting in all_settings(groups).items():
        value = values.get(key, setting.default)
        cleaned[key] = _clean_value(setting, value)
    return cleaned


def _clean_value(setting: Setting, value):
    if setting.kind == "number":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return setting.default
        return min(max(number, setting.minimum), setting.maximum)
    if setting.kind == "bool":
        return bool(value)
    if setting.kind == "choice":
        allowed = [choice for choice, _ in setting.choices]
        return value if value in allowed else setting.default
    if setting.kind == "colour":
        text = str(value)
        is_hex = len(text) == 7 and text.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in text[1:])
        return text if is_hex else setting.default
    return value


# ---------- The finished build ----------

# One motor: the joint it turns, its group (for strength settings) and its mirror partner.
# mirror is the name of the motor on the other side (itself for a motor on the middle line),
# mirror_sign is -1 for motors whose angle flips in the mirror (sideways and twisting joints)
@dataclass(frozen=True)
class Motor:
    name: str
    group: str
    mirror: str
    mirror_sign: int = 1
    side: str = "middle"      # "left", "right" or "middle"


# One foot: the shapes that count as touching the ground, and its mirror partner
@dataclass(frozen=True)
class Foot:
    name: str
    geoms: tuple[str, ...]
    mirror: str


# Everything the simulation needs to know about one built animal
@dataclass
class Build:
    animal: str                      # the recipe's key, for example "duck"
    body_xml: str                    # the <body name="torso"> element with everything inside
    actuator_xml: str                # the <actuator> lines, in motor order
    motors: list[Motor]
    feet: list[Foot]
    stand_pose: dict[str, float]     # joint angle of each motor's joint in the starting pose
    stride_pairs: list[tuple[str, str]] = field(default_factory=list)  # (left, right) hips that start apart
    materials: dict[str, str] = field(default_factory=dict)            # name -> "r g b a"
    joint_damping: float = 0.1
    joint_armature: float = 0.005
    size: float = 1.0                # overall size factor, used to scale cameras, rocks and wind
    torso_geoms: list[str] = field(default_factory=list)  # shapes that must not touch the ground
    symmetric_ok: bool = True
    motor_type: str = "position"     # "force" (direct) or "position" (springy)        # False when parts without a twin sit off the middle line

    @property
    def motor_names(self) -> list[str]:
        return [motor.name for motor in self.motors]


# ---------- Writing XML ----------

# Numbers written short but exact enough for the simulation
def num(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def vec(*values: float) -> str:
    return " ".join(num(value) for value in values)


# A hex colour such as "#f5c430" as the "r g b a" text MuJoCo uses
def rgba(hex_colour: str, alpha: float = 1.0) -> str:
    red, green, blue = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return vec(red, green, blue, alpha)


# A shape line; extra holds any further attributes, already written
def geom(name: str | None, kind: str, size, mass: float | None = None, material: str | None = None,
         **extra) -> str:
    parts = ["<geom"]
    if name:
        parts.append(f'name="{name}"')
    parts.append(f'type="{kind}"')
    parts.append(f'size="{vec(*size)}"')
    for key, value in extra.items():
        parts.append(f'{key}="{value}"')
    if mass is not None:
        parts.append(f'mass="{num(mass)}"')
    if material:
        parts.append(f'material="{material}"')
    return " ".join(parts) + "/>"


# A hinge joint around one axis, with its range in radians
def hinge(name: str, axis: tuple[float, float, float], limit: float) -> str:
    return f'<joint name="{name}" type="hinge" axis="{vec(*axis)}" range="{vec(-limit, limit)}" limited="true"/>'


# A position motor (a spring toward the asked angle) or a force motor (the brain sets the force)
def motor_line(name: str, limit: float, stiffness: float, force: float, motor_type: str) -> str:
    if motor_type == "force":
        return (f'<motor name="{name}" joint="{name}" ctrlrange="{vec(-force, force)}" '
                f'forcerange="{vec(-force, force)}" forcelimited="true"/>')
    return (f'<position name="{name}" joint="{name}" kp="{num(stiffness)}" ctrlrange="{vec(-limit, limit)}" '
            f'forcerange="{vec(-force, force)}" forcelimited="true"/>')


# Indent every line of a block of XML by some spaces
def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())
