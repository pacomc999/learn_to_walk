"""The duck: a long body, legs set far back, a bendy neck, folded wings and webbed feet.

With every setting at its default this builds exactly the duck of the learn_to_walk
project (duck/assets/duck.xml): the same shapes, masses, joints and 12 motors. Each
setting scales one part, and masses follow the size of the parts (a longer leg weighs more).
"""
from studio.creatures.kit import (Build, Foot, Motor, Setting, SettingGroup, clean_values, colour, geom, hinge,
                                  indent, motor_line, num, percent, switch, vec)

KEY = "duck"
NAME = "Duck"
DESCRIPTION = "Two webbed feet, a long body and a bendy neck. Balancing on two legs makes it a real challenge."

# ---------- The settings ----------

SETTING_GROUPS = (
    SettingGroup("size", "Size and shape", "How big the duck is and how its body is shaped.", (
        percent("overall_size", "Overall size", "Makes the whole duck bigger or smaller. Weights and muscles grow with it.",
                minimum=50, maximum=200),
        percent("body_length", "Body length", "How long the body is from tail to chest.", minimum=60, maximum=170),
        percent("body_width", "Body plumpness", "How wide and tall the body is.", minimum=60, maximum=170),
        percent("neck_length", "Neck length", "A longer neck reaches further but swings more.", minimum=40, maximum=220),
        percent("head_size", "Head size", minimum=50, maximum=200),
        percent("tail_size", "Tail size", "The tail is a counterweight to the neck and head.", minimum=0, maximum=250),
        percent("wing_size", "Wing size", "The folded wings. They do not flap.", minimum=0, maximum=200),
    )),
    SettingGroup("legs", "Legs and feet", "The parts that do the walking.", (
        percent("leg_length", "Leg length", "Longer legs take longer steps but are harder to balance on.",
                minimum=50, maximum=250),
        percent("leg_thickness", "Leg thickness", "Thicker legs are heavier and sturdier.", minimum=50, maximum=300),
        Setting("leg_position", "Leg position", "number", -40,
                "Where the legs join the body: far back (minus) or toward the chest (plus). Real ducks have them far back.",
                -90, 60, 5, "%"),
        percent("leg_spread", "Leg spread", "How far apart the two legs are. Wider is steadier.", minimum=50, maximum=220),
        percent("foot_size", "Foot size", "Bigger webbed feet give a wider base to stand on.", minimum=40, maximum=250),
        percent("foot_grip", "Foot grip", "How well the feet grip the ground. Low grip slides like on ice.",
                minimum=10, maximum=300),
        switch("has_toes", "Moving toes", "Toes that curl give extra control, with two more motors to learn."),
    )),
    SettingGroup("weight", "Weight", "How heavy each part is, on top of what its size gives.", (
        percent("body_weight", "Body weight", minimum=25, maximum=400),
        percent("head_weight", "Head and neck weight", minimum=25, maximum=400),
        percent("leg_weight", "Leg weight", minimum=25, maximum=400),
    )),
    SettingGroup("muscles", "Muscles", "The motors that move each joint.", (
        Setting("motor_type", "Kind of muscle", "choice", "position",
                "Springy muscles pull each joint toward the angle the brain asks for. Direct muscles push with the "
                "force the brain asks for and go loose at zero: more freedom, harder to learn.",
                choices=(("position", "Springy (aims for an angle)"), ("force", "Direct (pushes with a force)"))),
        percent("hip_strength", "Hip strength", minimum=25, maximum=1000, step=25),
        percent("knee_strength", "Knee strength", minimum=25, maximum=1000, step=25),
        percent("ankle_strength", "Ankle strength", minimum=25, maximum=1000, step=25),
        percent("toe_strength", "Toe strength", minimum=25, maximum=1000, step=25),
        percent("neck_strength", "Neck strength", minimum=25, maximum=1000, step=25),
        percent("stiffness", "Muscle springiness", "How hard springy muscles pull toward their angle. "
                "Higher feels more robotic.", minimum=25, maximum=1000, step=25),
        percent("flexibility", "Joint flexibility", "How far every joint can bend.", minimum=40, maximum=160),
        switch("neck_moves", "Moving neck", "A neck that can nod helps balance, with two more motors to learn."),
    )),
    SettingGroup("looks", "Colours", "Only for looks: colours do not change how it walks.", (
        colour("body_colour", "Body", "#f5c430"),
        colour("wing_colour", "Wings", "#e0b021"),
        colour("leg_colour", "Legs and feet", "#ff8c1a"),
        colour("beak_colour", "Beak", "#ff8c1a"),
    )),
)

# ---------- The standard duck's numbers (metres, kilograms, newton metres, radians) ----------

BODY_HALF_SIZE = (0.1875, 0.106, 0.094)
BODY_TILT = 0.12
BODY_MASS = 0.8
TAIL_HALF_SIZE = (0.0585, 0.036, 0.027)
TAIL_POSITION = (-0.1875, 0.0, 0.0375)
TAIL_MASS = 0.2
WING_HALF_SIZE = (0.13, 0.025, 0.05)
WING_POSITION = (-0.025, 0.09, 0.02)
WING_MASS = 0.05

NECK_BASE = (0.1375, 0.0, 0.075)
NECK_LOWER = (0.0325, 0.075)          # (forward, up) of the lower neck bone
NECK_LOWER_RADIUS, NECK_LOWER_MASS = 0.0275, 0.04
NECK_UPPER = (0.0175, 0.055)
NECK_UPPER_RADIUS, NECK_UPPER_MASS = 0.025, 0.04
HEAD_OFFSET = (0.0125, 0.0325)        # head centre from the end of the upper neck
HEAD_RADIUS, HEAD_MASS = 0.06, 0.06
BEAK_OFFSET = (0.075, -0.0125)        # beak centre from the head centre
BEAK_HALF_SIZE, BEAK_MASS = (0.0375, 0.021, 0.009), 0.01

HIP = (-0.075, 0.05, -0.0875)
THIGH = (0.025, -0.0675)              # (forward, down) of the thigh bone
THIGH_RADIUS, THIGH_MASS = 0.015, 0.06
SHIN = (-0.02, -0.075)
SHIN_RADIUS, SHIN_MASS = 0.011, 0.04
HEEL_DROP = 0.016                     # the foot plate sits this far below the ankle
FOOT_HALF_SIZE, FOOT_MASS = (0.0325, 0.025, 0.004), 0.03
TOES_HALF_SIZE, TOES_MASS = (0.0125, 0.035, 0.004), 0.01

# Joint ranges (plus or minus, radians)
RANGES = {"neck_base": 0.6, "neck_top": 0.8, "hip_swing": 1.0, "hip_side": 0.4, "knee": 1.0, "ankle": 0.8,
          "toes": 0.6}
# Springiness (kp) and strongest force of each kind of joint
STIFFNESS = {"neck": 4.0, "hip": 20.0, "knee": 20.0, "ankle": 20.0, "toes": 5.0}
FORCE = {"neck": 1.0, "hip": 4.0, "knee": 4.0, "ankle": 3.0, "toes": 0.5}


# ---------- Building ----------

# The duck built from its settings
def build(values: dict | None = None) -> Build:
    v = _scaled(clean_values(SETTING_GROUPS, values))
    motors = _motors(v)
    body = "\n".join([
        '<body name="torso" pos="0 0 0">',
        '  <freejoint name="root"/>',
        indent(_torso_shapes(v), 2),
        indent(_neck(v), 2),
        indent(_leg(v, "left", 1), 2),
        indent(_leg(v, "right", -1), 2),
        "</body>",
    ])
    actuators = "\n".join(_actuator(v, motor) for motor in motors)
    feet = [Foot("left", _foot_geoms(v, "left"), "right"), Foot("right", _foot_geoms(v, "right"), "left")]
    return Build(
        animal=KEY, body_xml=body, actuator_xml=actuators, motors=motors, feet=feet,
        stand_pose={motor.name: 0.0 for motor in motors},
        stride_pairs=[("left_hip_swing", "right_hip_swing")],
        materials={"duck_body": _rgba(v["body_colour"]), "duck_wing": _rgba(v["wing_colour"]),
                   "duck_leg": _rgba(v["leg_colour"]), "duck_beak": _rgba(v["beak_colour"]),
                   "eye_white": "1 1 1 1", "eye_dark": "0.13 0.13 0.13 1"},
        joint_damping=0.1 * v["s"] ** 4, joint_armature=0.005 * v["s"] ** 5, size=v["s"],
        torso_geoms=["body", "tail", "left_wing", "right_wing"],
    )


# The settings turned into plain factors (1.0 is the standard duck)
def _scaled(values: dict) -> dict:
    v = dict(values)
    for key, setting in [(s.key, s) for group in SETTING_GROUPS for s in group.settings]:
        if setting.unit == "%" and key != "leg_position":
            v[key] = values[key] / 100.0
    v["s"] = v["overall_size"]
    v["leg_position"] = values["leg_position"] / 100.0
    return v


def _rgba(hex_colour: str) -> str:
    red, green, blue = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return vec(red, green, blue, 1.0)


# The body, tail and wings, all fixed to the torso
def _torso_shapes(v: dict) -> str:
    s, length, width = v["s"], v["body_length"], v["body_width"]
    body_size = (BODY_HALF_SIZE[0] * s * length, BODY_HALF_SIZE[1] * s * width, BODY_HALF_SIZE[2] * s * width)
    body_mass = BODY_MASS * s ** 3 * length * width ** 2 * v["body_weight"]
    lines = [geom("body", "ellipsoid", body_size, body_mass, "duck_body", euler=f"0 {num(BODY_TILT)} 0")]
    tail = v["tail_size"]
    if tail > 0:
        tail_size = tuple(half * s * tail for half in TAIL_HALF_SIZE)
        tail_position = (TAIL_POSITION[0] * s * length, 0.0, TAIL_POSITION[2] * s * width)
        tail_mass = TAIL_MASS * s ** 3 * tail ** 3 * v["body_weight"]
        lines.append(geom("tail", "ellipsoid", tail_size, tail_mass, "duck_body", pos=vec(*tail_position)))
    wing = v["wing_size"]
    if wing > 0:
        wing_size = (WING_HALF_SIZE[0] * s * wing * length, WING_HALF_SIZE[1] * s * wing, WING_HALF_SIZE[2] * s * wing)
        wing_mass = WING_MASS * s ** 3 * wing ** 3 * length * v["body_weight"]
        for side, sign in (("left", 1), ("right", -1)):
            position = (WING_POSITION[0] * s * length, sign * WING_POSITION[1] * s * width, WING_POSITION[2] * s * width)
            lines.append(geom(f"{side}_wing", "ellipsoid", wing_size, wing_mass, "duck_wing", pos=vec(*position)))
    return "\n".join(lines)


# The two neck bones, the head with its beak and eyes. Without moving neck joints it is one stiff piece
def _neck(v: dict) -> str:
    s, n, h = v["s"], v["neck_length"], v["head_size"]
    weight = v["head_weight"]
    flex = v["flexibility"]
    base = (NECK_BASE[0] * s * v["body_length"], 0.0, NECK_BASE[2] * s * v["body_width"])
    lower = (NECK_LOWER[0] * s * n, NECK_LOWER[1] * s * n)
    upper = (NECK_UPPER[0] * s * n, NECK_UPPER[1] * s * n)
    head = (upper[0] + HEAD_OFFSET[0] * s * h, upper[1] + HEAD_OFFSET[1] * s * h)
    lower_mass = NECK_LOWER_MASS * s ** 3 * n * weight
    upper_mass = NECK_UPPER_MASS * s ** 3 * n * weight
    joint_base = hinge("neck_base", (0, 1, 0), RANGES["neck_base"] * flex) if v["neck_moves"] else ""
    joint_top = hinge("neck_top", (0, 1, 0), RANGES["neck_top"] * flex) if v["neck_moves"] else ""
    head_lines = _head(v, head)
    return "\n".join(line for line in [
        f'<body name="neck_lower" pos="{vec(*base)}">',
        f"  {joint_base}" if joint_base else "",
        "  " + geom("neck_lower", "capsule", (NECK_LOWER_RADIUS * s,), lower_mass, "duck_body",
                    fromto=vec(0, 0, 0, lower[0], 0, lower[1])),
        f'  <body name="neck_upper" pos="{vec(lower[0], 0, lower[1])}">',
        f"    {joint_top}" if joint_top else "",
        "    " + geom("neck_upper", "capsule", (NECK_UPPER_RADIUS * s,), upper_mass, "duck_body",
                      fromto=vec(0, 0, 0, upper[0], 0, upper[1])),
        indent(head_lines, 4),
        "  </body>",
        "</body>",
    ] if line)


# The head ball, the beak and the two eyes (the eyes have no mass and never touch anything)
def _head(v: dict, centre: tuple[float, float]) -> str:
    s, h, weight = v["s"], v["head_size"], v["head_weight"]
    x, z = centre
    beak = (x + BEAK_OFFSET[0] * s * h, z + BEAK_OFFSET[1] * s * h)
    lines = [
        geom("head", "sphere", (HEAD_RADIUS * s * h,), HEAD_MASS * s ** 3 * h ** 3 * weight, "duck_body",
             pos=vec(x, 0, z)),
        geom("beak", "box", tuple(half * s * h for half in BEAK_HALF_SIZE), BEAK_MASS * s ** 3 * h ** 3 * weight,
             "duck_beak", pos=vec(beak[0], 0, beak[1])),
    ]
    for sign in (1, -1):
        white = (x + 0.03 * s * h, sign * 0.045 * s * h, z + 0.015 * s * h)
        pupil = (x + 0.0372 * s * h, sign * 0.0528 * s * h, z + 0.0168 * s * h)
        lines.append(geom(None, "sphere", (0.0132 * s * h,), 0, "eye_white", pos=vec(*white), contype="0",
                          conaffinity="0"))
        lines.append(geom(None, "sphere", (0.0066 * s * h,), 0, "eye_dark", pos=vec(*pupil), contype="0",
                          conaffinity="0"))
    return "\n".join(lines)


# One leg: thigh, shin, webbed foot and toes. sign is +1 for the left leg and -1 for the right
def _leg(v: dict, side: str, sign: int) -> str:
    s, length, thick, foot = v["s"], v["leg_length"], v["leg_thickness"], v["foot_size"]
    weight, flex = v["leg_weight"], v["flexibility"]
    half_body = BODY_HALF_SIZE[0] * s * v["body_length"]
    hip = (v["leg_position"] * half_body, sign * HIP[1] * s * v["leg_spread"], HIP[2] * s * v["body_width"])
    thigh = (THIGH[0] * s * length, THIGH[1] * s * length)
    shin = (SHIN[0] * s * length, SHIN[1] * s * length)
    heel = HEEL_DROP * s * length
    foot_size = (FOOT_HALF_SIZE[0] * s * foot, FOOT_HALF_SIZE[1] * s * foot, FOOT_HALF_SIZE[2] * s)
    toes_size = (TOES_HALF_SIZE[0] * s * foot, TOES_HALF_SIZE[1] * s * foot, TOES_HALF_SIZE[2] * s)
    leg_mass = s ** 3 * length * thick ** 2 * weight
    foot_mass = s ** 3 * foot ** 2 * weight
    grip = f'{num(v["foot_grip"])} 0.05 0.01'
    toes = ""
    if v["has_toes"]:
        toes = "\n".join([
            f'<body name="{side}_toes" pos="{vec(2 * foot_size[0], 0, -heel)}">',
            "  " + hinge(f"{side}_toes", (0, 1, 0), RANGES["toes"] * flex),
            "  " + geom(f"{side}_toes", "box", toes_size, TOES_MASS * foot_mass, "duck_leg",
                        pos=vec(toes_size[0], 0, 0), friction=grip),
            "</body>",
        ])
    return "\n".join(line for line in [
        f'<body name="{side}_thigh" pos="{vec(*hip)}">',
        "  " + hinge(f"{side}_hip_swing", (0, 1, 0), RANGES["hip_swing"] * flex),
        "  " + hinge(f"{side}_hip_side", (1, 0, 0), RANGES["hip_side"] * flex),
        "  " + geom(f"{side}_thigh", "capsule", (THIGH_RADIUS * s * thick,), THIGH_MASS * leg_mass, "duck_leg",
                    fromto=vec(0, 0, 0, thigh[0], 0, thigh[1])),
        f'  <body name="{side}_shin" pos="{vec(thigh[0], 0, thigh[1])}">',
        "    " + hinge(f"{side}_knee", (0, 1, 0), RANGES["knee"] * flex),
        "    " + geom(f"{side}_shin", "capsule", (SHIN_RADIUS * s * thick,), SHIN_MASS * leg_mass, "duck_leg",
                      fromto=vec(0, 0, 0, shin[0], 0, shin[1])),
        f'    <body name="{side}_foot" pos="{vec(shin[0], 0, shin[1])}">',
        "      " + hinge(f"{side}_ankle", (0, 1, 0), RANGES["ankle"] * flex),
        "      " + geom(f"{side}_foot", "box", foot_size, FOOT_MASS * foot_mass, "duck_leg",
                        pos=vec(foot_size[0], 0, -heel), friction=grip),
        indent(toes, 6) if toes else "",
        "    </body>",
        "  </body>",
        "</body>",
    ] if line)


# The shapes of one foot that count as touching the ground
def _foot_geoms(v: dict, side: str) -> tuple[str, ...]:
    return (f"{side}_foot", f"{side}_toes") if v["has_toes"] else (f"{side}_foot",)


# The motors in the brain's order: neck, left leg, right leg (the learn_to_walk order)
def _motors(v: dict) -> list[Motor]:
    motors = []
    if v["neck_moves"]:
        motors += [Motor("neck_base", "neck", "neck_base"), Motor("neck_top", "neck", "neck_top")]
    leg = [("hip_swing", "hip", 1), ("hip_side", "hip", -1), ("knee", "knee", 1), ("ankle", "ankle", 1)]
    if v["has_toes"]:
        leg.append(("toes", "toes", 1))
    for side, other in (("left", "right"), ("right", "left")):
        for joint, group, sign in leg:
            motors.append(Motor(f"{side}_{joint}", group, f"{other}_{joint}", sign, side))
    return motors


# The motor line of one motor: its range from the joint, its strength from the settings.
# Strength and springiness grow with the size to the fourth power (weight times lever)
def _actuator(v: dict, motor: Motor) -> str:
    s, flex = v["s"], v["flexibility"]
    joint_kind = motor.name.removeprefix("left_").removeprefix("right_")
    limit = RANGES[joint_kind] * flex
    strength = v[f"{_setting_group(motor.group)}_strength"]
    force = FORCE[motor.group] * strength * s ** 4
    stiffness = STIFFNESS[motor.group] * v["stiffness"] * s ** 4
    return motor_line(motor.name, limit, stiffness, force, v["motor_type"])


def _setting_group(group: str) -> str:
    return {"toes": "toe"}.get(group, group)

