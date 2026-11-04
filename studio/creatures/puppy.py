"""The puppy: a four legged dog with a nodding head and a wagging tail.

Four legs are steadier than two, so the puppy usually learns to walk sooner than the duck,
but it has more motors to coordinate. Each leg has a sideways joint and a swing joint at
the shoulder or hip, and a knee. The paws are round, like a real dog's.
"""
from studio.creatures.kit import (Build, Foot, Motor, Setting, SettingGroup, clean_values, colour, geom, hinge,
                                  indent, motor_line, num, percent, switch, vec)

KEY = "puppy"
NAME = "Puppy"
DESCRIPTION = "Four legs, a nodding head and a wagging tail. Steadier than the duck, with more legs to coordinate."

SETTING_GROUPS = (
    SettingGroup("size", "Size and shape", "How big the puppy is and how its body is shaped.", (
        percent("overall_size", "Overall size", "Makes the whole puppy bigger or smaller. Weights and muscles grow "
                "with it.", minimum=50, maximum=200),
        percent("body_length", "Body length", "A long body like a dachshund, or a short one like a pug.",
                minimum=60, maximum=200),
        percent("body_width", "Body chunkiness", "How wide and deep the body is.", minimum=60, maximum=170),
        percent("neck_length", "Neck length", minimum=40, maximum=220),
        percent("head_size", "Head size", minimum=50, maximum=200),
        percent("tail_length", "Tail length", "A long tail can help with balance when it wags.", minimum=0,
                maximum=250),
        percent("ear_size", "Ear size", "Only for looks.", minimum=0, maximum=250),
    )),
    SettingGroup("legs", "Legs and paws", "The parts that do the walking.", (
        percent("front_leg_length", "Front leg length", minimum=50, maximum=250),
        percent("back_leg_length", "Back leg length", minimum=50, maximum=250),
        percent("leg_thickness", "Leg thickness", "Thicker legs are heavier and sturdier.", minimum=50, maximum=300),
        percent("leg_spread", "Leg spread", "How far apart the left and right legs are. Wider is steadier.",
                minimum=50, maximum=220),
        percent("paw_size", "Paw size", "Bigger paws give a little more to stand on.", minimum=50, maximum=250),
        percent("foot_grip", "Paw grip", "How well the paws grip the ground. Low grip slides like on ice.",
                minimum=10, maximum=300),
    )),
    SettingGroup("weight", "Weight", "How heavy each part is, on top of what its size gives.", (
        percent("body_weight", "Body weight", minimum=25, maximum=400),
        percent("head_weight", "Head weight", minimum=25, maximum=400),
        percent("leg_weight", "Leg weight", minimum=25, maximum=400),
    )),
    SettingGroup("muscles", "Muscles", "The motors that move each joint.", (
        Setting("motor_type", "Kind of muscle", "choice", "position",
                "Springy muscles pull each joint toward the angle the brain asks for. Direct muscles push with the "
                "force the brain asks for and go loose at zero: more freedom, harder to learn.",
                choices=(("position", "Springy (aims for an angle)"), ("force", "Direct (pushes with a force)"))),
        percent("hip_strength", "Shoulder and hip strength", minimum=25, maximum=1000, step=25),
        percent("knee_strength", "Knee strength", minimum=25, maximum=1000, step=25),
        percent("neck_strength", "Neck strength", minimum=25, maximum=1000, step=25),
        percent("tail_strength", "Tail strength", minimum=25, maximum=1000, step=25),
        percent("stiffness", "Muscle springiness", "How hard springy muscles pull toward their angle. "
                "Higher feels more robotic.", minimum=25, maximum=1000, step=25),
        percent("flexibility", "Joint flexibility", "How far every joint can bend.", minimum=40, maximum=160),
        switch("neck_moves", "Moving neck", "A head that can nod, with one more motor to learn."),
        switch("tail_moves", "Wagging tail", "A tail that swings side to side, with one more motor to learn."),
    )),
    SettingGroup("looks", "Colours", "Only for looks: colours do not change how it walks.", (
        colour("fur_colour", "Fur", "#c98a4b"),
        colour("ear_colour", "Ears", "#7a4a26"),
        colour("paw_colour", "Paws", "#f2e3cf"),
        colour("nose_colour", "Nose", "#2a1d17"),
    )),
)

# ---------- The standard puppy's numbers (metres, kilograms, newton metres, radians) ----------

BODY_HALF_SIZE, BODY_MASS = (0.17, 0.075, 0.065), 1.4
SHOULDER = (0.12, 0.055, -0.03)       # front legs; the back legs mirror x to the hips
UPPER_FRONT = (-0.02, -0.1)           # (forward, down) of the upper front leg
LOWER_FRONT = (0.025, -0.1)
UPPER_BACK = (0.035, -0.1)
LOWER_BACK = (-0.04, -0.1)
UPPER_RADIUS, UPPER_MASS = 0.018, 0.08
LOWER_RADIUS, LOWER_MASS = 0.013, 0.05
PAW_RADIUS, PAW_MASS = 0.022, 0.02

NECK_BASE = (0.15, 0.0, 0.035)
NECK = (0.04, 0.07)
NECK_RADIUS, NECK_MASS = 0.03, 0.06
HEAD_OFFSET = (0.025, 0.03)           # head centre from the end of the neck
HEAD_RADIUS, HEAD_MASS = 0.055, 0.15
SNOUT_HALF_SIZE, SNOUT_OFFSET = (0.035, 0.026, 0.022), (0.06, -0.015)
TAIL_BASE = (-0.165, 0.0, 0.03)
TAIL = (-0.09, 0.06)
TAIL_RADIUS, TAIL_MASS = 0.012, 0.05

RANGES = {"side": 0.35, "swing": 1.0, "knee": 1.2, "neck": 0.6, "tail": 0.7}
STIFFNESS = {"hip": 15.0, "knee": 15.0, "neck": 4.0, "tail": 2.0}
FORCE = {"hip": 3.0, "knee": 3.0, "neck": 1.0, "tail": 0.3}
LEGS = (("front", 1), ("back", -1))   # (name, +1 at the chest end or -1 at the tail end)


# ---------- Building ----------

# The puppy built from its settings
def build(values: dict | None = None) -> Build:
    v = _scaled(clean_values(SETTING_GROUPS, values))
    motors = _motors(v)
    legs = [_leg(v, side, sign, end, end_sign) for side, sign in (("left", 1), ("right", -1)) for end, end_sign in LEGS]
    body = "\n".join([
        '<body name="torso" pos="0 0 0">',
        '  <freejoint name="root"/>',
        indent(_torso_shapes(v), 2),
        indent(_neck(v), 2),
        indent(_tail(v), 2) if v["tail_length"] > 0 else "",
        *[indent(leg, 2) for leg in legs],
        "</body>",
    ])
    feet = [Foot(f"{side}_{end}", (f"{side}_{end}_paw",), f"{other}_{end}")
            for side, other in (("left", "right"), ("right", "left")) for end, _ in LEGS]
    return Build(
        animal=KEY, body_xml=body, actuator_xml="\n".join(_actuator(v, motor) for motor in motors), motors=motors,
        feet=feet, stand_pose={motor.name: 0.0 for motor in motors},
        stride_pairs=[("left_front_swing", "right_front_swing"), ("right_back_swing", "left_back_swing")],
        materials={"pup_fur": _rgba(v["fur_colour"]), "pup_ear": _rgba(v["ear_colour"]),
                   "pup_paw": _rgba(v["paw_colour"]), "pup_nose": _rgba(v["nose_colour"]),
                   "eye_white": "1 1 1 1", "eye_dark": "0.13 0.13 0.13 1"},
        joint_damping=0.1 * v["s"] ** 4, joint_armature=0.005 * v["s"] ** 5, size=v["s"],
        torso_geoms=["body", "head", "snout"],
    )


def _scaled(values: dict) -> dict:
    v = dict(values)
    for group in SETTING_GROUPS:
        for setting in group.settings:
            if setting.unit == "%":
                v[setting.key] = values[setting.key] / 100.0
    v["s"] = v["overall_size"]
    return v


def _rgba(hex_colour: str) -> str:
    red, green, blue = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return vec(red, green, blue, 1.0)


# The body: one long ellipsoid
def _torso_shapes(v: dict) -> str:
    s, length, width = v["s"], v["body_length"], v["body_width"]
    size = (BODY_HALF_SIZE[0] * s * length, BODY_HALF_SIZE[1] * s * width, BODY_HALF_SIZE[2] * s * width)
    mass = BODY_MASS * s ** 3 * length * width ** 2 * v["body_weight"]
    return geom("body", "ellipsoid", size, mass, "pup_fur")


# The neck with the head, snout, nose, ears and eyes
def _neck(v: dict) -> str:
    s, n, h, weight = v["s"], v["neck_length"], v["head_size"], v["head_weight"]
    base = (NECK_BASE[0] * s * v["body_length"], 0.0, NECK_BASE[2] * s * v["body_width"])
    neck = (NECK[0] * s * n, NECK[1] * s * n)
    head = (neck[0] + HEAD_OFFSET[0] * s * h, neck[1] + HEAD_OFFSET[1] * s * h)
    snout = (head[0] + SNOUT_OFFSET[0] * s * h, head[1] + SNOUT_OFFSET[1] * s * h)
    nose = (snout[0] + SNOUT_HALF_SIZE[0] * s * h, snout[1] + 0.01 * s * h)
    head_mass = HEAD_MASS * s ** 3 * h ** 3 * weight
    lines = [
        f'<body name="neck" pos="{vec(*base)}">',
        "  " + hinge("neck", (0, 1, 0), RANGES["neck"] * v["flexibility"]) if v["neck_moves"] else "",
        "  " + geom("neck", "capsule", (NECK_RADIUS * s,), NECK_MASS * s ** 3 * n * weight, "pup_fur",
                    fromto=vec(0, 0, 0, *_xz(neck))),
        "  " + geom("head", "sphere", (HEAD_RADIUS * s * h,), head_mass * 0.8, "pup_fur", pos=vec(*_xz(head))),
        "  " + geom("snout", "box", tuple(half * s * h for half in SNOUT_HALF_SIZE), head_mass * 0.2, "pup_fur",
                    pos=vec(*_xz(snout))),
        "  " + _looks("sphere", (0.012 * s * h,), "pup_nose", (nose[0], 0, nose[1])),
    ]
    ear = v["ear_size"]
    for sign in (1, -1):
        eye = (head[0] + 0.04 * s * h, sign * 0.03 * s * h, head[1] + 0.02 * s * h)
        lines.append("  " + _looks("sphere", (0.011 * s * h,), "eye_white", eye))
        lines.append("  " + _looks("sphere", (0.006 * s * h,), "eye_dark",
                                   (eye[0] + 0.007 * s * h, eye[1] + sign * 0.002 * s * h, eye[2])))
        if ear > 0:
            ear_position = (head[0] - 0.01 * s * h, sign * 0.05 * s * h, head[1] + 0.005 * s * h)
            lines.append("  " + _looks("ellipsoid", (0.018 * s * ear, 0.008 * s * ear, 0.04 * s * ear), "pup_ear",
                                       ear_position, euler=f"{num(sign * 0.35)} 0 0"))
    lines.append("</body>")
    return "\n".join(line for line in lines if line.strip())


# A shape only for looks: no mass and never touching anything
def _looks(kind: str, size, material: str, position, **extra) -> str:
    return geom(None, kind, size, 0, material, pos=vec(*position), contype="0", conaffinity="0", **extra)


# The tail, which can wag side to side
def _tail(v: dict) -> str:
    s, t = v["s"], v["tail_length"]
    base = (TAIL_BASE[0] * s * v["body_length"], 0.0, TAIL_BASE[2] * s * v["body_width"])
    end = (TAIL[0] * s * t, TAIL[1] * s * t)
    return "\n".join(line for line in [
        f'<body name="tail" pos="{vec(*base)}">',
        "  " + hinge("tail", (0, 0, 1), RANGES["tail"] * v["flexibility"]) if v["tail_moves"] else "",
        "  " + geom("tail", "capsule", (TAIL_RADIUS * s,), TAIL_MASS * s ** 3 * t * v["body_weight"], "pup_fur",
                    fromto=vec(0, 0, 0, *_xz(end))),
        "</body>",
    ] if line.strip())


# (forward, up) written as (x, 0, z)
def _xz(point: tuple[float, float]) -> tuple[float, float, float]:
    return point[0], 0.0, point[1]


# One leg: the upper leg on two shoulder or hip joints, the lower leg on a knee and a round paw
def _leg(v: dict, side: str, sign: int, end: str, end_sign: int) -> str:
    s, thick, weight, flex = v["s"], v["leg_thickness"], v["leg_weight"], v["flexibility"]
    length = v[f"{end}_leg_length"]
    upper_shape, lower_shape = (UPPER_FRONT, LOWER_FRONT) if end == "front" else (UPPER_BACK, LOWER_BACK)
    attach = (end_sign * SHOULDER[0] * s * v["body_length"], sign * SHOULDER[1] * s * v["leg_spread"],
              SHOULDER[2] * s * v["body_width"])
    upper = (upper_shape[0] * s * length, upper_shape[1] * s * length)
    lower = (lower_shape[0] * s * length, lower_shape[1] * s * length)
    leg_mass = s ** 3 * length * thick ** 2 * weight
    paw_radius = PAW_RADIUS * s * v["paw_size"]
    paw_mass = PAW_MASS * s ** 3 * v["paw_size"] ** 3 * weight
    name = f"{side}_{end}"
    return "\n".join([
        f'<body name="{name}_upper" pos="{vec(*attach)}">',
        "  " + hinge(f"{name}_side", (1, 0, 0), RANGES["side"] * flex),
        "  " + hinge(f"{name}_swing", (0, 1, 0), RANGES["swing"] * flex),
        "  " + geom(f"{name}_upper", "capsule", (UPPER_RADIUS * s * thick,), UPPER_MASS * leg_mass, "pup_fur",
                    fromto=vec(0, 0, 0, *_xz(upper))),
        f'  <body name="{name}_lower" pos="{vec(*_xz(upper))}">',
        "    " + hinge(f"{name}_knee", (0, 1, 0), RANGES["knee"] * flex),
        "    " + geom(f"{name}_lower", "capsule", (LOWER_RADIUS * s * thick,), LOWER_MASS * leg_mass, "pup_fur",
                      fromto=vec(0, 0, 0, *_xz(lower))),
        "    " + geom(f"{name}_paw", "sphere", (paw_radius,), paw_mass, "pup_paw",
                      pos=vec(lower[0], 0, lower[1]), friction=f'{num(v["foot_grip"])} 0.05 0.01'),
        "  </body>",
        "</body>",
    ])


# The motors in the brain's order: neck, tail, then the left legs and the right legs
def _motors(v: dict) -> list[Motor]:
    motors = []
    if v["neck_moves"]:
        motors.append(Motor("neck", "neck", "neck"))
    if v["tail_moves"] and v["tail_length"] > 0:
        motors.append(Motor("tail", "tail", "tail", -1))  # a wag to the left is a wag to the right in the mirror
    for side, other in (("left", "right"), ("right", "left")):
        for end, _ in LEGS:
            motors.append(Motor(f"{side}_{end}_side", "hip", f"{other}_{end}_side", -1, side))
            motors.append(Motor(f"{side}_{end}_swing", "hip", f"{other}_{end}_swing", 1, side))
            motors.append(Motor(f"{side}_{end}_knee", "knee", f"{other}_{end}_knee", 1, side))
    return motors


# The motor line of one motor; strength grows with the size to the fourth power (weight times lever)
def _actuator(v: dict, motor: Motor) -> str:
    s, flex = v["s"], v["flexibility"]
    kind = motor.name.split("_")[-1] if motor.group != "neck" and motor.group != "tail" else motor.group
    limit = RANGES[kind] * flex
    force = FORCE[motor.group] * v[f"{motor.group}_strength"] * s ** 4
    stiffness = STIFFNESS[motor.group] * v["stiffness"] * s ** 4
    return motor_line(motor.name, limit, stiffness, force, v["motor_type"])
