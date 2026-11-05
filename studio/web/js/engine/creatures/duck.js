// The duck: a long body, legs set far back, a bendy neck, folded wings and webbed feet.
// A port of studio/creatures/duck.py: with every setting at its default it builds learn_to_walk's duck.

import { animalGroups } from "../catalog.js";
import { allSettings, cleanValues, foot, geom, hinge, indent, makeBuild, motor, motorLine, num, rgba, vec } from "../kit.js";

export const KEY = "duck";

// ---------- The standard duck's numbers (metres, kilograms, newton metres, radians) ----------

const BODY_HALF_SIZE = [0.1875, 0.106, 0.094];
const BODY_TILT = 0.12;
const BODY_MASS = 0.8;
const TAIL_HALF_SIZE = [0.0585, 0.036, 0.027];
const TAIL_POSITION = [-0.1875, 0.0, 0.0375];
const TAIL_MASS = 0.2;
const WING_HALF_SIZE = [0.13, 0.025, 0.05];
const WING_POSITION = [-0.025, 0.09, 0.02];
const WING_MASS = 0.05;

const NECK_BASE = [0.1375, 0.0, 0.075];
const NECK_LOWER = [0.0325, 0.075];      // (forward, up) of the lower neck bone
const NECK_LOWER_RADIUS = 0.0275, NECK_LOWER_MASS = 0.04;
const NECK_UPPER = [0.0175, 0.055];
const NECK_UPPER_RADIUS = 0.025, NECK_UPPER_MASS = 0.04;
const HEAD_OFFSET = [0.0125, 0.0325];    // head centre from the end of the upper neck
const HEAD_RADIUS = 0.06, HEAD_MASS = 0.06;
const BEAK_OFFSET = [0.075, -0.0125];    // beak centre from the head centre
const BEAK_HALF_SIZE = [0.0375, 0.021, 0.009], BEAK_MASS = 0.01;

const HIP = [-0.075, 0.05, -0.0875];
const THIGH = [0.025, -0.0675];          // (forward, down) of the thigh bone
const THIGH_RADIUS = 0.015, THIGH_MASS = 0.06;
const SHIN = [-0.02, -0.075];
const SHIN_RADIUS = 0.011, SHIN_MASS = 0.04;
const HEEL_DROP = 0.016;                 // the foot plate sits this far below the ankle
const FOOT_HALF_SIZE = [0.0325, 0.025, 0.004], FOOT_MASS = 0.03;
const TOES_HALF_SIZE = [0.0125, 0.035, 0.004], TOES_MASS = 0.01;

// Joint ranges (plus or minus, radians), springiness (kp) and strongest force of each kind of joint
const RANGES = { neck_base: 0.6, neck_top: 0.8, hip_swing: 1.0, hip_side: 0.4, knee: 1.0, ankle: 0.8, toes: 0.6 };
const STIFFNESS = { neck: 4.0, hip: 20.0, knee: 20.0, ankle: 20.0, toes: 5.0 };
const FORCE_MOTOR_BOOST = 10;  // direct (force) muscles push up to this many times the springy force limits
const FORCE = { neck: 1.0, hip: 4.0, knee: 4.0, ankle: 3.0, toes: 0.5 };

// ---------- Building ----------

export function build(values = {}) {
  const groups = animalGroups(KEY);
  const v = scaled(cleanValues(groups, values), groups);
  const motors = makeMotors(v);
  const body = [
    '<body name="torso" pos="0 0 0">',
    '  <freejoint name="root"/>',
    indent(torsoShapes(v), 2),
    indent(neck(v), 2),
    indent(leg(v, "left", 1), 2),
    indent(leg(v, "right", -1), 2),
    "</body>",
  ].join("\n");
  return makeBuild({
    animal: KEY, bodyXml: body, actuatorXml: motors.map((m) => actuator(v, m)).join("\n"), motors,
    feet: [foot("left", footGeoms(v, "left"), "right"), foot("right", footGeoms(v, "right"), "left")],
    standPose: Object.fromEntries(motors.map((m) => [m.name, 0.0])),
    stridePairs: [["left_hip_swing", "right_hip_swing"]],
    materials: { duck_body: rgba(v.body_colour), duck_wing: rgba(v.wing_colour), duck_leg: rgba(v.leg_colour),
      duck_beak: rgba(v.beak_colour), eye_white: "1 1 1 1", eye_dark: "0.13 0.13 0.13 1" },
    motorType: v.motor_type, jointDamping: 0.1 * v.s ** 4, jointArmature: 0.005 * v.s ** 5, size: v.s,
    torsoGeoms: ["body", "tail", "left_wing", "right_wing"],
  });
}

// The settings turned into plain factors (1.0 is the standard duck)
function scaled(values, groups) {
  const v = { ...values };
  for (const [key, setting] of Object.entries(allSettings(groups))) {
    if (setting.unit === "%" && key !== "leg_position") v[key] = values[key] / 100.0;
  }
  v.s = v.overall_size;
  v.leg_position = values.leg_position / 100.0;
  return v;
}

// The body, tail and wings, all fixed to the torso
function torsoShapes(v) {
  const { s, body_length: length, body_width: width } = v;
  const bodySize = [BODY_HALF_SIZE[0] * s * length, BODY_HALF_SIZE[1] * s * width, BODY_HALF_SIZE[2] * s * width];
  const bodyMass = BODY_MASS * s ** 3 * length * width ** 2 * v.body_weight;
  const lines = [geom("body", "ellipsoid", bodySize, bodyMass, "duck_body", { euler: `0 ${num(BODY_TILT)} 0` })];
  const tail = v.tail_size;
  if (tail > 0) {
    const tailSize = TAIL_HALF_SIZE.map((half) => half * s * tail);
    const tailPosition = [TAIL_POSITION[0] * s * length, 0.0, TAIL_POSITION[2] * s * width];
    const tailMass = TAIL_MASS * s ** 3 * tail ** 3 * v.body_weight;
    lines.push(geom("tail", "ellipsoid", tailSize, tailMass, "duck_body", { pos: vec(...tailPosition) }));
  }
  const wing = v.wing_size;
  if (wing > 0) {
    const wingSize = [WING_HALF_SIZE[0] * s * wing * length, WING_HALF_SIZE[1] * s * wing, WING_HALF_SIZE[2] * s * wing];
    const wingMass = WING_MASS * s ** 3 * wing ** 3 * length * v.body_weight;
    for (const [side, sign] of [["left", 1], ["right", -1]]) {
      const position = [WING_POSITION[0] * s * length, sign * WING_POSITION[1] * s * width, WING_POSITION[2] * s * width];
      lines.push(geom(`${side}_wing`, "ellipsoid", wingSize, wingMass, "duck_wing", { pos: vec(...position) }));
    }
  }
  return lines.join("\n");
}

// The two neck bones, the head with its beak and eyes. Without moving neck joints it is one stiff piece
function neck(v) {
  const { s, neck_length: n, head_size: h, head_weight: weight, flexibility: flex } = v;
  const base = [NECK_BASE[0] * s * v.body_length, 0.0, NECK_BASE[2] * s * v.body_width];
  const lower = [NECK_LOWER[0] * s * n, NECK_LOWER[1] * s * n];
  const upper = [NECK_UPPER[0] * s * n, NECK_UPPER[1] * s * n];
  const headCentre = [upper[0] + HEAD_OFFSET[0] * s * h, upper[1] + HEAD_OFFSET[1] * s * h];
  const lowerMass = NECK_LOWER_MASS * s ** 3 * n * weight;
  const upperMass = NECK_UPPER_MASS * s ** 3 * n * weight;
  const jointBase = v.neck_moves ? hinge("neck_base", [0, 1, 0], RANGES.neck_base * flex) : "";
  const jointTop = v.neck_moves ? hinge("neck_top", [0, 1, 0], RANGES.neck_top * flex) : "";
  return [
    `<body name="neck_lower" pos="${vec(...base)}">`,
    jointBase ? `  ${jointBase}` : "",
    "  " + geom("neck_lower", "capsule", [NECK_LOWER_RADIUS * s], lowerMass, "duck_body", { fromto: vec(0, 0, 0, lower[0], 0, lower[1]) }),
    `  <body name="neck_upper" pos="${vec(lower[0], 0, lower[1])}">`,
    jointTop ? `    ${jointTop}` : "",
    "    " + geom("neck_upper", "capsule", [NECK_UPPER_RADIUS * s], upperMass, "duck_body", { fromto: vec(0, 0, 0, upper[0], 0, upper[1]) }),
    indent(head(v, headCentre), 4),
    "  </body>",
    "</body>",
  ].filter((line) => line).join("\n");
}

// The head ball, the beak and the two eyes (the eyes have no mass and never touch anything)
function head(v, [x, z]) {
  const { s, head_size: h, head_weight: weight } = v;
  const beak = [x + BEAK_OFFSET[0] * s * h, z + BEAK_OFFSET[1] * s * h];
  const lines = [
    geom("head", "sphere", [HEAD_RADIUS * s * h], HEAD_MASS * s ** 3 * h ** 3 * weight, "duck_body", { pos: vec(x, 0, z) }),
    geom("beak", "box", BEAK_HALF_SIZE.map((half) => half * s * h), BEAK_MASS * s ** 3 * h ** 3 * weight, "duck_beak", { pos: vec(beak[0], 0, beak[1]) }),
  ];
  for (const sign of [1, -1]) {
    const white = [x + 0.03 * s * h, sign * 0.045 * s * h, z + 0.015 * s * h];
    const pupil = [x + 0.0372 * s * h, sign * 0.0528 * s * h, z + 0.0168 * s * h];
    lines.push(geom(null, "sphere", [0.0132 * s * h], 0, "eye_white", { pos: vec(...white), contype: "0", conaffinity: "0" }));
    lines.push(geom(null, "sphere", [0.0066 * s * h], 0, "eye_dark", { pos: vec(...pupil), contype: "0", conaffinity: "0" }));
  }
  return lines.join("\n");
}

// One leg: thigh, shin, webbed foot and toes. sign is +1 for the left leg and -1 for the right
function leg(v, side, sign) {
  const { s, leg_length: length, leg_thickness: thick, foot_size: footSize, leg_weight: weight, flexibility: flex } = v;
  const halfBody = BODY_HALF_SIZE[0] * s * v.body_length;
  const hip = [v.leg_position * halfBody, sign * HIP[1] * s * v.leg_spread, HIP[2] * s * v.body_width];
  const thigh = [THIGH[0] * s * length, THIGH[1] * s * length];
  const shin = [SHIN[0] * s * length, SHIN[1] * s * length];
  const heel = HEEL_DROP * s * length;
  const footHalf = [FOOT_HALF_SIZE[0] * s * footSize, FOOT_HALF_SIZE[1] * s * footSize, FOOT_HALF_SIZE[2] * s];
  const toesHalf = [TOES_HALF_SIZE[0] * s * footSize, TOES_HALF_SIZE[1] * s * footSize, TOES_HALF_SIZE[2] * s];
  const legMass = s ** 3 * length * thick ** 2 * weight;
  const footMass = s ** 3 * footSize ** 2 * weight;
  const grip = `${num(v.foot_grip)} 0.05 0.01`;
  let toes = "";
  if (v.has_toes) {
    toes = [
      `<body name="${side}_toes" pos="${vec(2 * footHalf[0], 0, -heel)}">`,
      "  " + hinge(`${side}_toes`, [0, 1, 0], RANGES.toes * flex),
      "  " + geom(`${side}_toes`, "box", toesHalf, TOES_MASS * footMass, "duck_leg", { pos: vec(toesHalf[0], 0, 0), friction: grip }),
      "</body>",
    ].join("\n");
  }
  return [
    `<body name="${side}_thigh" pos="${vec(...hip)}">`,
    "  " + hinge(`${side}_hip_swing`, [0, 1, 0], RANGES.hip_swing * flex),
    "  " + hinge(`${side}_hip_side`, [1, 0, 0], RANGES.hip_side * flex),
    "  " + geom(`${side}_thigh`, "capsule", [THIGH_RADIUS * s * thick], THIGH_MASS * legMass, "duck_leg", { fromto: vec(0, 0, 0, thigh[0], 0, thigh[1]) }),
    `  <body name="${side}_shin" pos="${vec(thigh[0], 0, thigh[1])}">`,
    "    " + hinge(`${side}_knee`, [0, 1, 0], RANGES.knee * flex),
    "    " + geom(`${side}_shin`, "capsule", [SHIN_RADIUS * s * thick], SHIN_MASS * legMass, "duck_leg", { fromto: vec(0, 0, 0, shin[0], 0, shin[1]) }),
    `    <body name="${side}_foot" pos="${vec(shin[0], 0, shin[1])}">`,
    "      " + hinge(`${side}_ankle`, [0, 1, 0], RANGES.ankle * flex),
    "      " + geom(`${side}_foot`, "box", footHalf, FOOT_MASS * footMass, "duck_leg", { pos: vec(footHalf[0], 0, -heel), friction: grip }),
    toes ? indent(toes, 6) : "",
    "    </body>",
    "  </body>",
    "</body>",
  ].filter((line) => line).join("\n");
}

// The shapes of one foot that count as touching the ground
function footGeoms(v, side) {
  return v.has_toes ? [`${side}_foot`, `${side}_toes`] : [`${side}_foot`];
}

// The motors in the brain's order: neck, left leg, right leg (the learn_to_walk order)
function makeMotors(v) {
  const motors = [];
  if (v.neck_moves) motors.push(motor("neck_base", "neck", "neck_base"), motor("neck_top", "neck", "neck_top"));
  const legJoints = [["hip_swing", "hip", 1], ["hip_side", "hip", -1], ["knee", "knee", 1], ["ankle", "ankle", 1]];
  if (v.has_toes) legJoints.push(["toes", "toes", 1]);
  for (const [side, other] of [["left", "right"], ["right", "left"]]) {
    for (const [joint, group, sign] of legJoints) motors.push(motor(`${side}_${joint}`, group, `${other}_${joint}`, sign, side));
  }
  return motors;
}

// The motor line of one motor. Strength and springiness grow with the size to the fourth power.
// Direct muscles get FORCE_MOTOR_BOOST times the force, as learn_to_walk's force_motors body
function actuator(v, m) {
  const { s, flexibility: flex } = v;
  const jointKind = m.name.replace(/^left_/, "").replace(/^right_/, "");
  const limit = RANGES[jointKind] * flex;
  const strength = v[`${m.group === "toes" ? "toe" : m.group}_strength`];
  let force = FORCE[m.group] * strength * s ** 4;
  if (v.motor_type === "force") force *= FORCE_MOTOR_BOOST;
  const stiffness = STIFFNESS[m.group] * v.stiffness * s ** 4;
  return motorLine(m.name, limit, stiffness, force, v.motor_type);
}
