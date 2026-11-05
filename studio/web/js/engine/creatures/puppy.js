// The puppy: a four legged dog with a nodding head and a wagging tail. A port of studio/creatures/puppy.py.

import { animalGroups } from "../catalog.js";
import { allSettings, cleanValues, foot, geom, hinge, indent, makeBuild, motor, motorLine, num, rgba, vec } from "../kit.js";

export const KEY = "puppy";

// ---------- The standard puppy's numbers (metres, kilograms, newton metres, radians) ----------

const BODY_HALF_SIZE = [0.17, 0.075, 0.065], BODY_MASS = 1.4;
const SHOULDER = [0.12, 0.055, -0.03];   // front legs; the back legs mirror x to the hips
const UPPER_FRONT = [-0.02, -0.1];       // (forward, down) of the upper front leg
const LOWER_FRONT = [0.025, -0.1];
const UPPER_BACK = [0.035, -0.1];
const LOWER_BACK = [-0.04, -0.1];
const UPPER_RADIUS = 0.018, UPPER_MASS = 0.08;
const LOWER_RADIUS = 0.013, LOWER_MASS = 0.05;
const PAW_RADIUS = 0.022, PAW_MASS = 0.02;

const NECK_BASE = [0.15, 0.0, 0.035];
const NECK = [0.04, 0.07];
const NECK_RADIUS = 0.03, NECK_MASS = 0.06;
const HEAD_OFFSET = [0.025, 0.03];       // head centre from the end of the neck
const HEAD_RADIUS = 0.055, HEAD_MASS = 0.15;
const SNOUT_HALF_SIZE = [0.035, 0.026, 0.022], SNOUT_OFFSET = [0.06, -0.015];
const TAIL_BASE = [-0.165, 0.0, 0.03];
const TAIL = [-0.09, 0.06];
const TAIL_RADIUS = 0.012, TAIL_MASS = 0.05;

const RANGES = { side: 0.35, swing: 1.0, knee: 1.2, neck: 0.6, tail: 0.7 };
const STIFFNESS = { hip: 15.0, knee: 15.0, neck: 4.0, tail: 2.0 };
const FORCE_MOTOR_BOOST = 10;
const FORCE = { hip: 3.0, knee: 3.0, neck: 1.0, tail: 0.3 };
const LEGS = [["front", 1], ["back", -1]];  // (name, +1 at the chest end or -1 at the tail end)

// ---------- Building ----------

export function build(values = {}) {
  const groups = animalGroups(KEY);
  const v = scaled(cleanValues(groups, values), groups);
  const motors = makeMotors(v);
  const legs = [];
  for (const [side, sign] of [["left", 1], ["right", -1]]) for (const [end, endSign] of LEGS) legs.push(leg(v, side, sign, end, endSign));
  const body = [
    '<body name="torso" pos="0 0 0">',
    '  <freejoint name="root"/>',
    indent(torsoShapes(v), 2),
    indent(neck(v), 2),
    v.tail_length > 0 ? indent(tail(v), 2) : "",
    ...legs.map((text) => indent(text, 2)),
    "</body>",
  ].join("\n");
  const feet = [];
  for (const [side, other] of [["left", "right"], ["right", "left"]]) {
    for (const [end] of LEGS) feet.push(foot(`${side}_${end}`, [`${side}_${end}_paw`], `${other}_${end}`));
  }
  return makeBuild({
    animal: KEY, bodyXml: body, actuatorXml: motors.map((m) => actuator(v, m)).join("\n"), motors, feet,
    standPose: Object.fromEntries(motors.map((m) => [m.name, 0.0])),
    stridePairs: [["left_front_swing", "right_front_swing"], ["right_back_swing", "left_back_swing"]],
    materials: { pup_fur: rgba(v.fur_colour), pup_ear: rgba(v.ear_colour), pup_paw: rgba(v.paw_colour),
      pup_nose: rgba(v.nose_colour), eye_white: "1 1 1 1", eye_dark: "0.13 0.13 0.13 1" },
    motorType: v.motor_type, jointDamping: 0.1 * v.s ** 4, jointArmature: 0.005 * v.s ** 5, size: v.s,
    torsoGeoms: ["body", "head", "snout"],
  });
}

function scaled(values, groups) {
  const v = { ...values };
  for (const [key, setting] of Object.entries(allSettings(groups))) if (setting.unit === "%") v[key] = values[key] / 100.0;
  v.s = v.overall_size;
  return v;
}

// (forward, up) written as (x, 0, z)
const xz = ([forward, up]) => [forward, 0.0, up];

// The body: one long ellipsoid
function torsoShapes(v) {
  const { s, body_length: length, body_width: width } = v;
  const size = [BODY_HALF_SIZE[0] * s * length, BODY_HALF_SIZE[1] * s * width, BODY_HALF_SIZE[2] * s * width];
  return geom("body", "ellipsoid", size, BODY_MASS * s ** 3 * length * width ** 2 * v.body_weight, "pup_fur");
}

// A shape only for looks: no mass and never touching anything
function looks(kind, size, material, position, extra = {}) {
  return geom(null, kind, size, 0, material, { pos: vec(...position), contype: "0", conaffinity: "0", ...extra });
}

// The neck with the head, snout, nose, ears and eyes
function neck(v) {
  const { s, neck_length: n, head_size: h, head_weight: weight } = v;
  const base = [NECK_BASE[0] * s * v.body_length, 0.0, NECK_BASE[2] * s * v.body_width];
  const neckEnd = [NECK[0] * s * n, NECK[1] * s * n];
  const headCentre = [neckEnd[0] + HEAD_OFFSET[0] * s * h, neckEnd[1] + HEAD_OFFSET[1] * s * h];
  const snout = [headCentre[0] + SNOUT_OFFSET[0] * s * h, headCentre[1] + SNOUT_OFFSET[1] * s * h];
  const nose = [snout[0] + SNOUT_HALF_SIZE[0] * s * h, snout[1] + 0.01 * s * h];
  const headMass = HEAD_MASS * s ** 3 * h ** 3 * weight;
  const lines = [
    `<body name="neck" pos="${vec(...base)}">`,
    v.neck_moves ? "  " + hinge("neck", [0, 1, 0], RANGES.neck * v.flexibility) : "",
    "  " + geom("neck", "capsule", [NECK_RADIUS * s], NECK_MASS * s ** 3 * n * weight, "pup_fur", { fromto: vec(0, 0, 0, ...xz(neckEnd)) }),
    "  " + geom("head", "sphere", [HEAD_RADIUS * s * h], headMass * 0.8, "pup_fur", { pos: vec(...xz(headCentre)) }),
    "  " + geom("snout", "box", SNOUT_HALF_SIZE.map((half) => half * s * h), headMass * 0.2, "pup_fur", { pos: vec(...xz(snout)) }),
    "  " + looks("sphere", [0.012 * s * h], "pup_nose", [nose[0], 0, nose[1]]),
  ];
  const ear = v.ear_size;
  for (const sign of [1, -1]) {
    const eye = [headCentre[0] + 0.04 * s * h, sign * 0.03 * s * h, headCentre[1] + 0.02 * s * h];
    lines.push("  " + looks("sphere", [0.011 * s * h], "eye_white", eye));
    lines.push("  " + looks("sphere", [0.006 * s * h], "eye_dark", [eye[0] + 0.007 * s * h, eye[1] + sign * 0.002 * s * h, eye[2]]));
    if (ear > 0) {
      const earPosition = [headCentre[0] - 0.01 * s * h, sign * 0.05 * s * h, headCentre[1] + 0.005 * s * h];
      lines.push("  " + looks("ellipsoid", [0.018 * s * ear, 0.008 * s * ear, 0.04 * s * ear], "pup_ear", earPosition, { euler: `${num(sign * 0.35)} 0 0` }));
    }
  }
  lines.push("</body>");
  return lines.filter((line) => line.trim()).join("\n");
}

// The tail, which can wag side to side
function tail(v) {
  const { s, tail_length: t } = v;
  const base = [TAIL_BASE[0] * s * v.body_length, 0.0, TAIL_BASE[2] * s * v.body_width];
  const end = [TAIL[0] * s * t, TAIL[1] * s * t];
  return [
    `<body name="tail" pos="${vec(...base)}">`,
    v.tail_moves ? "  " + hinge("tail", [0, 0, 1], RANGES.tail * v.flexibility) : "",
    "  " + geom("tail", "capsule", [TAIL_RADIUS * s], TAIL_MASS * s ** 3 * t * v.body_weight, "pup_fur", { fromto: vec(0, 0, 0, ...xz(end)) }),
    "</body>",
  ].filter((line) => line.trim()).join("\n");
}

// One leg: the upper leg on two shoulder or hip joints, the lower leg on a knee and a round paw
function leg(v, side, sign, end, endSign) {
  const { s, leg_thickness: thick, leg_weight: weight, flexibility: flex } = v;
  const length = v[`${end}_leg_length`];
  const [upperShape, lowerShape] = end === "front" ? [UPPER_FRONT, LOWER_FRONT] : [UPPER_BACK, LOWER_BACK];
  const attach = [endSign * SHOULDER[0] * s * v.body_length, sign * SHOULDER[1] * s * v.leg_spread, SHOULDER[2] * s * v.body_width];
  const upper = [upperShape[0] * s * length, upperShape[1] * s * length];
  const lower = [lowerShape[0] * s * length, lowerShape[1] * s * length];
  const legMass = s ** 3 * length * thick ** 2 * weight;
  const pawRadius = PAW_RADIUS * s * v.paw_size;
  const pawMass = PAW_MASS * s ** 3 * v.paw_size ** 3 * weight;
  const name = `${side}_${end}`;
  return [
    `<body name="${name}_upper" pos="${vec(...attach)}">`,
    "  " + hinge(`${name}_side`, [1, 0, 0], RANGES.side * flex),
    "  " + hinge(`${name}_swing`, [0, 1, 0], RANGES.swing * flex),
    "  " + geom(`${name}_upper`, "capsule", [UPPER_RADIUS * s * thick], UPPER_MASS * legMass, "pup_fur", { fromto: vec(0, 0, 0, ...xz(upper)) }),
    `  <body name="${name}_lower" pos="${vec(...xz(upper))}">`,
    "    " + hinge(`${name}_knee`, [0, 1, 0], RANGES.knee * flex),
    "    " + geom(`${name}_lower`, "capsule", [LOWER_RADIUS * s * thick], LOWER_MASS * legMass, "pup_fur", { fromto: vec(0, 0, 0, ...xz(lower)) }),
    "    " + geom(`${name}_paw`, "sphere", [pawRadius], pawMass, "pup_paw", { pos: vec(lower[0], 0, lower[1]), friction: `${num(v.foot_grip)} 0.05 0.01` }),
    "  </body>",
    "</body>",
  ].join("\n");
}

// The motors in the brain's order: neck, tail, then the left legs and the right legs
function makeMotors(v) {
  const motors = [];
  if (v.neck_moves) motors.push(motor("neck", "neck", "neck"));
  if (v.tail_moves && v.tail_length > 0) motors.push(motor("tail", "tail", "tail", -1));  // a wag flips in the mirror
  for (const [side, other] of [["left", "right"], ["right", "left"]]) {
    for (const [end] of LEGS) {
      motors.push(motor(`${side}_${end}_side`, "hip", `${other}_${end}_side`, -1, side));
      motors.push(motor(`${side}_${end}_swing`, "hip", `${other}_${end}_swing`, 1, side));
      motors.push(motor(`${side}_${end}_knee`, "knee", `${other}_${end}_knee`, 1, side));
    }
  }
  return motors;
}

// The motor line of one motor; strength grows with the size to the fourth power (weight times lever)
function actuator(v, m) {
  const { s, flexibility: flex } = v;
  const kind = m.group !== "neck" && m.group !== "tail" ? m.name.split("_").pop() : m.group;
  const limit = RANGES[kind] * flex;
  let force = FORCE[m.group] * v[`${m.group}_strength`] * s ** 4;
  if (v.motor_type === "force") force *= FORCE_MOTOR_BOOST;
  const stiffness = STIFFNESS[m.group] * v.stiffness * s ** 4;
  return motorLine(m.name, limit, stiffness, force, v.motor_type);
}
