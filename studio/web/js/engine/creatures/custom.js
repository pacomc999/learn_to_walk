// Build your own: a creature made of parts (sticks, balls and plates) joined by motors.
// A port of studio/creatures/custom.py. The ready-made bodies (templates) come from catalog.json.

import { animalGroups, getCatalog } from "../catalog.js";
import { cleanValues, foot, geom, indent, makeBuild, motor, motorLine, num, vec } from "../kit.js";

export const KEY = "custom";

const SHAPES = ["stick", "ball", "plate"];
const JOINTS = ["fixed", "swing", "side", "twist", "hip"];
const TORSO_SHAPES = ["box", "ellipsoid", "ball", "capsule"];
const MOST_PARTS = 24;
const DENSITY = 1000.0;  // kilograms per cubic metre: parts weigh about as much as water

// Joint axes in the part's own frame, and the sign its angle gets in the mirror
const AXES = { swing: [[0, 1, 0], 1], side: [[1, 0, 0], -1], twist: [[0, 0, 1], -1] };

// ---------- Ready-made bodies ----------

function templates() {
  return getCatalog().animals.find((a) => a.key === KEY).templates;
}

export function defaultBody() {
  const table = templates().find((t) => t.key === "table");
  return { template: "table", torso: { ...table.torso }, parts: table.parts.map((part) => ({ ...part })) };
}

// ---------- Cleaning a design ----------

// A number in range, or the fallback for anything that is not a finite number (as Python's float())
function number(value, low, high, fallback) {
  if (value === null || value === undefined || typeof value === "object" || value === "") return fallback;
  const n = typeof value === "boolean" ? Number(value) : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, low), high);
}

function colour(value, fallback) {
  const text = String(value || "");
  return /^#[0-9a-fA-F]{6}$/.test(text) ? text : fallback;
}

export function cleanBody(body) {
  body = body && typeof body === "object" && !Array.isArray(body) ? body : defaultBody();
  const torsoIn = body.torso && typeof body.torso === "object" ? body.torso : {};
  const torso = {
    shape: TORSO_SHAPES.includes(torsoIn.shape) ? torsoIn.shape : "box",
    length: number(torsoIn.length, 0.04, 1.2, 0.3),
    width: number(torsoIn.width, 0.04, 1.0, 0.16),
    height: number(torsoIn.height, 0.02, 1.0, 0.07),
    weight: number(torsoIn.weight, 0.05, 20.0, 1.0),
    colour: colour(torsoIn.colour, "#f59e0b"),
  };
  const parts = [];
  const known = new Set(["torso"]);
  for (const raw of (Array.isArray(body.parts) ? body.parts : []).slice(0, MOST_PARTS)) {
    if (!raw || typeof raw !== "object") continue;
    const pid = String(raw.id ?? "").toLowerCase().replace(/[^a-z0-9_]/g, "").slice(0, 24);
    if (!pid || known.has(pid) || pid === "torso" || pid.startsWith("left_") || pid.startsWith("right_")) continue;
    parts.push({
      id: pid, name: String(raw.name || pid).slice(0, 40), parent: known.has(raw.parent) ? raw.parent : "torso",
      shape: SHAPES.includes(raw.shape) ? raw.shape : "stick",
      length: number(raw.length, 0.0, 1.0, 0.12), thickness: number(raw.thickness, 0.004, 0.2, 0.015),
      width: number(raw.width, 0.01, 0.5, 0.05),
      x: number(raw.x, -1, 1, 0), y: number(raw.y, -1, 1, 0), z: number(raw.z, -1, 1, 0), along: number(raw.along, 0, 1, 1),
      pitch: number(raw.pitch, -90, 90, 90), yaw: number(raw.yaw, -180, 180, 0),
      joint: JOINTS.includes(raw.joint) ? raw.joint : "swing",
      motion: number(raw.motion, 5, 170, 60), strength: number(raw.strength, 0.1, 100, 3),
      mirror: Boolean(raw.mirror), foot: Boolean(raw.foot),
      heaviness: number(raw.heaviness, 10, 1000, 100), colour: colour(raw.colour, "#4f46e5"),
    });
    known.add(pid);
  }
  // A crawler (a worm, a snake) lies on the ground: falling and staying up mean nothing for it
  return { template: String(body.template ?? ""), torso, parts, crawler: Boolean(body.crawler) };
}

// ---------- Building ----------

export function build(values = {}) {
  values = values || {};
  const settings = cleanValues(animalGroups(KEY), values);
  const body = cleanBody(values.body);
  return new Maker(body, settings).build();
}

const radians = (degrees) => (degrees * Math.PI) / 180;

function direction(pitch, yaw) {
  const p = radians(pitch), y = radians(yaw);
  return [Math.cos(p) * Math.cos(y), Math.cos(p) * Math.sin(y), -Math.sin(p)];
}

// The half sizes of a part, used to place what hangs on it
function halfSize(part) {
  if (part.shape === "plate") return [part.length / 2, part.width / 2, part.thickness / 2];
  return [part.thickness, part.thickness, part.thickness];
}

// Turns a cleaned design into MuJoCo XML, motors and feet
class Maker {
  constructor(body, settings) {
    this.body = body;
    this.settings = settings;
    this.parts = body.parts;
    this.children = {};
    for (const part of this.parts) (this.children[part.parent] ||= []).push(part);
    this.motors = [];
    this.actuators = [];
    this.feet = [];
    this.materials = { torso: rgbaOf(body.torso.colour) };
    this.symmetricOk = true;
  }

  build() {
    const torso = this.body.torso;
    const lines = ['<body name="torso" pos="0 0 0">', '  <freejoint name="root"/>', "  " + this.torsoGeom(torso)];
    for (const part of this.children.torso || []) {
      lines.push(indent(this.partTree(part, "torso", "middle"), 2));
      if (part.mirror) lines.push(indent(this.partTree(part, "torso", "right"), 2));
    }
    lines.push("</body>");
    this.orderMotors();
    const size = Math.max(0.5, Math.max(torso.length, torso.width) / 0.3);
    const pairs = this.motors.filter((m) => m.side === "left" && m.name.endsWith("_swing") && this.topLevel(m.name))
      .map((m) => [m.name, m.mirror]);
    return makeBuild({
      animal: KEY, bodyXml: lines.join("\n"), actuatorXml: this.actuators.join("\n"), motors: this.motors, feet: this.feet,
      standPose: Object.fromEntries(this.motors.map((m) => [m.name, 0.0])), stridePairs: pairs.slice(0, 2),
      materials: this.materials, size, torsoGeoms: ["torso"], symmetricOk: this.symmetricOk,
      motorType: this.settings.motor_type, crawler: this.body.crawler,
    });
  }

  topLevel(motorName) {
    const withoutSide = motorName.startsWith("left_") ? motorName.slice(5) : motorName;
    const partId = withoutSide.slice(0, withoutSide.lastIndexOf("_"));
    return this.parts.some((p) => p.id === partId && p.parent === "torso");
  }

  torsoGeom(torso) {
    const half = [torso.length / 2, torso.width / 2, torso.height / 2];
    const mass = torso.weight;
    if (torso.shape === "ball") return geom("torso", "sphere", [half[0]], mass, "torso");
    if (torso.shape === "ellipsoid") return geom("torso", "ellipsoid", half, mass, "torso");
    if (torso.shape === "capsule") {
      const radius = Math.min(half[1], half[2]);
      const reach = Math.max(half[0] - radius, 0.001);
      return geom("torso", "capsule", [radius], mass, "torso", { fromto: vec(-reach, 0, 0, reach, 0, 0) });
    }
    return geom("torso", "box", half, mass, "torso");
  }

  // Where a child joins its parent, in the parent's frame
  attachPoint(part, parent) {
    if (!parent) {
      const torso = this.body.torso;
      let half = [torso.length / 2, torso.width / 2, torso.height / 2];
      if (torso.shape === "ball") half = [torso.length / 2, torso.length / 2, torso.length / 2];
      return [part.x * half[0], part.y * half[1], part.z * half[2]];
    }
    if (parent.shape === "stick") {
      return direction(parent.pitch, parent.yaw).map((d) => part.along * parent.length * d);
    }
    const half = halfSize(parent);
    const centre = this.shapeCentre(parent);
    return [centre[0] + part.x * half[0], centre[1] + part.y * half[1], centre[2] + part.z * half[2]];
  }

  // The centre of a part's shape, from its joint, in its own (unmirrored) frame
  shapeCentre(part) {
    const d = direction(part.pitch, part.yaw);
    const reach = part.shape === "ball" ? part.length : part.length / 2;
    return d.map((value) => reach * value);
  }

  // A part and everything that hangs on it. side: "middle" (or the left copy of a mirrored part) or "right"
  partTree(part, parentId, side, mirrored = false) {
    mirrored = mirrored || part.mirror;
    if (side === "middle" && mirrored) side = "left";
    const flip = side === "right" ? -1.0 : 1.0;
    const name = side === "middle" ? part.id : `${side}_${part.id}`;
    const parent = parentId === "torso" ? null : this.parts.find((p) => p.id === parentId);
    const point = this.attachPoint(part, parent);
    const position = [point[0], flip * point[1], point[2]];
    if (side === "middle" && !mirrored && Math.abs(position[1]) > 1e-9) this.symmetricOk = false;  // off the middle line
    const lines = [`<body name="${name}" pos="${vec(...position)}">`];
    for (const kind of jointKinds(part)) {
      const [axis, mirrorSign] = AXES[kind];
      const joint = `${name}_${kind}`;
      const limit = radians(part.motion / 2);
      lines.push(`  <joint name="${joint}" type="hinge" axis="${vec(...axis)}" range="${vec(-limit, limit)}" limited="true"/>`);
      this.addMotor(joint, part, kind, side, mirrorSign, limit);
    }
    lines.push("  " + this.partGeom(part, name, flip));
    if (part.foot) {
      const other = side === "middle" ? name : `${side === "left" ? "right" : "left"}_${part.id}`;
      this.feet.push(foot(name, [name], other));
    }
    for (const child of this.children[part.id] || []) {
      lines.push(indent(this.partTree(child, part.id, side, mirrored), 2));
      if (child.mirror && !mirrored) lines.push(indent(this.partTree(child, part.id, "right", true), 2));
    }
    lines.push("</body>");
    return lines.join("\n");
  }

  partGeom(part, name, flip) {
    const material = `m_${part.id}`;
    this.materials[material] = rgbaOf(part.colour);
    const d = direction(part.pitch, flip * part.yaw);
    const grip = `${num(this.settings.foot_grip / 100)} 0.05 0.01`;
    const weight = (part.heaviness / 100) * DENSITY;
    if (part.shape === "stick") {
      const radius = part.thickness;
      const length = Math.max(part.length, 0.001);
      const mass = weight * (Math.PI * radius ** 2 * length + (4 / 3) * Math.PI * radius ** 3);
      const end = d.map((value) => length * value);
      return geom(name, "capsule", [radius], mass, material, { fromto: vec(0, 0, 0, ...end), friction: grip });
    }
    if (part.shape === "ball") {
      const radius = part.thickness;
      const mass = weight * (4 / 3) * Math.PI * radius ** 3;
      const centre = d.map((value) => part.length * value);
      return geom(name, "sphere", [radius], mass, material, { pos: vec(...centre), friction: grip });
    }
    const half = [Math.max(part.length, 0.01) / 2, part.width / 2, part.thickness / 2];
    const mass = weight * 8 * half[0] * half[1] * half[2];
    const centre = d.map((value) => half[0] * value);
    const quat = turn(part.pitch, flip * part.yaw);
    return geom(name, "box", half, mass, material, { pos: vec(...centre), quat: vec(...quat), friction: grip });
  }

  addMotor(joint, part, kind, side, mirrorSign, limit) {
    let mirrorName, sign;
    if (side === "middle") {
      mirrorName = joint;
      sign = kind === "swing" ? 1 : -1;
    } else {
      const other = side === "left" ? "right" : "left";
      mirrorName = `${other}_${joint.slice(joint.indexOf("_") + 1)}`;
      sign = mirrorSign;
    }
    this.motors.push(motor(joint, "custom", mirrorName, sign, side));
    const strength = (part.strength * this.settings.strength) / 100;
    const stiffness = (5 * strength * this.settings.stiffness) / 100;
    this.actuators.push(motorLine(joint, limit, stiffness, strength, this.settings.motor_type));
  }

  // The brain's order: middle motors, then the left side, then the right side in the same order
  orderMotors() {
    const lines = Object.fromEntries(this.motors.map((m, i) => [m.name, this.actuators[i]]));
    const middle = this.motors.filter((m) => m.side === "middle");
    const left = this.motors.filter((m) => m.side === "left");
    const rightByName = Object.fromEntries(this.motors.filter((m) => m.side === "right").map((m) => [m.name, m]));
    const right = left.map((m) => rightByName[m.mirror]);
    this.motors = [...middle, ...left, ...right];
    this.actuators = this.motors.map((m) => lines[m.name]);
  }
}

function jointKinds(part) {
  if (part.joint === "fixed") return [];
  if (part.joint === "hip") return ["swing", "side"];
  return [part.joint];
}

// A turn (quaternion w x y z) that points a shape's x axis along pitch (down) and yaw (left)
function turn(pitch, yaw) {
  const halfYaw = radians(yaw) / 2, halfPitch = radians(pitch) / 2;
  const yawQ = [Math.cos(halfYaw), 0.0, 0.0, Math.sin(halfYaw)];
  const pitchQ = [Math.cos(halfPitch), 0.0, Math.sin(halfPitch), 0.0];
  return multiply(yawQ, pitchQ);
}

function multiply([aw, ax, ay, az], [bw, bx, by, bz]) {
  return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw];
}

function rgbaOf(hexColour) {
  const channel = (i) => parseInt(hexColour.slice(i, i + 2), 16) / 255;
  return vec(channel(1), channel(3), channel(5), 1.0);
}
