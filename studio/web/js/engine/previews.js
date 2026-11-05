// Quick looks for the page: a creature standing still and a drop test. A port of part of studio/previews.py.
// Both come back as replays, so the page shows them in the same 3D viewer.

import { buildCreature } from "./creatures/index.js";
import { WalkEnv, observationSize } from "./env.js";
import { GEOM, HINGE, idOf, nameOf, setStandPose } from "./physics.js";
import { frameOf, groundPatch, record, round, sceneShapes } from "./replay.js";
import { buildScene, disposeScene } from "./scene.js";
import { plainWorld, worldFromDict } from "./worlds.js";

// The creature standing in its starting pose, with a few facts about it
export function creaturePreview(mujoco, animalKey, values) {
  const build = buildCreature(animalKey, values);
  const scene = buildScene(mujoco, build, plainWorld());
  const env = new WalkEnv(mujoco, build, scene.world, { scene });
  try {
    const { model, data } = env;
    setStandPose(mujoco, model, data, build, scene.standHeight);
    mujoco.mj_forward(model, data);
    const { bodies, shapes } = sceneShapes(env);
    const torso = env.torso;
    const com = data.subtree_com;
    return {
      bodies, shapes, frames: [frameOf(env)], fps: 25, ground: { kind: "plane" }, style: "studio", size: build.size,
      stand_height: scene.standHeight, facts: creatureFacts(mujoco, build, scene, data),
      centre_of_mass: [com[torso * 3], com[torso * 3 + 1], com[torso * 3 + 2]].map((v) => round(v)),
      support: supportArea(mujoco, model, data, build), joints: jointMarks(mujoco, model, build), view_size: viewSize(model, data, build, torso),
    };
  } finally {
    env.dispose();
    disposeScene(scene);
  }
}

// Where every joint sits, for the motor rings in the 3D view (kind from the axis; a part with no hinge is glued)
const JOINT_KINDS = ["side", "swing", "twist"];  // axis x, y, z

export function jointMarks(mujoco, model, build) {
  const motors = new Set(build.motorNames);
  const marks = [];
  const torso = idOf(mujoco, model, "body", "torso");
  const jntBody = model.jnt_bodyid, jntType = model.jnt_type, axisOf = model.jnt_axis, posOf = model.jnt_pos;
  const geomBody = model.geom_bodyid, parentOf = model.body_parentid;
  for (let body = 1; body < model.nbody; body++) {
    const hinges = [];
    for (let j = 0; j < model.njnt; j++) if (jntBody[j] === body && jntType[j] === HINGE) hinges.push(j);
    for (const joint of hinges) {
      const axis = [axisOf[joint * 3], axisOf[joint * 3 + 1], axisOf[joint * 3 + 2]];
      const biggest = axis.map(Math.abs).reduce((best, value, i, all) => (value > all[best] ? i : best), 0);
      const name = nameOf(mujoco, model, "joint", joint);
      marks.push({ body: body - 1, pos: [posOf[joint * 3], posOf[joint * 3 + 1], posOf[joint * 3 + 2]].map((v) => round(v)),
        axis: axis.map((v) => round(v)), kind: JOINT_KINDS[biggest], name, motor: motors.has(name) });
    }
    let hasShape = false;
    for (let g = 0; g < model.ngeom; g++) if (geomBody[g] === body) { hasShape = true; break; }
    if (!hinges.length && body !== torso && hasShape && parentOf[body] !== 0) {
      marks.push({ body: body - 1, pos: [0, 0, 0], kind: "glued", name: nameOf(mujoco, model, "body", body), motor: false });
    }
  }
  return marks;
}

// How big the creature looks, for the camera only: its reach from the torso, 0.3 m counting as size 1
function viewSize(model, data, build, torso) {
  const xpos = data.xpos, gx = data.geom_xpos, size = model.geom_size, geomBody = model.geom_bodyid;
  let reach = 0.0;
  for (let g = 0; g < model.ngeom; g++) {
    if (geomBody[g] === 0) continue;
    const away = Math.hypot(gx[g * 3] - xpos[torso * 3], gx[g * 3 + 1] - xpos[torso * 3 + 1]) + Math.max(size[g * 3], size[g * 3 + 1], size[g * 3 + 2]);
    reach = Math.max(reach, away);
  }
  return round(Math.max(build.size, reach / 0.3), 3);
}

// Where the feet stand: the outline (convex hull) of the lowest points of the parts that feel the ground
function supportArea(mujoco, model, data, build) {
  const geoms = build.feet.flatMap((foot) => foot.geoms).map((name) => idOf(mujoco, model, "geom", name));
  const points = [];
  for (const g of geoms) {
    const cx = data.geom_xpos[g * 3], cy = data.geom_xpos[g * 3 + 1];
    const s = model.geom_size;
    const reach = model.geom_type[g] !== GEOM.capsule ? Math.max(s[g * 3], s[g * 3 + 1], s[g * 3 + 2]) : s[g * 3];
    for (let k = 0; k < 8; k++) {
      const angle = (2 * Math.PI * k) / 8;
      points.push([cx + reach * Math.cos(angle) * 0.5, cy + reach * Math.sin(angle) * 0.5]);
    }
  }
  if (points.length < 3) return [];
  return hull(points).map(([x, y]) => [round(x), round(y)]);
}

function hull(points) {
  const unique = [...new Map(points.map((p) => [p.join(","), p])).values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (const p of [...unique].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

export function creatureFacts(mujoco, build, scene, standingData = null) {
  const model = scene.model;
  let strongest = 0.0;
  for (let i = 0; i < model.nu; i++) strongest = Math.max(strongest, model.actuator_forcerange[i * 2 + 1]);
  const ranges = {};
  for (const motor of build.motors) ranges[motor.name] = round((model.jnt_range[idOf(mujoco, model, "joint", motor.name) * 2 + 1] * 180) / Math.PI, 1);
  return {
    mass_kg: round(scene.totalMass, 3), height_cm: round(topOf(mujoco, scene, standingData) * 100, 1),
    hip_height_cm: round(scene.standHeight * 100, 1), motors: build.motors.length, feet: build.feet.length,
    motor_names: build.motorNames, motor_groups: [...new Set(build.motors.map((m) => m.group))].sort(),
    strongest_motor_nm: round(strongest, 2), senses: observationSize(build, { feet: true, ground: false }),
    symmetric_ok: build.symmetricOk, senses_with_eyes: observationSize(build, { feet: true, ground: true }),
    joint_ranges_deg: ranges,
  };
}

// The highest point of the standing creature above the ground
function topOf(mujoco, scene, standingData) {
  const model = scene.model;
  const data = standingData || new mujoco.MjData(model);
  try {
    if (!standingData) {
      setStandPose(mujoco, model, data, scene.build, scene.standHeight);
      mujoco.mj_forward(model, data);
    }
    let top = 0.0;
    const size = model.geom_size;
    for (let g = 0; g < model.ngeom; g++) {
      if (model.geom_bodyid[g] === 0) continue;
      top = Math.max(top, data.geom_xpos[g * 3 + 2] + Math.max(size[g * 3], size[g * 3 + 1], size[g * 3 + 2]));
    }
    return top;
  } finally {
    if (!standingData) data.delete();
  }
}

// The creature placed in a world with its muscles holding the starting pose for a few seconds
export function dropTest(mujoco, animalKey, values, worldData = null, seconds = 4.0) {
  const build = buildCreature(animalKey, values);
  const world = worldData ? worldFromDict(worldData) : plainWorld();
  const height = world.terrain !== "flat" ? world.tallest : 0.0;
  const env = new WalkEnv(mujoco, build, world, { terrainHeight: height, targetSpeed: (world.speedRange[0] + world.speedRange[1]) / 2 });
  try {
    const replay = record(env, { seed: 7, seconds });
    replay.stats.stood = !replay.stats.fell;
    if (worldData) {  // a good stretch of the ground ahead, so stairs, hurdles and slopes can be seen
      replay.ground = groundPatch(env, -1.5, 8.0);
      replay.marks_to = 8;
    }
    return replay;
  } finally {
    env.dispose();
  }
}
