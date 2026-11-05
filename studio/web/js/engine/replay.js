// Replays: a walk recorded as 3D frames that the page plays back with three.js. A port of studio/replay.py.
// A replay holds the walker's shapes, the ground, and for every frame the position and turn of every part.

import { BIAS_NONE, GEOM, nameOf } from "./physics.js";

export const REPLAY_FPS = 25;  // every second control step (the walker is controlled 50 times a second)
const GEOM_KINDS = { [GEOM.sphere]: "sphere", [GEOM.capsule]: "capsule", [GEOM.ellipsoid]: "ellipsoid", [GEOM.box]: "box", [GEOM.cylinder]: "cylinder" };

export const round = (value, places = 4) => {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
};
const roundAll = (values, places = 4) => Array.from(values, (value) => round(value, places));

// ---------- Holding the pose ----------

// Motor commands that hold the standing pose. Springy muscles simply aim for it
function holdingAction(env) {
  const range = env.model.actuator_ctrlrange;
  return env.build.motors.map((motor, m) => {
    const target = env.springy[m] ? env.build.standPose[motor.name] : 0;
    const low = range[m * 2], high = range[m * 2 + 1];
    return Math.min(Math.max(((target - low) / (high - low)) * 2 - 1, -1), 1);
  });
}

// Direct muscles push back toward the pose like a firm spring, as hard as they can, which shows whether they
// are strong enough to carry the body (a spring of the force limit per radian with a little damping)
export function holder(env) {
  const hold = holdingAction(env);
  if (env.springy.every(Boolean)) return () => hold;
  const targets = env.build.motors.map((motor) => env.build.standPose[motor.name]);
  return () => {
    const qpos = env.data.qpos, qvel = env.data.qvel;
    return env.build.motors.map((_, m) => {
      if (env.springy[m]) return hold[m];
      const push = -(qpos[env.jointQpos[m]] - targets[m]) - qvel[env.jointQvel[m]] / 200;
      return Math.min(Math.max(push, -1), 1);
    });
  };
}

// ---------- Shapes and ground ----------

// Every shape of the walker (and the rock) with its body, size, place and colour
export function sceneShapes(env) {
  const { mujoco, model } = env;
  const bodies = [];
  for (let i = 1; i < model.nbody; i++) bodies.push(nameOf(mujoco, model, "body", i));
  const shapes = [];
  const type = model.geom_type, bodyOf = model.geom_bodyid, matid = model.geom_matid;
  const size = model.geom_size, pos = model.geom_pos, quat = model.geom_quat, matRgba = model.mat_rgba, geomRgba = model.geom_rgba;
  for (let g = 0; g < model.ngeom; g++) {
    const kind = GEOM_KINDS[type[g]];
    const body = bodyOf[g];
    if (!kind || body === 0) continue;
    const material = matid[g];
    const colour = material >= 0 ? matRgba.slice(material * 4, material * 4 + 4) : geomRgba.slice(g * 4, g * 4 + 4);
    shapes.push({ body: body - 1, type: kind, size: roundAll(size.slice(g * 3, g * 3 + 3)), pos: roundAll(pos.slice(g * 3, g * 3 + 3)),
      quat: roundAll(quat.slice(g * 4, g * 4 + 4)), rgba: roundAll(colour, 3), name: nameOf(mujoco, model, "geom", g) });
  }
  return { bodies, shapes };
}

// The ground near the walk, as a coarse grid of heights (tenths of a millimetre), or a plain plane
export function groundPatch(env, xFrom, xTo, yHalf = null) {
  const layout = env.scene.layout;
  if (!layout || !env.heights) return { kind: "plane" };
  const step = env.world.terrain === "cobblestones" ? 0.015 : 0.03;
  xFrom = Math.max(layout.xMin, xFrom);
  xTo = Math.min(layout.xMax, xTo);
  yHalf = Math.min(layout.yHalf, yHalf ?? 0.9 * Math.max(env.build.size, 1.0));
  const xs = [], ys = [];
  for (let x = xFrom; x < xTo + step / 2; x += step) xs.push(x);
  for (let y = -yHalf; y < yHalf + step / 2; y += step) ys.push(y);
  const clampIndex = (value, most) => Math.min(Math.max(Math.round(value), 0), most);
  const columns = xs.map((x) => clampIndex((x - layout.xMin) / layout.cell, layout.columns - 1));
  const rows = ys.map((y) => clampIndex((y + layout.yHalf) / layout.cell, layout.rows - 1));
  const data = [];
  for (const row of rows) for (const column of columns) data.push(Math.round(env.heights[row * layout.columns + column] * 10000));
  return { kind: "heights", x0: round(xs[0]), y0: round(ys[0]), step, rows: ys.length, cols: xs.length, data };
}

// ---------- Recording ----------

// The place and turn of every body except the world, flattened: x y z qw qx qy qz for each
export function frameOf(env) {
  const xpos = env.data.xpos, xquat = env.data.xquat;
  const frame = [];
  for (let b = 1; b < env.model.nbody; b++) {
    frame.push(round(xpos[b * 3]), round(xpos[b * 3 + 1]), round(xpos[b * 3 + 2]));
    frame.push(round(xquat[b * 4]), round(xquat[b * 4 + 1]), round(xquat[b * 4 + 2]), round(xquat[b * 4 + 3]));
  }
  return frame;
}

// Run one attempt and record it. act(observation) picks the motor commands; null holds the pose.
// brain (optional) has cells(observation) for showing the brain at work
export function record(env, { act = null, seed = 1000, seconds = null, keepGoingAfterFall = 1.0, groundHalfWidth = null, brain = null } = {}) {
  let observation = env.reset(seed);
  const { bodies, shapes } = sceneShapes(env);
  const stepsPerFrame = Math.round(50 / REPLAY_FPS);
  let limit = seconds === null ? env.maxSteps : Math.round(seconds / env.controlSeconds);
  const frames = [frameOf(env)];
  const contacts = [env.footContacts().map(Number)];
  const commands = [env.build.motors.map(() => 0.0)];
  const cells = brain ? [brain.cells(observation)] : null;
  const hold = holder(env);
  const extraSteps = Math.round(keepGoingAfterFall / env.controlSeconds);
  let total = 0.0, fellAt = null, step = 0;
  while (step < limit) {
    const action = act ? act(observation) : hold(observation);
    const result = env.step(action);
    observation = result.observation;
    step += 1;
    if (fellAt === null) total += result.reward;
    if (step % stepsPerFrame === 0) {
      frames.push(frameOf(env));
      contacts.push(env.footContacts().map(Number));
      commands.push(Array.from(action, (value) => round(Math.min(Math.max(value, -1), 1), 2)));
      if (cells) cells.push(brain.cells(observation));
    }
    if (result.terminated && fellAt === null) {
      fellAt = step;
      if (act) limit = Math.min(limit, step + extraSteps);
    }
    if (!env.finite()) break;
  }
  const end = fellAt || step;
  const index = Math.min(frames.length - 1, Math.floor(end / stepsPerFrame));
  const distance = frames[index][0] - frames[0][0];
  const secondsWalked = end * env.controlSeconds;
  const xs = frames.map((frame) => frame[0]);
  return {
    fps: REPLAY_FPS, bodies, shapes, frames,
    ground: groundPatch(env, Math.min(...xs) - 1.5, Math.max(...xs) + 2.5, groundHalfWidth),
    style: env.world.values.style, world: env.world.name, animal: env.build.animal,
    feet: env.build.feet.map((foot) => foot.name), motors: env.build.motorNames,
    contacts, commands, cells, size: env.build.size, stand_height: env.scene.standHeight,
    stats: { distance: round(distance, 3), seconds: round(secondsWalked, 2), fell: fellAt !== null,
      target_speed: round(env.targetSpeed, 3), average_speed: round(distance / Math.max(secondsWalked, 1e-9), 3),
      score: round(total, 1), terrain_height_cm: round(env.attemptHeight * 100, 1), rock_hits: env.rockHits,
      turn_rate: round((env.turnRate * 180) / Math.PI, 1) },
  };
}

export { BIAS_NONE };
