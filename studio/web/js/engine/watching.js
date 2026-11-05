// Watching trained walkers: a test drive in any world, races side by side, the report card and live mode.
// A port of previews.watch/race, studio/challenges.py and studio/live.py. A brain here is the plain data saved by
// training (brain/best), with its sense scaling; see network.js.

import { buildCreature } from "./creatures/index.js";
import { cleanDesign, designSenses } from "./design.js";
import { WalkEnv } from "./env.js";
import { deterministicActor, cellReader } from "./network.js";
import { frameOf, groundPatch, record, REPLAY_FPS, round, sceneShapes } from "./replay.js";
import { buildScene, disposeScene } from "./scene.js";
import { plainWorld, worldFromDict, worldToDict } from "./worlds.js";

// ---------- A test drive ----------

// A trained walker in any world. trained: { setup, brain }
export function watch(mujoco, trained, { world: worldData = null, speed = null, height = null, seed = 1000, seconds = null, groundHalfWidth = null, turn = 0 } = {}) {
  const { setup, brain } = trained;
  const build = buildCreature(setup.creature.animal, setup.creature.values);
  const world = worldFromDict(worldData || setup.world);
  const senses = designSenses(cleanDesign(setup.brain));
  speed = speed === null || speed === undefined ? (world.speedRange[0] + world.speedRange[1]) / 2 : Number(speed);
  if (height === null || height === undefined) height = world.values.mixed_heights ? world.tallest * 0.75 : world.tallest;
  const env = new WalkEnv(mujoco, build, world, { senses, targetSpeed: speed, terrainHeight: Number(height), turnRate: Number(turn || 0) });
  try {
    const replay = record(env, { act: deterministicActor(brain), seed: Math.trunc(seed), seconds, keepGoingAfterFall: 2.0,
      groundHalfWidth, brain: groundHalfWidth ? null : cellReader(brain) });
    replay.name = setup.name || "";
    replay.creature_name = setup.creature.name || setup.creature.animal;
    return replay;
  } finally {
    env.dispose();
  }
}

// ---------- Races ----------

// Several trained walkers in the same world, side by side in lanes, as one replay. Each walks in its own simulation
// on the same ground (same seed); the lanes only exist in the picture. walkers: [{ setup, brain }]
export function race(mujoco, walkers, { world: worldData = {}, speed = null, height = null, seed = 1000, seconds = 20 } = {}) {
  if (!walkers.length) throw new Error("Pick at least one walker for the race.");
  const values = { ...(worldData.values || {}) };
  values.attempt_seconds = Math.max(Number(values.attempt_seconds ?? 20), Number(seconds));
  const world = worldFromDict({ ...worldData, values });
  const sizes = walkers.map(({ setup }) => buildCreature(setup.creature.animal, setup.creature.values).size);
  const lane = Math.max(0.9, Math.max(...sizes) * 0.9);
  const replays = walkers.map((walker) => watch(mujoco, walker, { world: worldToDict(world), speed, height, seed, seconds, groundHalfWidth: lane / 2 }));
  return mergeLanes(replays, lane);
}

function mergeLanes(replays, lane) {
  const count = replays.length;
  const offsets = replays.map((_, i) => ((count - 1) / 2) * lane - i * lane);  // first walker on the left
  const bodies = [], shapes = [], torsos = [], labels = [], grounds = [];
  replays.forEach((replay, i) => {
    const first = bodies.length;
    torsos.push(first);
    bodies.push(...replay.bodies);
    shapes.push(...replay.shapes.map((shape) => ({ ...shape, body: shape.body + first })));
    labels.push({ text: replay.name.replace(" (from learn_to_walk)", ""), body: first, z: replay.stand_height * 2.4 + 0.12 });
    const ground = { ...replay.ground };
    if (ground.kind === "heights") ground.y0 = round(ground.y0 + offsets[i]);
    grounds.push(ground);
  });
  const length = Math.max(...replays.map((r) => r.frames.length));
  const frames = [];
  for (let index = 0; index < length; index++) {
    const frame = [];
    replays.forEach((replay, i) => {
      const values = [...replay.frames[Math.min(index, replay.frames.length - 1)]];
      for (let start = 0; start < values.length; start += 7) values[start + 1] = round(values[start + 1] + offsets[i]);
      frame.push(...values);
    });
    frames.push(frame);
  }
  const results = replays.map((r) => ({ name: r.name, creature: r.creature_name, ...r.stats })).sort((a, b) => b.distance - a.distance);
  return {
    fps: replays[0].fps, bodies, shapes, frames, ground: { kind: "lanes", patches: grounds, lane, count }, style: replays[0].style,
    size: Math.max(1.0, (count * lane) / 1.2), stand_height: Math.max(...replays.map((r) => r.stand_height)), torsos, labels, results,
    stats: { seconds: Math.max(...replays.map((r) => r.stats.seconds)) },
  };
}

// ---------- The report card ----------

export const CHALLENGE_SECONDS = 15.0;
const MIN_SPEED_SHARE = 0.6;
const CHALLENGE_SEED = 2024;

export const CHALLENGES = [
  { key: "speed", title: "Speed", description: "Walk or run at faster and faster asked speeds on flat ground.", levels: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5], unit: "m/s" },
  { key: "stones", title: "Cobblestones", description: "Cobblestones that get taller.", levels: [1, 2, 3, 4, 5, 6], unit: "cm" },
  { key: "stairs", title: "Stairs", description: "Flights of steps that get taller.", levels: [1, 2, 3, 4, 5, 6], unit: "cm" },
  { key: "ice", title: "Ice", description: "Ground that gets more and more slippery.", levels: [60, 40, 25, 15, 10], unit: "% grip" },
  { key: "backpack", title: "Backpack", description: "A heavier and heavier box on its back.", levels: [10, 25, 50, 100, 200], unit: "% of its weight" },
  { key: "wind", title: "Wind", description: "Stronger and stronger gusts from random directions.", levels: [10, 20, 35, 50, 75], unit: "% of its weight" },
  { key: "gravity", title: "Other planets", description: "Walking where gravity is different.", levels: [3.7, 1.6, 20.0, 25.0], unit: "m/s²", ladder: false },
];

// The world of one level, built on top of the walker's own world (its speeds and scoring)
function levelWorld(base, challenge, level, mass) {
  const values = { ...(base.values || {}), terrain: "flat", terrain_height: 0.0, rocks: false, wind: false, backpack: false,
    gradually_harder: false, mixed_heights: false, attempt_seconds: CHALLENGE_SECONDS };
  if (challenge === "stones") Object.assign(values, { terrain: "cobblestones", terrain_height: level, style: "street" });
  else if (challenge === "stairs") Object.assign(values, { terrain: "stairs", terrain_height: level, style: "street" });
  else if (challenge === "ice") Object.assign(values, { ground_grip: level, style: "snow" });
  else if (challenge === "backpack") Object.assign(values, { backpack: true, backpack_weight: Math.max(10.0, (level / 100) * mass * 1000) });
  else if (challenge === "wind") Object.assign(values, { wind: true, wind_strength: level, wind_every: 2.0, style: "night" });
  else if (challenge === "gravity") Object.assign(values, { gravity: level, style: level < 5 ? "moon" : "desert" });
  else if (challenge === "speed") Object.assign(values, { slowest_speed: Math.min(level, values.slowest_speed ?? 0.2), fastest_speed: Math.max(level, 0.1) });
  return { id: `challenge_${challenge}`, name: `${challenge[0].toUpperCase()}${challenge.slice(1)} challenge`, values };
}

export function runLevel(mujoco, trained, challenge, level, keepReplay = false) {
  const { setup, brain } = trained;
  const build = buildCreature(setup.creature.animal, setup.creature.values);
  const senses = designSenses(cleanDesign(setup.brain));
  const plain = buildScene(mujoco, build, plainWorld("plain"));
  const mass = plain.totalMass;
  disposeScene(plain);
  const world = worldFromDict(levelWorld(setup.world, challenge, level, mass));
  const own = worldFromDict(setup.world);
  const speed = challenge === "speed" ? Number(level) : (own.speedRange[0] + own.speedRange[1]) / 2;
  const env = new WalkEnv(mujoco, build, world, { senses, targetSpeed: speed, terrainHeight: world.tallest });
  try {
    const replay = record(env, { act: deterministicActor(brain), seed: CHALLENGE_SEED, seconds: CHALLENGE_SECONDS });
    const stats = replay.stats;
    const passed = !stats.fell && stats.average_speed >= MIN_SPEED_SHARE * speed;
    const result = { level, passed, fell: stats.fell, seconds: stats.seconds, distance: stats.distance, average_speed: stats.average_speed, asked_speed: speed };
    if (keepReplay) result.replay = replay;
    return result;
  } finally {
    env.dispose();
  }
}

function medal(passed, total) {
  const share = passed / total;
  if (share >= 0.99) return "gold";
  if (share >= 0.6) return "silver";
  if (share >= 0.3) return "bronze";
  return "none";
}

// Every challenge, level by level; each ladder stops at the first level it fails. onProgress(share, message)
export function reportCard(mujoco, trained, onProgress = () => {}, date = new Date().toISOString().slice(0, 19)) {
  const results = { challenges: [], seconds: CHALLENGE_SECONDS, date };
  const total = CHALLENGES.reduce((sum, c) => sum + c.levels.length, 0);
  let done = 0;
  for (const challenge of CHALLENGES) {
    const levels = [];
    let failed = false;
    for (const level of challenge.levels) {
      onProgress(done / total, `${challenge.title}: ${level} ${challenge.unit}`);
      if (failed) levels.push({ level, passed: false, skipped: true });
      else {
        const outcome = runLevel(mujoco, trained, challenge.key, level);
        levels.push(outcome);
        failed = (challenge.ladder ?? true) && !outcome.passed;
      }
      done += 1;
    }
    const passed = levels.filter((l) => l.passed).length;
    results.challenges.push({ key: challenge.key, title: challenge.title, description: challenge.description, unit: challenge.unit,
      levels, passed, medal: medal(passed, levels.length) });
  }
  onProgress(1, "Done");
  return results;
}

// ---------- Live mode ----------

const PUSH_SHARE = 0.6;        // a push: this share of the walker's weight, for this long
const PUSH_SECONDS = 0.15;
const MOST_FRAMES = 25 * 60;
const STAND_UP_AFTER = 2.0;    // a fallen walker is put back on its feet after this long

// A trained walker running in real time on this page. The page asks for new frames and sends commands
export class LiveSession {
  constructor(mujoco, trained, worldData, speed) {
    const { setup, brain } = trained;
    this.build = buildCreature(setup.creature.animal, setup.creature.values);
    const world = { ...(worldData || setup.world) };
    world.values = { ...(world.values || {}), attempt_seconds: 60 };
    this.world = worldFromDict(world);
    this.env = new WalkEnv(mujoco, this.build, this.world, { senses: designSenses(cleanDesign(setup.brain)), targetSpeed: Number(speed),
      terrainHeight: this.world.tallest * 0.75, turnRate: 0.0 });
    this.act = deterministicActor(brain);
    this.id = Math.random().toString(16).slice(2, 12);
    this.speed = Number(speed);
    this.frames = [];
    this.firstFrame = 0;
    this.events = [];
    this.pushUntil = -1.0;
    this.push = [0, 0, 0];
    this.fallenAt = null;
    this.seed = 11;
    this.running = true;
    this.observation = this.env.reset(this.seed);
    this.frames.push(frameOf(this.env));
    this.count = 0;
    this.timer = setInterval(() => this.tick(), this.env.controlSeconds * 1000 * 2);  // two steps per tick, 25 ticks a second
    this.lastAsked = performance.now();
  }

  // What the page needs once: the shapes and the whole floor
  header() {
    const { bodies, shapes } = sceneShapes(this.env);
    const layout = this.env.scene.layout;
    return { id: this.id, fps: REPLAY_FPS, bodies, shapes, ground: layout ? groundPatch(this.env, layout.xMin, layout.xMax) : { kind: "plane" },
      style: this.world.values.style, size: this.build.size, stand_height: this.env.scene.standHeight, frames: [this.frames[this.frames.length - 1]],
      floor_end: layout ? layout.xMax - 1.0 : null };
  }

  tick() {
    if (!this.running) return;
    if (performance.now() - this.lastAsked > 20000) { this.stop(); return; }  // nobody watching any more
    this.step();
    this.step();
    this.frames.push(frameOf(this.env));
    if (this.frames.length > MOST_FRAMES) {
      const drop = this.frames.length - MOST_FRAMES;
      this.frames.splice(0, drop);
      this.firstFrame += drop;
    }
  }

  step() {
    const env = this.env;
    env.targetSpeed = this.speed;
    const action = this.act(this.observation);
    env.data.xfrc_applied.set(env.data.time < this.pushUntil ? this.push : [0, 0, 0], env.torso * 6);
    const result = env.step(action);
    this.observation = result.observation;
    const layout = env.scene.layout;
    const offTheFloor = layout && env.data.qpos[0] > layout.xMax - 1.0;
    if ((result.terminated || offTheFloor) && this.fallenAt === null) {
      this.fallenAt = env.data.time;
      this.events.push(result.terminated ? "fell" : "end");
    }
    if (this.fallenAt !== null && env.data.time - this.fallenAt > STAND_UP_AFTER) this.restart();
  }

  restart() {
    this.seed += 1;
    this.observation = this.env.reset(this.seed);
    this.fallenAt = null;
    this.pushUntil = -1.0;
    this.events.push("restart");
  }

  framesAfter(number) {
    this.lastAsked = performance.now();
    const start = Math.max(number + 1, this.firstFrame);
    const frames = this.frames.slice(start - this.firstFrame);
    const events = this.events;
    this.events = [];
    const data = this.env.data;
    return { first: start, frames, events, running: this.running,
      state: { x: this.env.progress, speed: Math.hypot(data.qvel[0], data.qvel[1]), asked: this.speed, fallen: this.fallenAt !== null, time: data.time } };
  }

  command(command) {
    this.lastAsked = performance.now();
    if ("turn" in command) this.env.setTurnRate(Math.min(Math.max(Number(command.turn), -120.0), 120.0));
    if ("speed" in command) this.speed = Math.min(Math.max(Number(command.speed), 0.0), 4.0);
    if (["left", "right", "front", "back"].includes(command.push)) {
      // The side the push comes from: from the left (+y) it shoves the walker toward -y
      const [dx, dy] = { left: [0, -1], right: [0, 1], front: [-1, 0], back: [1, 0] }[command.push];
      const strength = PUSH_SHARE * this.env.scene.totalMass * this.world.values.gravity;
      this.push = [strength * dx, strength * dy, 0.0];
      this.pushUntil = this.env.data.time + PUSH_SECONDS;
    }
    if (command.restart) this.restart();
    if (command.stop) this.stop();
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    clearInterval(this.timer);
    this.env.dispose();
  }
}
