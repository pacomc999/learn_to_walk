// The practice world: any animal in any world, stepped 50 times a second. A port of studio/env.py.
// Each attempt the walker starts standing with one leg a little ahead, is asked for a speed, gets a fresh floor,
// and earns points every moment (see the world's reward settings) until it falls or its time is up.

import { BIAS_NONE, idOf, setStandPose } from "./physics.js";
import { Random } from "./random.js";
import { buildScene } from "./scene.js";
import { groundHeights, heightAt, highestNear } from "./terrain.js";

export const FRAME_SKIP = 4;              // physics steps per control step
const START_NOISE = 0.02;                 // how far each joint may start away from the standing pose
const START_STRIDE = 2 * START_NOISE;     // the stride stance (learn_to_walk)
export const FALL_HEIGHT_SHARE = 0.6;     // fallen below this share of the standing height
export const FALL_UP_Z = 0.64;            // or tilted more than about 50 degrees
export const GROUND_SCAN = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.45, 0.6];  // ground heights ahead (times the size)
const GUST_SECONDS = 0.5;
const FIRST_EVENT = 2.0;
const THROW_DISTANCE = 0.8;
const ROCK_LIFETIME = 1.0;
const OBSERVATION_LIMIT = 1e6;
export const PART_NAMES = ["speed", "alive", "energy", "straight", "fall", "end_stops", "level", "crouch", "smooth", "feet", "heading", "ride"];
const AIR_TIME_GOAL = 0.25;
export const DEFAULT_SENSES = { feet: true, ground: false, turn: false };
const TURN_CHANGE_SECONDS = [3.0, 7.0];
const TURN_SENSES = 3;

// How many numbers the walker senses with these senses
export function observationSize(build, senses) {
  let size = 12 + 2 * build.motors.length + 1;
  if (senses.feet ?? true) size += build.feet.length;
  if (senses.ground ?? false) size += GROUND_SCAN.length;
  if (senses.turn ?? false) size += TURN_SENSES;
  return size;
}

const zeroParts = () => Object.fromEntries(PART_NAMES.map((name) => [name, 0.0]));
const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const radians = (degrees) => (degrees * Math.PI) / 180;

export class WalkEnv {
  // options: senses, targetSpeed, terrainHeight, scene (to share one), turnRate (degrees per second)
  constructor(mujoco, build, world, { senses = null, targetSpeed = null, terrainHeight = null, scene = null, turnRate = null } = {}) {
    this.mujoco = mujoco;
    this.build = build;
    this.world = world;
    this.senses = { ...DEFAULT_SENSES, ...(senses || {}) };
    this.ownScene = !scene;
    this.scene = scene || buildScene(mujoco, build, world);
    this.model = this.scene.model;
    this.data = new mujoco.MjData(this.model);
    this.random = new Random(0);
    this.fixedTargetSpeed = targetSpeed;
    this.fixedHeight = terrainHeight;
    this.fixedTurnRate = turnRate;
    this.turnRate = 0.0;
    this.askedHeading = 0.0;
    this.nextTurnChange = 0.0;
    this.progress = 0.0;
    this.targetSpeed = targetSpeed ?? (world.speedRange[0] + world.speedRange[1]) / 2;
    this.difficulty = 1.0;
    this.heights = null;
    this.attemptHeight = 0.0;
    this.controlSeconds = this.model.opt.timestep * FRAME_SKIP;
    this.maxSteps = Math.round(world.values.attempt_seconds / this.controlSeconds);
    this.stepCount = 0;
    this.startX = 0.0;
    this.fallHeight = FALL_HEIGHT_SHARE * this.scene.standHeight;
    this.findParts();
    this.rock = world.values.rocks ? this.rockParts() : null;
    this.nextRock = this.nextGust = FIRST_EVENT;
    this.rockThrownAt = null;
    this.gustUntil = -1.0;
    this.gustForce = [0, 0, 0];
    this.rockHits = 0;
    this.previousAction = new Float64Array(build.motors.length);
    this.airTime = new Float64Array(build.feet.length);
    this.episodeParts = zeroParts();
    this.lastEnergy = 0.0;
    this.contactCache = null;
    this.actionSize = build.motors.length;
    this.observationSize = observationSize(build, this.senses);
  }

  // Where things are in the simulation's lists
  findParts() {
    const { mujoco, model, build } = this;
    this.torso = idOf(mujoco, model, "body", "torso");
    const joints = build.motors.map((m) => idOf(mujoco, model, "joint", m.name));
    this.jointQpos = Int32Array.from(joints, (j) => model.jnt_qposadr[j]);
    this.jointQvel = Int32Array.from(joints, (j) => model.jnt_dofadr[j]);
    this.jointRanges = joints.map((j) => [model.jnt_range[j * 2], model.jnt_range[j * 2 + 1]]);
    this.floor = idOf(mujoco, model, "geom", "floor");
    this.footOfGeom = new Int32Array(model.ngeom).fill(-1);  // the foot each shape belongs to (-1 for none)
    build.feet.forEach((foot, index) => { for (const name of foot.geoms) this.footOfGeom[idOf(mujoco, model, "geom", name)] = index; });
    const bodyOf = model.geom_bodyid, rootOf = model.body_rootid;
    this.walkerGeoms = new Set();
    for (let g = 0; g < model.ngeom; g++) if (bodyOf[g] !== 0 && rootOf[bodyOf[g]] === this.torso) this.walkerGeoms.add(g);
    this.springy = Array.from({ length: model.nu }, (_, i) => model.actuator_biastype[i] !== BIAS_NONE);
  }

  dispose() {
    if (this.data) { this.data.delete(); this.data = null; }
    if (this.ownScene && this.scene.model) { this.scene.model.delete(); this.scene.model = null; }
  }

  // ---------- Facts ----------

  groundAt(x, y) {
    return this.heights ? heightAt(this.heights, this.scene.layout, x, y) : 0.0;
  }

  torsoHeight() {
    const xpos = this.data.xpos, t = this.torso * 3;
    return xpos[t + 2] - this.groundAt(xpos[t], xpos[t + 1]);
  }

  torsoUp() {
    return this.data.xmat[this.torso * 9 + 8];
  }

  finite() {
    const qpos = this.data.qpos, qvel = this.data.qvel;
    let sum = 0;
    for (let i = 0; i < qpos.length; i++) sum += qpos[i];
    for (let i = 0; i < qvel.length; i++) sum += qvel[i];
    return Number.isFinite(sum);
  }

  // ---------- Starting an attempt ----------

  setDifficulty(difficulty) {
    this.difficulty = clamp(Number(difficulty), 0.0, 1.0);
  }

  pickHeight() {
    if (this.fixedHeight !== null) return this.fixedHeight;
    let tallest = this.world.tallest;
    if (this.world.values.gradually_harder) tallest *= this.difficulty;
    if (this.world.values.mixed_heights) return tallest > 0 ? this.random.uniform(0.0, tallest) : 0.0;
    return tallest;
  }

  pickTurnRate() {
    if (this.fixedTurnRate !== null) return radians(this.fixedTurnRate);
    if (!this.world.turning) return 0.0;
    const most = radians(this.world.values.turn_rate);
    return this.random.uniform(-most, most);
  }

  // The asked heading moves on with the asked turn; in practice the turn changes every few seconds
  advanceHeading() {
    this.askedHeading += this.turnRate * this.controlSeconds;
    if (this.fixedTurnRate === null && this.world.turning && this.data.time >= this.nextTurnChange) {
      this.turnRate = this.pickTurnRate();
      this.nextTurnChange = this.data.time + this.random.uniform(...TURN_CHANGE_SECONDS);
    }
  }

  setTurnRate(degreesPerSecond) {
    this.fixedTurnRate = Number(degreesPerSecond);
    this.turnRate = radians(this.fixedTurnRate);
  }

  // The torso's heading (yaw) and how far it is from the asked heading, in radians from -pi to pi
  yaw() {
    const q = this.data.qpos;
    const [w, x, y, z] = [q[3], q[4], q[5], q[6]];
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  headingError() {
    const twoPi = 2 * Math.PI;
    const value = (this.askedHeading - this.yaw() + Math.PI) % twoPi;
    return (value < 0 ? value + twoPi : value) - Math.PI;  // Python's % never goes negative
  }

  pickSpeed() {
    if (this.fixedTargetSpeed !== null) return this.fixedTargetSpeed;
    return this.random.uniform(...this.world.speedRange);
  }

  reset(seed = null) {
    if (seed !== null && seed !== undefined) this.random = new Random(seed);
    const { mujoco, model, data } = this;
    this.targetSpeed = this.pickSpeed();
    setStandPose(mujoco, model, data, this.build, this.scene.standHeight);
    const qpos = data.qpos;
    for (const adr of this.jointQpos) qpos[adr] += this.random.uniform(-START_NOISE, START_NOISE);
    this.takeStrideStance();
    this.layGround();
    if (this.rock) this.parkRock();
    mujoco.mj_forward(model, data);
    this.contactCache = null;
    this.stepCount = 0;
    this.startX = data.qpos[0];
    this.askedHeading = 0.0;
    this.progress = 0.0;
    this.turnRate = this.pickTurnRate();
    this.nextTurnChange = this.random.uniform(...TURN_CHANGE_SECONDS);
    this.nextRock = this.nextGust = FIRST_EVENT;
    this.rockThrownAt = null;
    this.gustUntil = -1.0;
    this.gustForce = [0, 0, 0];
    data.xfrc_applied.fill(0.0);
    this.rockHits = 0;
    this.previousAction = new Float64Array(this.build.motors.length);
    this.airTime = new Float64Array(this.build.feet.length);
    this.episodeParts = zeroParts();
    return this.observation();
  }

  // A coin flip picks the leading leg of each pair
  takeStrideStance() {
    const lead = this.random.random() < 0.5 ? 1.0 : -1.0;
    const qpos = this.data.qpos, adr = this.model.jnt_qposadr;
    for (const [first, second] of this.build.stridePairs) {
      qpos[adr[idOf(this.mujoco, this.model, "joint", first)]] += lead * START_STRIDE;
      qpos[adr[idOf(this.mujoco, this.model, "joint", second)]] -= lead * START_STRIDE;
    }
  }

  // A fresh floor for this attempt, and the walker lifted above the ground near its start
  layGround() {
    this.attemptHeight = this.pickHeight();
    const layout = this.scene.layout;
    if (!layout) { this.heights = null; return; }
    this.heights = groundHeights(this.world.terrain, this.random, layout, this.attemptHeight);
    const most = Math.max(this.scene.floorCapacity, 1e-4);
    const field = this.model.hfield_data;
    for (let i = 0; i < this.heights.length; i++) field[i] = clamp(this.heights[i] / most, 0.0, 1.0);
    this.data.qpos[2] += highestNear(this.heights, layout, 0.0, 0.0, 0.3 * this.build.size);
  }

  // ---------- Senses ----------

  observation() {
    const out = new Float32Array(this.observationSize);
    if (!this.finite()) return out;
    const qpos = this.data.qpos, qvel = this.data.qvel;
    let i = 0;
    const put = (value) => { out[i++] = clamp(value, -OBSERVATION_LIMIT, OBSERVATION_LIMIT); };
    put(this.torsoHeight());
    put(qpos[1]);
    for (let k = 3; k < 7; k++) put(qpos[k]);
    for (let k = 0; k < 6; k++) put(qvel[k]);
    for (const adr of this.jointQpos) put(qpos[adr]);
    for (const adr of this.jointQvel) put(qvel[adr]);
    if (this.senses.feet) for (const touch of this.footContacts()) put(touch);
    if (this.senses.ground) for (const height of this.groundScan()) put(height);
    if (this.senses.turn) {
      const error = this.headingError();
      put(Math.sin(error)); put(Math.cos(error)); put(this.turnRate);
    }
    put(this.targetSpeed);
    return out;
  }

  // 1 for each foot touching the ground, else 0. Read once per step: copying contacts out of MuJoCo costs time
  footContacts() {
    if (this.contactCache && this.contactCache.step === this.stepCount && this.contactCache.time === this.data.time) {
      return this.contactCache.touching;
    }
    const touching = new Array(this.build.feet.length).fill(0);
    const count = this.data.ncon;
    if (count) {
      const contacts = this.data.contact;
      for (let c = 0; c < count; c++) {
        const contact = contacts.get(c);
        const g1 = contact.geom1, g2 = contact.geom2;
        contact.delete();
        const other = g1 === this.floor ? g2 : g2 === this.floor ? g1 : -1;
        if (other >= 0 && this.footOfGeom[other] >= 0) touching[this.footOfGeom[other]] = 1;
      }
      contacts.delete();
    }
    this.contactCache = { step: this.stepCount, time: this.data.time, touching };
    return touching;
  }

  // How far below the torso the ground is at points ahead of it
  groundScan() {
    const xpos = this.data.xpos, t = this.torso * 3;
    const [x, y, z] = [xpos[t], xpos[t + 1], xpos[t + 2]];
    return GROUND_SCAN.map((step) => z - this.groundAt(x + step * this.build.size, y));
  }

  info(rewardParts, final = false) {
    const finite = this.finite();
    const qpos = this.data.qpos;
    const info = { x_position: finite ? qpos[0] : 0.0, y_position: finite ? qpos[1] : 0.0,
      forward_distance: finite ? this.progress : 0.0, reward_parts: rewardParts, target_speed: this.targetSpeed,
      rock_hits: this.rockHits };
    if (final) for (const [name, value] of Object.entries(this.episodeParts)) info[`pts_${name}`] = value;
    return info;
  }

  // ---------- Moving ----------

  simulate() {
    const { mujoco, model, data } = this;
    let energy = 0.0;
    const timestep = model.opt.timestep;
    for (let k = 0; k < FRAME_SKIP; k++) {
      mujoco.mj_step(model, data);
      const force = data.actuator_force, velocity = data.actuator_velocity;
      let power = 0;
      for (let m = 0; m < force.length; m++) power += Math.abs(force[m] * velocity[m]);
      energy += power * timestep;
    }
    return energy;
  }

  // One control step. action: motor commands from -1 to 1. Returns { observation, reward, terminated, truncated, info }
  step(actionIn) {
    const action = Float64Array.from(actionIn, (value) => clamp(Number.isFinite(value) ? value : 0, -1.0, 1.0));
    const { data, model } = this;
    const xBefore = data.qpos[0], yBefore = data.qpos[1];
    const range = model.actuator_ctrlrange, ctrl = data.ctrl;
    for (let m = 0; m < action.length; m++) {
      const low = range[m * 2], high = range[m * 2 + 1];
      ctrl[m] = low + ((action[m] + 1.0) / 2.0) * (high - low);
    }
    this.extrasBeforeStep();
    const energy = this.simulate();
    this.lastEnergy = energy;
    this.extrasAfterStep();
    this.stepCount += 1;
    const terminated = this.fallen();
    const [reward, parts] = this.reward(xBefore, yBefore, energy, action, terminated);
    this.advanceHeading();
    for (const [name, value] of Object.entries(parts)) this.episodeParts[name] += value;
    const truncated = this.stepCount >= this.maxSteps;
    return { observation: this.observation(), reward, terminated, truncated, info: this.info(parts, terminated || truncated) };
  }

  fallen() {
    if (!this.finite()) return true;
    return this.torsoHeight() < this.fallHeight || this.torsoUp() < FALL_UP_Z;
  }

  // ---------- Points ----------

  reward(xBefore, yBefore, energy, action, fell) {
    if (!this.finite()) return [0.0, {}];
    const values = this.world.values;
    const qpos = this.data.qpos;
    // Speed is the distance moved along the asked heading (straight ahead, unless it is asked to turn)
    const moved = (qpos[0] - xBefore) * Math.cos(this.askedHeading) + (qpos[1] - yBefore) * Math.sin(this.askedHeading);
    this.progress += moved;
    const speed = moved / this.controlSeconds;
    const offCourse = this.world.turning || this.fixedTurnRate ? Math.abs(this.headingError()) : Math.abs(qpos[1]);
    const parts = {
      speed: values.speed_points * this.speedScore(speed),
      alive: values.alive_points,
      energy: -values.energy_cost * energy,
      straight: -values.straight_cost * offCourse,
    };
    if (fell && values.fall_cost) parts.fall = -values.fall_cost;
    if (values.end_stops && values.end_stop_cost) parts.end_stops = -values.end_stop_cost * this.endStopCloseness();
    if (values.level_body) parts.level = -values.level_cost * (1.0 - Math.max(this.torsoUp(), 0.0));
    if (values.no_crouch) parts.crouch = -values.crouch_cost * Math.max(0.0, 1.0 - this.torsoHeight() / this.scene.standHeight);
    if (values.smooth_moves) {
      let sum = 0;
      for (let m = 0; m < action.length; m++) sum += ((action[m] - this.previousAction[m]) / 2.0) ** 2;
      parts.smooth = -values.smooth_cost * (action.length ? sum / action.length : 0);
    }
    this.previousAction = Float64Array.from(action);
    if (values.lift_feet) parts.feet = values.lift_points * this.landings();
    if (values.face_forward) parts.heading = -values.heading_cost * Math.abs(this.headingError());
    if (values.steady_ride) parts.ride = -values.bounce_cost * this.data.qvel[2] ** 2;
    let total = 0;
    for (const value of Object.values(parts)) total += value;
    return [total, parts];
  }

  // The share of the speed points earned at this speed, by the world's rule
  speedScore(speed) {
    const target = this.targetSpeed;
    const shape = this.world.values.speed_shape;
    if (shape === "faster") return speed;
    if (shape === "bell") {
      const width = Math.max(0.25 * target, 0.05);
      return target * Math.exp(-(((speed - target) / width) ** 2));
    }
    return target - Math.abs(speed - target);
  }

  // "Lift its feet": for each foot that just landed, its time in the air beyond the goal (tenths of a second)
  landings() {
    const touching = this.footContacts();
    let earned = 0.0;
    if (this.targetSpeed > 0) {
      touching.forEach((touch, f) => {
        if (touch && this.airTime[f] > 0) earned += clamp(this.airTime[f] - AIR_TIME_GOAL, 0.0, 0.5);
      });
      earned *= 10;
    }
    this.airTime = Float64Array.from(this.airTime, (time, f) => (touching[f] ? 0.0 : time + this.controlSeconds));
    return earned;
  }

  // Each joint counts 0 until it enters the last 5% of its range, rising to 1 at the stop
  endStopCloseness() {
    const qpos = this.data.qpos;
    let total = 0;
    this.jointRanges.forEach(([low, high], j) => {
      const angle = qpos[this.jointQpos[j]];
      const room = Math.min(angle - low, high - angle) / (high - low);
      total += clamp((0.05 - room) / 0.05, 0.0, 1.0);
    });
    return total;
  }

  // ---------- Wind and rocks ----------

  extrasBeforeStep() {
    const values = this.world.values, now = this.data.time;
    if (values.wind) {
      if (now >= this.nextGust) {
        const angle = this.random.uniform(0, 2 * Math.PI);
        const strength = (values.wind_strength / 100.0) * this.scene.totalMass * values.gravity;
        this.gustForce = [strength * Math.cos(angle), strength * Math.sin(angle), 0.0];
        this.gustUntil = now + GUST_SECONDS;
        this.nextGust = now + values.wind_every * this.random.uniform(0.7, 1.3);
      }
      const force = now < this.gustUntil ? this.gustForce : [0, 0, 0];
      this.data.xfrc_applied.set(force, this.torso * 6);
    }
    if (this.rock && now >= this.nextRock) {
      this.throwRock();
      this.nextRock = now + values.rock_every;
    }
  }

  extrasAfterStep() {
    if (!this.rock || this.rockThrownAt === null) return;
    if (!this.rock.hit && this.rockTouchesWalker()) {
      this.rock.hit = true;
      this.rockHits += 1;
    }
    if (this.data.time - this.rockThrownAt >= ROCK_LIFETIME) this.parkRock();
  }

  rockParts() {
    const { mujoco, model } = this;
    const joint = idOf(mujoco, model, "joint", "rock");
    const body = idOf(mujoco, model, "body", "rock");
    const pos = model.body_pos;
    return { qpos: model.jnt_qposadr[joint], qvel: model.jnt_dofadr[joint], geom: idOf(mujoco, model, "geom", "rock"),
      parked: [pos[body * 3], pos[body * 3 + 1], pos[body * 3 + 2]], hit: false };
  }

  parkRock() {
    this.data.qpos.set([...this.rock.parked, 1, 0, 0, 0], this.rock.qpos);
    this.data.qvel.fill(0.0, this.rock.qvel, this.rock.qvel + 6);
    this.rockThrownAt = null;
  }

  // Throw the rock from one side, aiming where the walker will be and arcing for gravity
  throwRock() {
    const side = this.random.random() < 0.5 ? 1.0 : -1.0;
    const distance = THROW_DISTANCE * this.build.size;
    const flight = distance / 4.0;
    const com = this.data.subtree_com, qvel = this.data.qvel, t = this.torso * 3;
    const target = [0, 1, 2].map((k) => com[t + k] + qvel[k] * flight);
    const start = [target[0], target[1] + side * distance, target[2]];
    const velocity = target.map((value, k) => (value - start[k]) / flight);
    velocity[2] += 0.5 * this.world.values.gravity * flight;
    this.data.qpos.set([...start, 1, 0, 0, 0], this.rock.qpos);
    this.data.qvel.set([...velocity, 0, 0, 0], this.rock.qvel);
    this.rockThrownAt = this.data.time;
    this.rock.hit = false;
  }

  rockTouchesWalker() {
    const count = this.data.ncon;
    if (!count) return false;
    const contacts = this.data.contact;
    let touched = false;
    for (let c = 0; c < count && !touched; c++) {
      const contact = contacts.get(c);
      const g1 = contact.geom1, g2 = contact.geom2;
      contact.delete();
      if ((g1 === this.rock.geom && this.walkerGeoms.has(g2)) || (g2 === this.rock.geom && this.walkerGeoms.has(g1))) touched = true;
    }
    contacts.delete();
    return touched;
  }
}
