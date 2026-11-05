// A group of practice worlds that practise together with one copy of the brain (Stable-Baselines3's
// collect_rollouts, for the worlds of one worker). Each world keeps its attempt going between lessons.

import { buildCreature } from "./creatures/index.js";
import { WalkEnv, PART_NAMES } from "./env.js";
import { actionStd, judgeValue, sampleAction, scaleObservation } from "./network.js";
import { Random } from "./random.js";
import { buildScene, disposeScene } from "./scene.js";
import { worldFromDict } from "./worlds.js";

const LOG_TWO_PI = Math.log(2 * Math.PI);

export class PracticeGroup {
  // count worlds of one creature in one world; seed: the first world's seed (each world gets seed + its number)
  constructor(mujoco, { animal, values, world, senses, count, seed }) {
    this.mujoco = mujoco;
    this.build = buildCreature(animal, values);
    this.world = worldFromDict(world);
    // On flat ground the worlds can share one physics model. On bumpy ground each world lays its own floor into the
    // model's height field, so each needs its own model (sharing one let the worlds overwrite each other's ground)
    const shareScene = this.world.terrain === "flat";
    this.scene = shareScene ? buildScene(mujoco, this.build, this.world) : null;
    this.envs = Array.from({ length: count }, () => new WalkEnv(mujoco, this.build, this.world, { senses, scene: this.scene }));
    this.random = new Random(seed * 7919 + 13);
    this.observations = this.envs.map((env, i) => env.reset(seed + i));  // VecEnv.seed(seed) then reset()
    this.episodeStarts = this.envs.map(() => 1);
    this.attempt = this.envs.map(() => ({ reward: 0, length: 0 }));
  }

  setDifficulty(difficulty) {
    for (const env of this.envs) env.setDifficulty(difficulty);
  }

  // Practise `steps` steps in every world with this brain and sense scaling. Everything comes back as plain
  // arrays, [step][world] (and for vectors one flat Float32Array per step, worlds one after the other).
  // judge: false leaves the judge's values out (the learner works them out for the whole lesson at once, on the
  // graphics card); the scaled senses it needs come back in terminalObservations and lastObservations
  rollout(brain, steps, { judge = true } = {}) {
    const worlds = this.envs.length;
    const std = actionStd(brain);
    const logStdSum = std.reduce((sum, s) => sum + Math.log(s), 0);
    const motorCount = std.length;
    const out = { worlds, steps, observations: [], rawObservations: [], actions: [], logProbs: [], values: [], rewards: [],
      dones: [], episodeStarts: [], terminalValues: [], terminalObservations: [], finished: [] };
    for (let t = 0; t < steps; t++) {
      const obsRow = [], rawRow = [], actionRow = [], logRow = [], valueRow = [], rewardRow = [], doneRow = [], terminalRow = [];
      out.episodeStarts.push([...this.episodeStarts]);
      this.envs.forEach((env, e) => {
        const raw = this.observations[e];
        const scaled = scaleObservation(brain, raw);
        const { action, mean } = sampleAction(brain, scaled, this.random, std);
        let squares = 0;
        for (let m = 0; m < motorCount; m++) squares += ((action[m] - mean[m]) / std[m]) ** 2;
        const logProb = -0.5 * squares - logStdSum - 0.5 * motorCount * LOG_TWO_PI;
        const value = judge ? judgeValue(brain, scaled) : 0;
        const result = env.step(action);  // the world clips the action to -1..1, the brain remembers the raw try
        const record = this.attempt[e];
        record.reward += result.reward;
        record.length += 1;
        let terminal = 0;
        let next = result.observation;
        const done = result.terminated || result.truncated;
        if (done) {
          if (result.truncated && !result.terminated) {  // time ran out: count on the points still to come
            const scaledEnd = scaleObservation(brain, result.observation);
            if (judge) terminal = judgeValue(brain, scaledEnd);
            else out.terminalObservations.push({ t, e, scaled: scaledEnd });
          }
          out.finished.push(attemptSummary(record, result.info));
          this.attempt[e] = { reward: 0, length: 0 };
          next = env.reset();
        }
        obsRow.push(scaled); rawRow.push(raw); actionRow.push(action); logRow.push(logProb); valueRow.push(value);
        rewardRow.push(result.reward); doneRow.push(done ? 1 : 0); terminalRow.push(terminal);
        this.observations[e] = next;
        this.episodeStarts[e] = done ? 1 : 0;
      });
      out.observations.push(flatten(obsRow)); out.rawObservations.push(flatten(rawRow)); out.actions.push(flatten(actionRow));
      out.logProbs.push(logRow); out.values.push(valueRow); out.rewards.push(rewardRow); out.dones.push(doneRow);
      out.terminalValues.push(terminalRow);
    }
    out.lastObservations = this.observations.map((raw) => scaleObservation(brain, raw));
    out.lastValues = judge ? out.lastObservations.map((scaled) => judgeValue(brain, scaled)) : out.lastObservations.map(() => 0);
    out.lastDones = [...this.episodeStarts];
    return out;
  }

  dispose() {
    for (const env of this.envs) env.dispose();
    disposeScene(this.scene);
  }
}

// One finished attempt, as Stable-Baselines3's Monitor reports it (with the training's extra keys)
function attemptSummary(record, info) {
  const summary = { r: record.reward, l: record.length, forward_distance: info.forward_distance ?? 0 };
  for (const name of PART_NAMES) summary[`pts_${name}`] = info[`pts_${name}`] ?? 0;
  return summary;
}

function flatten(rows) {
  const size = rows[0] ? rows[0].length : 0;
  const out = new Float32Array(rows.length * size);
  rows.forEach((row, i) => out.set(row, i * size));
  return out;
}
