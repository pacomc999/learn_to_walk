// A training session: the walker practises, takes exams and keeps its best brain. A port of studio/train.py.
// It runs inside a background worker (trainer.worker.js). Everything the page shows is written through `store`:
//   status, progress rows, exam rows, replays (best and one per exam), brains (latest and best) and a log.
// Practising happens in a pool of practice groups (see practice.js), possibly in further workers.

import { buildCreature } from "./creatures/index.js";
import { cleanDesign, designForBody, designSenses, mirrorTables, startLogStd } from "./design.js";
import { WalkEnv, PART_NAMES, observationSize } from "./env.js";
import { cleanValues } from "./kit.js";
import { learningGroups } from "./catalog.js";
import { Learner, PPO_FIXED } from "./learner.js";
import { cellReader, deterministicActor } from "./network.js";
import { adjustedLearningRate, advantagesAndReturns, fittedBatchSize, normalizeRewards, rewardState, RunningMeanStd, scalingOf } from "./ppo.js";
import { Random } from "./random.js";
import { record } from "./replay.js";
import { buildScene, disposeScene } from "./scene.js";
import { worldFromDict } from "./worlds.js";
import { now } from "./library.js";

export const EXAM_ATTEMPTS = 8;
const EXAM_FIRST_SEED = 1000;
const RAMP_SHARE = 0.5;            // gradually harder worlds reach full height over this share of a stage
const MOST_EXAM_REPLAYS = 30;
const SAVE_EVERY = 250_000;

export const PROGRESS_HEADER = ["time", "steps", "score", "distance", "seconds", "fall_share", "steps_per_second", "difficulty", "stage", ...PART_NAMES.map((n) => `pts_${n}`)];
export const EXAM_HEADER = ["time", "steps", "score", "falls", "attempts", "distance", "quality", "speed_error", "stage", "joules_per_metre"];

export function cleanLearning(values) {
  return cleanValues(learningGroups(), values);
}

// ---------- Courses ----------

// The stages of a session: its own world first, then any course stages, one after the other
export function courseStages(setup, learning) {
  const parts = [[setup.world, learning.million_steps], ...(setup.course || []).map((stage) => [stage.world, Number(stage.million_steps)])];
  let start = 0;
  return parts.map(([world, million], index) => {
    const end = start + Math.trunc(million * 1_000_000);
    const stage = { index, count: parts.length, world, start, end };
    start = end;
    return stage;
  });
}

// ---------- The exam ----------

// The asked speeds and ground heights of the exam: spread evenly over the world's ranges
export function examPlan(world, attempts = EXAM_ATTEMPTS) {
  const [low, high] = world.speedRange;
  let speeds = Array.from({ length: attempts }, (_, i) => low + ((high - low) * i) / Math.max(attempts - 1, 1));
  speeds = speeds.map((_, i) => speeds[(i * 3) % attempts]);  // mix speeds and heights
  const heights = world.values.mixed_heights
    ? Array.from({ length: attempts }, (_, i) => (world.tallest * (i + 1)) / attempts)
    : Array.from({ length: attempts }, () => world.tallest);
  return speeds.map((speed, i) => [speed, heights[i]]);
}

export function takeExam(mujoco, brain, build, world, senses, scene) {
  const act = deterministicActor(brain);
  const plan = examPlan(world);
  const scores = [], distances = [], qualities = [], speedErrors = [], energies = [];
  let falls = 0;
  plan.forEach(([speed, height], index) => {
    let turn = null;
    if (world.turning) {  // turns spread from sharp left to sharp right, mixed with the speeds
      const most = world.values.turn_rate;
      turn = most * (-1 + (2 * ((index * 5) % plan.length)) / Math.max(plan.length - 1, 1));
    }
    const env = new WalkEnv(mujoco, build, world, { senses, targetSpeed: speed, terrainHeight: height, scene, turnRate: turn });
    try {
      let observation = env.reset(EXAM_FIRST_SEED + index);
      let total = 0, fell = false, steps = 0, joules = 0, info = {};
      for (;;) {
        const result = env.step(act(observation));
        observation = result.observation;
        total += result.reward;
        steps += 1;
        joules += env.lastEnergy;
        info = result.info;
        fell = result.terminated;
        if (result.terminated || result.truncated) break;
      }
      const seconds = steps * env.controlSeconds;
      const distance = info.forward_distance;
      const asked = speed * world.values.attempt_seconds;
      scores.push(total);
      falls += fell ? 1 : 0;
      distances.push(distance);
      qualities.push(Math.min(Math.max(distance / asked, 0.0), 1.0));
      speedErrors.push(Math.abs(distance / Math.max(seconds, 1e-9) - speed));
      if (distance > 0.5) energies.push(joules / distance);
    } finally {
      env.dispose();
    }
  });
  const average = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  return { score: average(scores), falls, attempts: scores.length, distance: average(distances), quality: average(qualities) * 100,
    speed_error: average(speedErrors), joules_per_metre: energies.length ? average(energies) : null };
}

// A replay of the brain walking at the middle speed on fairly tall ground
export function bestWalkReplay(mujoco, brain, build, world, senses, scene) {
  const speed = (world.speedRange[0] + world.speedRange[1]) / 2;
  const height = world.values.mixed_heights ? world.tallest * 0.75 : world.tallest;
  const turn = world.turning ? world.values.turn_rate / 2 : null;  // a turning walker shows a gentle turn
  const env = new WalkEnv(mujoco, build, world, { senses, targetSpeed: speed, terrainHeight: height, scene, turnRate: turn });
  try {
    return record(env, { act: deterministicActor(brain), seed: EXAM_FIRST_SEED, brain: cellReader(brain) });
  } finally {
    env.dispose();
  }
}

// ---------- The session ----------

const round = (value, places) => Math.round(value * 10 ** places) / 10 ** places;
const mean = (values) => values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);

// tf: TensorFlow.js; makePool(stage world, seed) -> a practice pool; store: where everything is written
export async function runTraining({ tf, mujoco, store, makePool, resume, shouldStop, onProgress = () => {} }) {
  const setup = await store.readSetup();
  let design = cleanDesign(setup.brain);
  const learning = cleanLearning(setup.learning);
  const senses = designSenses(design);
  const stages = courseStages(setup, learning);
  const totalSteps = stages[stages.length - 1].end;
  const previous = (await store.readStatus()) || {};
  const status = {
    state: "starting", steps: previous.steps || 0, total_steps: totalSteps, steps_per_second: 0,
    best_score: previous.best_score ?? null, best_steps: previous.best_steps ?? null, best_quality: previous.best_quality ?? null,
    last_exam: previous.last_exam ?? null, difficulty: previous.difficulty ?? 1.0, message: "Building the practice worlds",
    started: now(), exams_without_best: previous.exams_without_best || 0,
  };
  const update = async (values) => { Object.assign(status, values, { updated: now() }); await store.writeStatus(status); onProgress(status); };
  await update({});

  const build = buildCreature(setup.creature.animal, setup.creature.values);
  design = designForBody(design, build);
  const symmetric = design.values.symmetric;
  const mirror = symmetric ? mirrorTables(build, senses) : null;
  const worlds = Math.trunc(learning.parallel_worlds);
  const seed = Math.trunc(learning.seed);
  const settings = {
    steps: Math.trunc(learning.steps_per_update), batchSize: fittedBatchSize(worlds, Math.trunc(learning.steps_per_update), Math.trunc(learning.batch_size)),
    epochs: Math.trunc(learning.epochs), gamma: Number(learning.gamma), entropyWeight: Number(learning.curiosity),
  };
  const inputSize = observationSize(build, senses);
  const learner = new Learner(tf, { kind: symmetric ? "symmetric" : "free", activation: design.values.activation, inputSize,
    motorCount: build.motors.length, actorLayers: design.layers, criticLayers: design.judge_layers, mirror, seed,
    logStd: startLogStd(design, build.motorType), learningRate: Number(learning.learning_rate) });
  const rmsOptions = mirror ? { order: mirror.observation_order, signs: mirror.observation_signs } : {};
  let obsRms = new RunningMeanStd(inputSize, rmsOptions);
  let rewards = rewardState(worlds);
  let steps = 0;
  let lastKl = null;

  // Where to begin: a paused session's latest brain, another session's best brain, or a fresh one
  const latest = resume ? await store.loadBrain("latest") : null;
  if (latest) {
    learner.loadBrain(latest.brain);
    await learner.loadOptimizerState(latest.optimizer);
    learner.setLearningRate(latest.learning_rate);
    obsRms = RunningMeanStd.fromData(latest.obs_rms, rmsOptions);
    rewards = rewardState(worlds, latest.ret_rms);
    steps = latest.steps;
    await store.log(`Carrying on from ${steps.toLocaleString("en")} steps.`);
  } else if (setup.start_from && store.loadOtherBrain) {
    const source = await store.loadOtherBrain(setup.start_from);
    if (!source) throw new Error("The brain to start from was not found.");
    learner.loadBrain(source.brain);
    if (source.learning_rate) learner.setLearningRate(source.learning_rate);  // a trained brain keeps its small step size
    if (source.obs_rms) obsRms = RunningMeanStd.fromData(source.obs_rms, rmsOptions);
    if (source.ret_rms) rewards = rewardState(worlds, source.ret_rms);
    await store.log(`Starting from the best brain of ${setup.start_from}.`);
  } else {
    const count = learner.variables().reduce((sum, v) => sum + v.size, 0);
    await store.log(`A new brain with ${count.toLocaleString("en")} numbers to learn.`);
  }
  status.steps = steps;

  const brainNow = () => ({ ...learner.toBrain(), scaling: scalingOf(obsRms) });
  const saveBrain = async (which) => {
    const data = { brain: brainNow(), obs_rms: obsRms.toData(), ret_rms: rewards.retRms.toData(), learning_rate: learner.learningRate, steps };
    if (which === "latest") data.optimizer = await learner.optimizerState();
    await store.saveBrain(which, data);
  };

  let stopped = false, outOfPatience = false;
  let lessonsDone = 0;
  let current = stages.find((stage) => stage.end > steps) || stages[stages.length - 1];
  let examTools = null;
  try {
    for (const stage of stages) {
      if (stage.end <= steps && stage.index < stages.length - 1) continue;  // already done
      if (stage.index !== current.index) {  // a new stage: new practice worlds, the same brain and scaling
        await store.log(`Stage ${stage.index + 1} of ${stages.length}: ${worldFromDict(stage.world).name}.`);
        Object.assign(status, { best_score: null, best_steps: null, best_quality: null, exams_without_best: 0 });
        current = stage;
      }
      const world = worldFromDict(stage.world);
      if (examTools) disposeScene(examTools.scene);
      examTools = { build, world, scene: buildScene(mujoco, build, world) };
      const pool = await makePool(stage.world, seed + stage.index, senses);
      try {
        const difficulty = () => {
          if (!world.values.gradually_harder) return 1.0;
          const done = steps - stage.start;
          return Math.min(1.0, Math.max(done, 0) / Math.max(RAMP_SHARE * (stage.end - stage.start), 1));
        };
        const examine = async () => {
          await update({ state: "examining", message: "Taking an exam", steps });
          const brain = brainNow();
          const result = takeExam(mujoco, brain, build, world, senses, examTools.scene);
          const row = { time: now(), steps, stage: stage.index };
          for (const [key, value] of Object.entries(result)) row[key] = typeof value === "number" && !Number.isInteger(value) ? round(value, 3) : value;
          if (row.joules_per_metre === null) row.joules_per_metre = "";
          await store.appendExam(row);
          const replay = bestWalkReplay(mujoco, brain, build, world, senses, examTools.scene);
          replay.steps = steps;
          replay.exam = row;
          await store.writeReplay(`exam_${String(steps).padStart(12, "0")}`, replay);
          await thinReplays(store);
          const best = status.best_score;
          if (best === null || best === undefined || result.score > best) {
            await saveBrain("best");
            await store.writeReplay("best", replay);
            Object.assign(status, { best_score: result.score, best_steps: steps, best_quality: result.quality, exams_without_best: 0 });
            await store.log(`New best at ${steps.toLocaleString("en")} steps: exam score ${Math.round(result.score)}, ${result.falls} falls, ${Math.round(result.quality)}% of the asked distance.`);
          } else {
            status.exams_without_best = (status.exams_without_best || 0) + 1;
            const patience = Math.trunc(learning.patience);
            if (patience && status.exams_without_best >= patience) {
              outOfPatience = true;
              await store.log(`No new best for ${patience} exams in a row: stopping early.`);
            }
          }
          await update({ state: "training", message: "Practising", last_exam: row });
        };

        const examEvery = Math.trunc(learning.exam_every) * 1000;
        let nextExam = steps;  // an exam at the start shows where it began
        let nextSave = steps + SAVE_EVERY;
        await update({ state: "training", message: "Practising", stage: stage.index, stages: stage.count, stage_world: world.name });
        const finishedAttempts = [];
        let clock = performance.now(), clockSteps = steps;
        while (steps < stage.end && !outOfPatience) {
          if (await shouldStop()) { stopped = true; break; }
          // Before practising (on_rollout_start): learning speed, difficulty, exam, save
          if (learning.adaptive) learner.setLearningRate(adjustedLearningRate(learner.learningRate, lastKl));
          await pool.setDifficulty(difficulty());
          if (steps >= nextExam) {
            await examine();
            nextExam = (Math.floor(steps / examEvery) + 1) * examEvery;
            if (outOfPatience) break;
          }
          if (steps >= nextSave) { await saveBrain("latest"); nextSave = steps + SAVE_EVERY; }

          // Practise
          const practiceStart = performance.now();
          const rollout = await pool.rollout(brainNow(), settings.steps, { judge: false });
          const practiceSeconds = (performance.now() - practiceStart) / 1000;
          fillJudgeValues(learner, rollout, inputSize);
          steps += settings.steps * worlds;
          finishedAttempts.push(...rollout.finished);
          if (finishedAttempts.length > 100) finishedAttempts.splice(0, finishedAttempts.length - 100);  // ep_info_buffer

          // After practising (on_rollout_end): the progress row
          const timeNow = performance.now();
          const speed = (steps - clockSteps) / Math.max((timeNow - clock) / 1000, 1e-9);
          clock = timeNow; clockSteps = steps;
          const row = { time: now(), steps, steps_per_second: Math.round(speed), difficulty: round(difficulty(), 3), stage: stage.index };
          if (finishedAttempts.length) {
            const maxSteps = Math.round(world.values.attempt_seconds / 0.02);
            Object.assign(row, { score: round(mean(finishedAttempts.map((e) => e.r)), 2), distance: round(mean(finishedAttempts.map((e) => e.forward_distance)), 3),
              seconds: round(mean(finishedAttempts.map((e) => e.l)) * 0.02, 2), fall_share: round(mean(finishedAttempts.map((e) => (e.l < maxSteps ? 1 : 0))), 3) });
            for (const name of PART_NAMES) row[`pts_${name}`] = round(mean(finishedAttempts.map((e) => e[`pts_${name}`] || 0)), 2);
          }
          await store.appendProgress(row);
          await update({ steps, steps_per_second: Math.round(speed), difficulty: row.difficulty, state: "training", message: "Practising" });

          // Study (train())
          const batch = studyBatch(rollout, rewards, settings);
          obsRms.update(rollout.rawObservations.flatMap((flat) => splitFlat(flat, inputSize)));
          const studyStart = performance.now();
          const stats = learner.update(batch, { epochs: settings.epochs, batchSize: settings.batchSize, entropyWeight: settings.entropyWeight, learningRate: learner.learningRate }, new Random(seed + steps));
          lastKl = stats.approx_kl;
          lessonsDone += 1;
          if (lessonsDone <= 3) {
            await store.log(`Lesson ${lessonsDone}: practice ${practiceSeconds.toFixed(1)} s, study ${((performance.now() - studyStart) / 1000).toFixed(1)} s.`);
          }
        }
      } finally {
        await pool.dispose();
      }
      if (stopped || outOfPatience) break;
    }
    await saveBrain("latest");
    if (!stopped) {
      const world = examTools.world;
      await update({ state: "examining", message: "Taking the final exam", steps });
      const brain = brainNow();
      const result = takeExam(mujoco, brain, build, world, senses, examTools.scene);
      const row = { time: now(), steps, stage: current.index, ...Object.fromEntries(Object.entries(result).map(([k, v]) => [k, typeof v === "number" && !Number.isInteger(v) ? round(v, 3) : v])) };
      if (row.joules_per_metre === null) row.joules_per_metre = "";
      await store.appendExam(row);
      if (status.best_score === null || status.best_score === undefined || result.score > status.best_score) {
        await saveBrain("best");
        const replay = bestWalkReplay(mujoco, brain, build, world, senses, examTools.scene);
        replay.steps = steps;
        replay.exam = row;
        await store.writeReplay("best", replay);
        Object.assign(status, { best_score: result.score, best_steps: steps, best_quality: result.quality });
      }
      status.last_exam = row;
    }
    const [state, message] = stopped ? ["stopped", "Paused. Press Continue to carry on from here."]
      : outOfPatience ? ["finished", "Finished early: it had stopped getting better."]
        : ["finished", `Finished all ${totalSteps.toLocaleString("en")} practice steps.`];
    await update({ state, message, steps });
    await store.log(message);
  } finally {
    if (examTools) disposeScene(examTools.scene);
    learner.dispose();
  }
}

// The judge's values of a lesson, worked out together on the graphics card: every step, every attempt that ran
// out of time, and the last position of every world (what the practice workers left out)
function fillJudgeValues(learner, rollout, inputSize) {
  const steps = rollout.steps, worlds = rollout.worlds;
  const extra = [...rollout.terminalObservations.map((item) => item.scaled), ...rollout.lastObservations];
  const all = new Float32Array((steps * worlds + extra.length) * inputSize);
  rollout.observations.forEach((flat, t) => all.set(flat, t * worlds * inputSize));
  extra.forEach((scaled, i) => all.set(scaled, (steps * worlds + i) * inputSize));
  const values = learner.judgeValues(all, steps * worlds + extra.length);
  rollout.values = Array.from({ length: steps }, (_, t) => Array.from(values.subarray(t * worlds, (t + 1) * worlds)));
  let next = steps * worlds;
  for (const { t, e } of rollout.terminalObservations) rollout.terminalValues[t][e] = values[next++];
  rollout.lastValues = rollout.lastObservations.map(() => values[next++]);
}

function splitFlat(flat, size) {
  const rows = [];
  for (let start = 0; start < flat.length; start += size) rows.push(flat.subarray(start, start + size));
  return rows;
}

// The practice of one lesson as the learner wants it: scaled points with time-out credit, GAE, flattened
function studyBatch(rollout, rewards, settings) {
  const scaled = normalizeRewards(rollout.rewards, rollout.dones, rewards, settings.gamma);
  rollout.terminalValues.forEach((row, t) => row.forEach((value, e) => { if (value) scaled[t][e] += settings.gamma * value; }));
  const { advantages, returns } = advantagesAndReturns({ rewards: scaled, values: rollout.values, episodeStarts: rollout.episodeStarts,
    lastValues: rollout.lastValues, lastDones: rollout.lastDones, gamma: settings.gamma, lambda: PPO_FIXED.gaeLambda });
  const steps = rollout.rewards.length, worlds = rollout.worlds;
  const count = steps * worlds;
  const obsSize = rollout.observations[0].length / worlds, actSize = rollout.actions[0].length / worlds;
  const batch = { count, observations: new Float32Array(count * obsSize), actions: new Float32Array(count * actSize),
    logProbs: new Float32Array(count), advantages: new Float32Array(count), returns: new Float32Array(count) };
  let i = 0;
  for (let t = 0; t < steps; t++) {
    for (let e = 0; e < worlds; e++, i++) {
      batch.observations.set(rollout.observations[t].subarray(e * obsSize, (e + 1) * obsSize), i * obsSize);
      batch.actions.set(rollout.actions[t].subarray(e * actSize, (e + 1) * actSize), i * actSize);
      batch.logProbs[i] = rollout.logProbs[t][e];
      batch.advantages[i] = advantages[t][e];
      batch.returns[i] = returns[t][e];
    }
  }
  return batch;
}

// Keep at most MOST_EXAM_REPLAYS exam replays, spread evenly over the session
async function thinReplays(store) {
  const names = (await store.listReplays()).filter((name) => name.startsWith("exam_")).sort();
  while (names.length > MOST_EXAM_REPLAYS) {
    // drop the one whose neighbours are closest together, but never the first or the last
    let best = 1, smallest = Infinity;
    for (let i = 1; i < names.length - 1; i++) {
      const gap = Number(names[i + 1].slice(5)) - Number(names[i - 1].slice(5));
      if (gap < smallest) { smallest = gap; best = i; }
    }
    await store.deleteReplay(names[best]);
    names.splice(best, 1);
  }
}

// ---------- Practice pools ----------

// Merge the practice of several groups (each [step][its worlds]) into one [step][all worlds]
export function mergeRollouts(parts) {
  if (parts.length === 1) return parts[0];
  const steps = parts[0].steps;
  const joinFlat = (key, t) => {
    const total = parts.reduce((sum, p) => sum + p[key][t].length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    for (const p of parts) { out.set(p[key][t], offset); offset += p[key][t].length; }
    return out;
  };
  const joinRows = (key, t) => parts.flatMap((p) => p[key][t]);
  const offsets = parts.map((_, i) => parts.slice(0, i).reduce((sum, p) => sum + p.worlds, 0));
  const merged = { worlds: parts.reduce((sum, p) => sum + p.worlds, 0), steps, finished: parts.flatMap((p) => p.finished),
    lastValues: parts.flatMap((p) => p.lastValues), lastDones: parts.flatMap((p) => p.lastDones),
    lastObservations: parts.flatMap((p) => p.lastObservations),
    terminalObservations: parts.flatMap((p, i) => p.terminalObservations.map((item) => ({ ...item, e: item.e + offsets[i] }))) };
  for (const key of ["observations", "rawObservations", "actions"]) merged[key] = Array.from({ length: steps }, (_, t) => joinFlat(key, t));
  for (const key of ["logProbs", "values", "rewards", "dones", "episodeStarts", "terminalValues"]) merged[key] = Array.from({ length: steps }, (_, t) => joinRows(key, t));
  return merged;
}
