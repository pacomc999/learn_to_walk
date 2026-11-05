// A tiny real training session in Node: practice, lessons, exams, the best brain, pausing and carrying on.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadTensorflowForTests } from "./tf_node.mjs";
import { useCatalog, getCatalog } from "../../studio/web/js/engine/catalog.js";
import { loadPhysics } from "../../studio/web/js/engine/physics.js";
import { InlinePool } from "../../studio/web/js/engine/pools.js";
import { runTraining, examPlan } from "../../studio/web/js/engine/trainer.js";
import { makeWorld } from "../../studio/web/js/engine/worlds.js";

useCatalog(JSON.parse(readFileSync(new URL("../../studio/web/data/catalog.json", import.meta.url), "utf8")));
const tf = await loadTensorflowForTests();
const mujoco = await loadPhysics();

// Everything a training writes, kept in memory
function memoryStore(setup) {
  const files = { status: null, progress: [], exams: [], replays: {}, brains: {}, log: [] };
  return {
    files,
    readSetup: async () => setup,
    readStatus: async () => files.status,
    writeStatus: async (status) => { files.status = structuredClone(status); },
    appendProgress: async (row) => { files.progress.push(row); },
    appendExam: async (row) => { files.exams.push(row); },
    writeReplay: async (name, replay) => { files.replays[name] = replay; },
    listReplays: async () => Object.keys(files.replays),
    deleteReplay: async (name) => { delete files.replays[name]; },
    saveBrain: async (which, data) => { files.brains[which] = structuredClone(data); },
    loadBrain: async (which) => files.brains[which] || null,
    log: async (line) => { files.log.push(line); },
  };
}

function tinySetup(stepsMillion = 0.002) {
  const table = getCatalog().animals.find((a) => a.key === "custom").templates.find((t) => t.key === "table");
  return {
    id: "test", name: "Test", creature: { animal: "custom", values: { body: { torso: table.torso, parts: table.parts } } },
    world: { id: "flat", name: "Flat", values: { attempt_seconds: 5 } }, brain: { layers: [32, 32] },
    learning: { million_steps: stepsMillion, parallel_worlds: 2, steps_per_update: 256, batch_size: 128, epochs: 2, exam_every: 50, patience: 0 },
  };
}

const makePool = (setup) => async (world, seed, senses) =>
  new InlinePool(mujoco, { animal: setup.creature.animal, values: setup.creature.values, world, senses, count: setup.learning.parallel_worlds, seed });

test("the exam plan mixes the speeds and spreads the heights", () => {
  const plan = examPlan(makeWorld({ values: { slowest_speed: 0.2, fastest_speed: 0.9, terrain: "cobblestones", terrain_height: 4, mixed_heights: true } }));
  assert.equal(plan.length, 8);
  assert.deepEqual(plan.map(([s]) => Math.round(s * 100)).sort((a, b) => a - b), [20, 30, 40, 50, 60, 70, 80, 90]);
  assert.ok(Math.abs(plan[7][1] - 0.04) < 1e-12);
});

// Stop after this many lessons (the shortest real session is 0.1 million steps, too long for a test)
const stopAfter = (lessons) => { let asked = 0; return async () => ++asked > lessons; };

test("a training session practises, studies, takes an exam, keeps its brains, pauses and carries on", async () => {
  const setup = tinySetup();
  const store = memoryStore(setup);
  await runTraining({ tf, mujoco, store, makePool: makePool(setup), resume: false, shouldStop: stopAfter(4) });
  const { status, progress, exams, replays, brains } = store.files;
  assert.equal(status.state, "stopped", status.message);
  assert.equal(status.steps, 2048);  // 4 lessons of 2 worlds x 256 steps
  assert.equal(progress.length, 4);
  assert.ok(progress.every((row) => Number.isFinite(row.steps_per_second)));
  assert.equal(exams.length, 1, "the exam at the start (a paused session skips the final exam)");
  assert.ok(replays.best && replays.best.frames.length > 1 && replays.best.cells.length > 1);
  assert.equal(brains.latest.steps, 2048);
  assert.ok(brains.best && brains.latest.optimizer.length > 0);

  await runTraining({ tf, mujoco, store, makePool: makePool(setup), resume: true, shouldStop: stopAfter(2) });
  assert.equal(store.files.status.state, "stopped");
  assert.ok(store.files.log.some((line) => line.startsWith("Carrying on from 2,048 steps")));
  assert.equal(store.files.status.steps, 3072);
  assert.equal(store.files.progress.length, 6);
});
