// The trainer worker: runs one training session in the background, so the page stays smooth.
// It learns with TensorFlow.js (on the graphics card when it can), practises in further workers (one per core),
// and writes everything into the browser's storage, where the pages read it.
// Messages in: start { catalog, trainingId, resume, cores }, stop. Messages out: status, done, error.
// While it runs it holds the Web Lock "training-<id>", so any tab can tell the session is alive.

import * as tf from "../../vendor/tfjs/tf.fesm.min.js";
import { useCatalog } from "./catalog.js";
import { startTensorflow } from "./learner.js";
import { loadPhysics } from "./physics.js";
import { InlinePool, WorkerPool } from "./pools.js";
import { deleteRecord, getRecord, listFiles, putRecord, readFile, writeFile } from "./storage.js";
import { runTraining } from "./trainer.js";

const MOST_LOG_LINES = 300;
let stopAsked = false;

self.onmessage = async (event) => {
  if (event.data.type === "stop") { stopAsked = true; return; }
  if (event.data.type !== "start") return;
  const { catalog, trainingId, resume, cores } = event.data;
  useCatalog(catalog);
  try {
    await navigator.locks.request(`training-${trainingId}`, { ifAvailable: true }, async (lock) => {
      if (!lock) throw new Error("This training is already running in another tab.");
      await train(catalog, trainingId, resume, cores);
    });
    self.postMessage({ type: "done" });
  } catch (error) {
    await failed(trainingId, error);
    self.postMessage({ type: "error", message: String(error?.message || error) });
  }
};

async function train(catalog, trainingId, resume, cores) {
  const store = indexedDbStore(trainingId);
  await deleteRecord("files", `${trainingId}/stop`);
  const backend = await startTensorflow(tf);
  await store.log(`Learning runs on ${backend === "cpu" ? "the processor" : `the graphics card (${backend === "webgpu" ? "WebGPU" : "WebGL"})`}.`);
  const mujoco = await loadPhysics();
  const canNest = typeof Worker !== "undefined";
  const practiceWorkers = Math.max(1, (cores || 4) - 1);  // one core stays free for the learner and the page
  const makePool = async (world, seed, senses) => {
    const setup = await store.readSetup();
    const spec = { animal: setup.creature.animal, values: setup.creature.values, world, senses, count: Math.trunc(setup.learning.parallel_worlds), seed };
    if (canNest) {
      try {
        return await WorkerPool.start(spec, catalog, practiceWorkers);
      } catch (error) {
        await store.log(`Practising in one thread (${error.message}).`);
      }
    }
    return new InlinePool(mujoco, spec);
  };
  const shouldStop = async () => {
    if (stopAsked) return true;
    return Boolean(await getRecord("files", `${trainingId}/stop`));  // a stop asked from another tab
  };
  await runTraining({ tf, mujoco, store, makePool, resume, shouldStop, onProgress: (status) => self.postMessage({ type: "status", status }) });
}

// A session that crashed says so on its page
async function failed(trainingId, error) {
  try {
    const record = await getRecord("trainings", trainingId);
    if (!record) return;
    record.status = { ...(record.status || {}), state: "error", message: `Something went wrong: ${error?.message || error}`, updated: new Date().toISOString() };
    await putRecord("trainings", record);
    await appendLog(trainingId, String(error?.stack || error));
  } catch { /* nothing more can be done */ }
}

async function appendLog(trainingId, line) {
  const lines = (await readFile(trainingId, "log", [])) || [];
  lines.push(line);
  await writeFile(trainingId, "log", lines.slice(-MOST_LOG_LINES));
}

// ---------- Where a training writes ----------

function indexedDbStore(trainingId) {
  const appendRow = async (name, row) => {
    const rows = (await readFile(trainingId, name, [])) || [];
    rows.push(row);
    await writeFile(trainingId, name, rows);
  };
  return {
    readSetup: async () => (await getRecord("trainings", trainingId)).setup,
    readStatus: async () => (await getRecord("trainings", trainingId))?.status || null,
    writeStatus: async (status) => {
      const record = await getRecord("trainings", trainingId);
      if (!record) throw new Error("This training was deleted.");
      record.status = { ...status };
      await putRecord("trainings", record);
    },
    appendProgress: (row) => appendRow("progress", row),
    appendExam: (row) => appendRow("exams", row),
    writeReplay: (name, replay) => writeFile(trainingId, `replays/${name}`, replay),
    listReplays: async () => (await listFiles(trainingId)).filter((f) => f.startsWith("replays/")).map((f) => f.slice(8)),
    deleteReplay: (name) => deleteRecord("files", `${trainingId}/replays/${name}`),
    saveBrain: (which, data) => writeFile(trainingId, `brain/${which}`, data),
    loadBrain: (which) => readFile(trainingId, `brain/${which}`),
    loadOtherBrain: (otherId) => readFile(otherId, "brain/best"),
    log: (line) => appendLog(trainingId, `${new Date().toLocaleTimeString()}  ${line}`),
  };
}
