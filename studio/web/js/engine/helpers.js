// Watching trained walkers from the page: test drives, races and the report card run in the helper worker;
// live mode runs right here on the page (one small step every fiftieth of a second).

import { getCatalog } from "./catalog.js";
import { loadPhysics } from "./physics.js";
import { getRecord, readFile, writeFile } from "./storage.js";
import { LiveSession } from "./watching.js";

let helper = null;
let nextId = 1;
const waiting = new Map();  // job number -> { resolve, reject, onProgress }

function helperWorker() {
  if (!helper) {
    helper = new Worker(new URL("./helper.worker.js", import.meta.url), { type: "module" });
    helper.onmessage = (event) => {
      const { id, result, error, progress, message } = event.data;
      const job = waiting.get(id);
      if (!job) return;
      if (progress !== undefined) { job.onProgress(progress, message); return; }
      waiting.delete(id);
      if (error) job.reject(new Error(error)); else job.resolve(result);
    };
  }
  return helper;
}

// Ask the helper worker for one job and wait for its answer
export function askHelper(kind, args, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    waiting.set(id, { resolve, reject, onProgress });
    helperWorker().postMessage({ id, kind, args, catalog: getCatalog() });
  });
}

// ---------- The report card as a job the page follows ----------

const jobs = new Map();

export function startReportCard(trainingId) {
  for (const job of jobs.values()) if (job.training === trainingId && job.state === "running") return job.id;
  const id = Math.random().toString(16).slice(2, 12);
  const job = { id, training: trainingId, state: "running", progress: 0, message: "Starting" };
  jobs.set(id, job);
  askHelper("report", { training: trainingId }, (progress, message) => Object.assign(job, { progress, message }))
    .then(async (result) => {
      await writeFile(trainingId, "challenges", result);
      Object.assign(job, { state: "done", progress: 1, message: "Done", result });
    })
    .catch((error) => Object.assign(job, { state: "error", message: `Something went wrong: ${error.message}` }));
  return id;
}

export function jobStatus(id) {
  const job = jobs.get(id);
  if (!job) throw new Error("That job is not known (was the page reloaded?).");
  return job;
}

export async function savedReportCard(trainingId) {
  return (await readFile(trainingId, "challenges")) || {};
}

// ---------- Live mode ----------

let live = null;

export async function startLive(trainingId, worldData, speed) {
  const record = await getRecord("trainings", trainingId);
  if (!record) throw new Error("That walker was not found.");
  const best = await readFile(trainingId, "brain/best");
  if (!best) throw new Error("This walker has no trained brain yet.");
  if (live) live.stop();  // one live walker at a time is plenty
  live = new LiveSession(await loadPhysics(), { setup: record.setup, brain: best.brain }, worldData, speed);
  return live.header();
}

export function liveSession(id) {
  if (!live || live.id !== id) throw new Error("That live session has ended. Start it again.");
  return live;
}
