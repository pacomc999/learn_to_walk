// Starting, pausing and following training sessions in the browser. A port of studio/manager.py.
// Each session runs in its own background worker (trainer.worker.js) while this tab is open. A session is alive
// while its worker holds the Web Lock "training-<id>" (in this tab or another one).

import { getCatalog } from "./catalog.js";
import { buildCreature } from "./creatures/index.js";
import { cleanDesign, designForBody, designSenses } from "./design.js";
import { observationSize } from "./env.js";
import { newId, now, rememberSpeed, safeId, systemFacts } from "./library.js";
import { deleteFiles, deleteRecord, allRecords, getRecord, listFiles, putRecord, readFile, writeFile } from "./storage.js";
import { cleanLearning } from "./trainer.js";
import { worldFromDict, worldToDict } from "./worlds.js";

const ACTIVE_STATES = ["starting", "training", "examining"];
const workers = new Map();   // training id -> its trainer worker, for sessions started in this tab
let wakeLock = null;

// ---------- Starting ----------

export async function createTraining(request) {
  const creature = request.creature || {};
  const animal = getCatalog().animals.find((a) => a.key === (creature.animal || "duck"));
  if (!animal) throw new Error(`There is no animal called "${creature.animal}".`);
  const creatureValues = creature.values || {};
  const world = worldFromDict(request.world || {});
  const design = cleanDesign(request.brain);
  const learning = cleanLearning(request.learning);
  const course = cleanCourse(request.course);
  if (world.turning || course.some((stage) => worldFromDict(stage.world).turning)) design.values.feel_turn = true;
  if (!buildCreature(animal.key, creatureValues).motors.length) {
    throw new Error("This creature has no motors, so there is nothing for its brain to learn. Give at least one part a joint.");
  }
  const name = String(request.name || `${creature.name || animal.name} in ${world.name}`).trim().slice(0, 80);
  const id = newId(name);
  const setup = {
    id, name, created: now(),
    creature: { animal: animal.key, name: creature.name || animal.name, values: creatureValues, id: creature.id ?? null },
    world: worldToDict(world), brain: design, learning, start_from: request.start_from || null, course,
  };
  if (setup.start_from) await checkStartFrom(setup);
  await putRecord("trainings", { id, setup, status: { state: "starting", steps: 0, total_steps: Math.trunc(learning.million_steps * 1e6), message: "Starting", updated: now() } });
  await startWorker(id, false);
  return setup;
}

// Course stages after the first world: each a world and its practice steps (at most five)
function cleanCourse(course) {
  return (Array.isArray(course) ? course : []).slice(0, 5).filter((stage) => stage && typeof stage === "object").map((stage) => {
    const million = Number(stage.million_steps ?? 2.0);
    return { world: worldToDict(worldFromDict(stage.world || {})), million_steps: Number.isFinite(million) ? Math.min(Math.max(million, 0.1), 100.0) : 2.0 };
  });
}

// A brain can only carry on in a body with the same motors and senses
async function checkStartFrom(setup) {
  const source = await getRecord("trainings", setup.start_from);
  if (!source) throw new Error("The brain to start from was not found.");
  if (!(await readFile(setup.start_from, "brain/best"))) throw new Error("That training has no best brain yet.");
  if (JSON.stringify(brainShape(source.setup)) !== JSON.stringify(brainShape(setup))) {
    throw new Error("That brain does not fit: the animal needs the same motors, senses and brain layers.");
  }
}

function brainShape(setup) {
  const build = buildCreature(setup.creature.animal, setup.creature.values);
  const design = designForBody(cleanDesign(setup.brain), build);
  return [build.motorNames, observationSize(build, designSenses(design)), design.layers, design.judge_layers, design.values.symmetric, design.values.activation];
}

async function startWorker(id, resume) {
  if (await isRunning(id)) return;
  const worker = new Worker(new URL("./trainer.worker.js", import.meta.url), { type: "module" });
  workers.set(id, worker);
  worker.onmessage = (event) => {
    if (event.data.type === "done" || event.data.type === "error") finishWorker(id);
  };
  worker.onerror = (event) => { console.error("Training worker failed", event.message); finishWorker(id); };
  worker.postMessage({ type: "start", catalog: getCatalog(), trainingId: id, resume, cores: navigator.hardwareConcurrency || 4 });
  keepScreenAwake();
}

function finishWorker(id) {
  const worker = workers.get(id);
  if (worker) worker.terminate();
  workers.delete(id);
  if (!workers.size && wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
}

// Ask the screen to stay on while training (the browser may say no; training goes on while the tab is open)
async function keepScreenAwake() {
  try {
    if (!wakeLock && navigator.wakeLock) wakeLock = await navigator.wakeLock.request("screen");
  } catch { wakeLock = null; }
}

// Training stops when the tab closes: say so before leaving
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", (event) => {
    if (workers.size) { event.preventDefault(); event.returnValue = ""; }
  });
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && workers.size) keepScreenAwake(); });
}

export async function continueTraining(id) {
  const record = await getRecord("trainings", safeId(id));
  if (!record) throw new Error("That training was not found.");
  record.status = { ...(record.status || {}), state: "starting", message: "Carrying on", updated: now() };
  await putRecord("trainings", record);
  await startWorker(id, true);
}

// More practice for a session: raise its total and carry on from its latest brain
export async function extendTraining(id, extraMillion) {
  if (await isRunning(id)) throw new Error("It is still training. Pause it first, or wait until it finishes.");
  extraMillion = Math.min(Math.max(Number(extraMillion) || 2, 0.1), 100.0);
  const record = await getRecord("trainings", safeId(id));
  const { setup } = record;
  const status = record.status || {};
  const learning = cleanLearning(setup.learning);
  const done = Math.max(Math.trunc(status.steps || 0), 0);
  const course = setup.course || [];
  let total;
  if (course.length) {  // the last stage of a course gets the extra practice
    const lastStart = learning.million_steps * 1e6 + course.slice(0, -1).reduce((sum, stage) => sum + stage.million_steps * 1e6, 0);
    course[course.length - 1].million_steps = Math.round((Math.max(done - lastStart, 0) / 1e6 + extraMillion) * 1000) / 1000;
    total = lastStart + course[course.length - 1].million_steps * 1e6;
  } else {
    learning.million_steps = Math.round((done / 1e6 + extraMillion) * 1000) / 1000;
    total = learning.million_steps * 1e6;
  }
  setup.learning = learning;
  setup.course = course;
  record.status = { ...status, total_steps: Math.trunc(total), state: "starting", message: "Training longer", updated: now() };
  await putRecord("trainings", record);
  await startWorker(id, true);
}

export async function requestStop(id) {
  await writeFile(safeId(id), "stop", true);
  const worker = workers.get(id);
  if (worker) worker.postMessage({ type: "stop" });
}

export async function isRunning(id) {
  if (!navigator.locks) return workers.has(id);
  const { held } = await navigator.locks.query();
  return held.some((lock) => lock.name === `training-${id}`);
}

// ---------- Reading back ----------

async function statusOf(record) {
  const status = { ...(record.status || {}) };
  const running = await isRunning(record.id);
  if (ACTIVE_STATES.includes(status.state || "starting") && !running) {
    const starting = workers.has(record.id);  // a worker of this tab that has not taken its lock yet
    if (!starting) {
      status.state = "interrupted";
      status.message = "It stopped when its tab was closed. Press Continue to carry on.";
    }
  }
  status.running = running || workers.has(record.id);
  return status;
}

async function summary(record) {
  const { setup } = record;
  const brain = await getRecord("files", `${record.id}/brain/best`);
  return {
    id: record.id, name: setup.name, created: setup.created,
    creature: { animal: setup.creature.animal, name: setup.creature.name }, creature_values: setup.creature.values || {},
    world: { name: setup.world?.name, id: setup.world?.id, style: setup.world?.values?.style },
    status: await statusOf(record), has_brain: Boolean(brain), imported: Boolean(setup.imported_from),
    thumbnail: record.thumbnail || null,
  };
}

export async function listTrainings() {
  const records = await allRecords("trainings");
  const items = await Promise.all(records.filter((r) => r && r.setup).map(summary));
  return items.sort((a, b) => String(b.created || "").localeCompare(String(a.created || "")));
}

export async function trainingDetails(id) {
  const record = await getRecord("trainings", safeId(id));
  if (!record) throw new Error("That training was not found.");
  const files = await listFiles(id);
  const replays = files.filter((f) => f.startsWith("replays/exam_")).map((f) => f.slice(8)).sort();
  const status = await statusOf(record);
  await rememberMeasuredSpeed(record.setup, status);
  const setup = { ...record.setup, brain: cleanDesign(record.setup.brain), learning: cleanLearning(record.setup.learning),
    world: worldToDict(worldFromDict(record.setup.world || {})) };
  return {
    setup, status, progress: (await readFile(id, "progress", [])) || [], exams: (await readFile(id, "exams", [])) || [], replays,
    has_best: files.includes("replays/best"), has_brain: files.includes("brain/best"),
    log: ((await readFile(id, "log", [])) || []).slice(-40).join("\n"),
  };
}

// Keep this computer's measured practice speed, so the Training plan can estimate how long a run takes
async function rememberMeasuredSpeed(setup, status) {
  const speed = status.steps_per_second || 0;
  if (status.state === "training" && speed > 0 && (status.steps || 0) > 50_000) {
    const worlds = Math.trunc(cleanLearning(setup.learning).parallel_worlds);
    const known = ((await systemFacts()).speeds || {})[String(worlds)];
    if (known === undefined || Math.abs(known - speed) > 0.1 * known) await rememberSpeed(worlds, speed);
  }
}

export async function readReplay(id, name) {
  const replay = await readFile(safeId(id), `replays/${safeId(name)}`);
  if (!replay) throw new Error("That replay was not found.");
  return replay;
}

export async function renameTraining(id, name) {
  const record = await getRecord("trainings", safeId(id));
  record.setup.name = String(name || record.setup.name).trim().slice(0, 80);
  await putRecord("trainings", record);
}

export async function deleteTraining(id) {
  if (await isRunning(id)) throw new Error("Pause this training before deleting it.");
  await deleteFiles(safeId(id));
  await deleteRecord("trainings", id);
}

// The picture on the walker's card, as a small JPEG data URL
export async function saveThumbnail(id, dataUrl) {
  if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:image/jpeg") || dataUrl.length > 2_000_000) throw new Error("That is not a small JPEG picture.");
  const record = await getRecord("trainings", safeId(id));
  record.thumbnail = dataUrl;
  await putRecord("trainings", record);
}
