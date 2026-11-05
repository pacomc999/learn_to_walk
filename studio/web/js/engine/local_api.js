// The studio's answers, worked out in the browser. The pages call api("preview/creature", ...) as they always did;
// this module answers each request here instead of asking a server. A request it cannot answer yet goes to the
// Python server if one is running, so nothing breaks while parts are still being moved into the browser.

import { getCatalog, useCatalog } from "./catalog.js";
import { ANIMALS, buildCreature } from "./creatures/index.js";
import { countWeights, cleanDesign, designForBody, designSenses } from "./design.js";
import { observationSize } from "./env.js";
import { deleteItem, getItem, KINDS, listItems, saveItem, systemFacts } from "./library.js";
import { loadPhysics } from "./physics.js";
import { askHelper, jobStatus, liveSession, savedReportCard, startLive, startReportCard } from "./helpers.js";
import { creaturePreview, dropTest } from "./previews.js";
import { continueTraining, createTraining, deleteTraining, extendTraining, listTrainings, readReplay, renameTraining, requestStop,
  saveThumbnail, trainingDetails } from "./trainings.js";

// ---------- Starting up ----------

let ready = null;

// Load the catalog (and start loading MuJoCo) once
export function startEngine() {
  if (!ready) {
    ready = fetch(new URL("../../data/catalog.json", import.meta.url))
      .then((response) => {
        if (!response.ok) throw new Error("The studio's data file (data/catalog.json) is missing.");
        return response.json();
      })
      .then((data) => { useCatalog(data); loadPhysics(); });
  }
  return ready;
}

// A request the browser cannot answer: say so with a status, like a server would
class NotHere extends Error {}

// Whether the Python studio is serving this page (then it can still import learn_to_walk ducks)
let pythonStudio = null;
async function pythonStudioAnswers() {
  if (pythonStudio === null) {
    try {
      const response = await fetch("/api/import/guess");
      pythonStudio = response.ok && (response.headers.get("content-type") || "").includes("json");
    } catch {
      pythonStudio = false;
    }
  }
  return pythonStudio;
}

// ---------- Answers ----------

// Answer one request. path: "preview/creature", "creatures/<id>", ...; method: GET, POST or DELETE
export async function localAnswer(path, method = "GET", body = null) {
  await startEngine();
  const [route, ...rest] = path.split("?")[0].split("/").filter(Boolean);
  if (route === "catalog" && method === "GET") return catalog();
  if (route === "preview") return preview(rest[0], body || {});
  if (KINDS.includes(route)) return libraryRequest(route, method, rest, body);
  if (route === "trainings") {
    const answer = await trainingRequest(method, rest, body || {});
    if (answer !== undefined) return answer;
  }
  if (route === "race" && method === "POST") {
    const b = body || {};
    return askHelper("race", { ids: [...new Set(b.ids || [])].slice(0, 6), options: { world: b.world || {}, speed: b.speed ?? null,
      height: b.height ?? null, seed: Math.trunc(b.seed ?? 1000), seconds: Number(b.seconds ?? 20) } });
  }
  if (route === "jobs" && rest.length) return jobStatus(rest[0]);
  if (route === "live") return liveRequest(method, rest, body || {}, path);
  if (route === "import" && !(await pythonStudioAnswers())) {
    throw new Error("Bringing ducks over from learn_to_walk needs the Python studio (it reads PyTorch brains from your disk). "
      + "Import them there, convert them with: python -m studio.export_web walker <the walker's id>, then open the .walker file here.");
  }
  throw new NotHere(path);
}

// Live mode: start a walker, read its new frames (?after=N), send it commands
async function liveRequest(method, rest, body, path) {
  if (method === "POST" && !rest.length) return startLive(String(body.training || ""), body.world || null, Number(body.speed ?? 0.4));
  const session = liveSession(rest[0]);
  if (method === "GET") {
    const after = new URLSearchParams(path.split("?")[1] || "").get("after");
    return session.framesAfter(Number(after ?? -1));
  }
  session.command(body);
  return { ok: true };
}

// The training requests the browser answers (watching, races and the report card come next)
async function trainingRequest(method, rest, body) {
  if (!rest.length) {
    if (method === "GET") return listTrainings();
    if (method === "POST") return createTraining(body);
    return undefined;
  }
  const [id, action = "", name] = rest;
  if (method === "GET" && !action) return trainingDetails(id);
  if (method === "DELETE" && !action) { await deleteTraining(id); return { ok: true }; }
  if (method === "GET" && action === "replay" && name) return readReplay(id, name);
  if (method === "POST" && action === "stop") { await requestStop(id); return { ok: true }; }
  if (method === "POST" && action === "continue") { await continueTraining(id); return { ok: true }; }
  if (method === "POST" && action === "extend") { await extendTraining(id, body.million ?? 2.0); return { ok: true }; }
  if (method === "POST" && action === "rename") { await renameTraining(id, body.name); return { ok: true }; }
  if (method === "POST" && action === "thumbnail") { await saveThumbnail(id, body.image); return { ok: true }; }
  if (method === "POST" && action === "watch") {
    return askHelper("watch", { training: id, options: { world: body.world || null, speed: body.speed ?? null, height: body.height ?? null,
      seed: Math.trunc(body.seed ?? 1000), seconds: body.seconds ?? null, turn: Number(body.turn || 0) } });
  }
  if (action === "challenges") {
    if (method === "GET") return savedReportCard(id);
    if (method === "POST" && name) return askHelper("level", { training: id, challenge: name, level: Number(body.level) });
    if (method === "POST") return { job: startReportCard(id) };
  }
  return undefined;
}

async function catalog() {
  const data = { ...getCatalog() };
  const cores = Math.max(1, navigator.hardwareConcurrency || 4);
  data.system = { cores, platform: "browser", ...(await systemFacts()) };
  return data;
}

async function preview(kind, body) {
  const mujoco = await loadPhysics();
  const animal = body.animal || "duck";
  if (kind === "creature") return creaturePreview(mujoco, animal, body.values || {});
  if (kind === "drop") return dropTest(mujoco, animal, body.values || {}, body.world || null, Number(body.seconds ?? 4.0));
  if (kind === "brain") return brainFacts(animal, body.values || {}, body.design);
  throw new Error("Unknown preview.");
}

// Facts about a brain design for one creature: senses, motors and how many numbers it learns
function brainFacts(animalKey, values, designIn) {
  const build = buildCreature(animalKey, values);
  const design = designForBody(cleanDesign(designIn), build);
  const inputs = observationSize(build, designSenses(design));
  const animal = getCatalog().animals.find((a) => a.key === animalKey);
  if (!ANIMALS[animalKey] || !animal) throw new Error(`There is no animal called "${animalKey}".`);
  return {
    inputs, outputs: build.motors.length, motor_names: build.motorNames,
    weights: countWeights(inputs, build.motors.length, design), design,
    left_motors: build.motors.filter((m) => m.side === "left").length,
    middle_motors: build.motors.filter((m) => m.side === "middle").length,
    animal_name: animal.name, symmetric_ok: build.symmetricOk,
  };
}

async function libraryRequest(kind, method, rest, body) {
  if (method === "GET" && !rest.length) return listItems(kind);
  if (method === "POST" && !rest.length) return saveItem(kind, body || {});
  if (method === "DELETE" && rest.length) { await deleteItem(kind, rest[0]); return { ok: true }; }
  if (method === "GET" && rest.length) {
    const item = await getItem(kind, rest[0]);
    if (!item) throw new Error("Not found.");
    return item;
  }
  throw new Error("Unknown request.");
}

export { NotHere };
