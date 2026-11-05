// Sharing a walker: everything about one training packed into a single .walker file, and opened again in another
// browser as a walker of its own. A port of studio/sharing.py with a browser format: compressed (gzip) JSON,
// with number arrays (brain weights) written as base64. The Python studio can write the same format
// (python -m studio.export_web walker ...), so walkers trained there can come over.

import { newId, now } from "./library.js";
import { getRecord, listFiles, putRecord, readFile, writeFile } from "./storage.js";

export const FORMAT = "walk-studio-walker";
const MOST_BYTES = 300 * 1024 * 1024;

// ---------- Number arrays in JSON ----------

function toBase64(bytes) {
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

function fromBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

const replacer = (key, value) => {
  if (value instanceof Float32Array) return { $f32: toBase64(new Uint8Array(value.buffer, value.byteOffset, value.byteLength)) };
  if (value instanceof Float64Array) return { $f64: toBase64(new Uint8Array(value.buffer, value.byteOffset, value.byteLength)) };
  return value;
};

const reviver = (key, value) => {
  if (value && typeof value === "object" && typeof value.$f32 === "string") return new Float32Array(fromBase64(value.$f32).buffer);
  if (value && typeof value === "object" && typeof value.$f64 === "string") return new Float64Array(fromBase64(value.$f64).buffer);
  return value;
};

async function gzip(text) {
  const stream = new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).blob();
}

async function gunzip(blob) {
  const stream = blob.stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).text();
}

// ---------- Packing and opening ----------

// The walker as a file: { blob, filename }
export async function packWalker(trainingId) {
  const record = await getRecord("trainings", trainingId);
  if (!record) throw new Error("That walker was not found.");
  const files = await listFiles(trainingId);
  const replays = {};
  for (const file of files.filter((f) => f.startsWith("replays/"))) replays[file.slice(8)] = await readFile(trainingId, file);
  const brains = {};
  for (const which of ["best", "latest"]) {
    const brain = await readFile(trainingId, `brain/${which}`);
    if (brain) brains[which] = brain;
  }
  const data = {
    format: FORMAT, version: 1, setup: record.setup, status: record.status || {}, thumbnail: record.thumbnail || null,
    progress: (await readFile(trainingId, "progress", [])) || [], exams: (await readFile(trainingId, "exams", [])) || [],
    challenges: await readFile(trainingId, "challenges"), replays, brains,
  };
  const name = String(record.setup.name || "walker").replace(/[^A-Za-z0-9 \-_.()]/g, "_").trim() || "walker";
  return { blob: await gzip(JSON.stringify(data, replacer)), filename: `${name}.walker` };
}

// Open a .walker file as a new walker; returns its id
export async function openWalker(file) {
  if (file.size > MOST_BYTES) throw new Error("That file is too big to be a walker.");
  const start = new Uint8Array(await file.slice(0, 2).arrayBuffer());
  if (start[0] === 0x50 && start[1] === 0x4b) {  // "PK": a zip from the Python studio
    throw new Error("This walker was saved by the Python studio. Convert it first: python -m studio.export_web walker \"" + file.name + "\"");
  }
  if (start[0] !== 0x1f || start[1] !== 0x8b) throw new Error("That is not a .walker file.");
  let data;
  try {
    data = JSON.parse(await gunzip(file), reviver);
  } catch {
    throw new Error("That .walker file is damaged.");
  }
  if (!data || data.format !== FORMAT || !data.setup) throw new Error("That is not a .walker file (it has no recipe).");
  const id = newId(data.setup.name || "walker");
  const setup = { ...data.setup, id, start_from: null, shared: true };
  const status = { ...(data.status || {}) };
  if (["starting", "training", "examining"].includes(status.state)) status.state = "stopped";
  Object.assign(status, { message: "Opened from a shared walker file.", updated: now() });
  delete status.pid;
  await putRecord("trainings", { id, setup, status, thumbnail: typeof data.thumbnail === "string" ? data.thumbnail : null });
  await writeFile(id, "progress", Array.isArray(data.progress) ? data.progress : []);
  await writeFile(id, "exams", Array.isArray(data.exams) ? data.exams : []);
  if (data.challenges) await writeFile(id, "challenges", data.challenges);
  for (const [name, replay] of Object.entries(data.replays || {})) {
    if (/^(best|exam_\d{12})$/.test(name)) await writeFile(id, `replays/${name}`, replay);  // only known names
  }
  for (const which of ["best", "latest"]) if (data.brains && data.brains[which]) await writeFile(id, `brain/${which}`, data.brains[which]);
  return id;
}

// Hand a file to the person (the browser saves it in their downloads)
export function saveFile(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
