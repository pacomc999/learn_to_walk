// The library: creatures, worlds and brain designs the user saved. A port of studio/library.py,
// kept in the browser's storage instead of files.

import { allRecords, deleteRecord, getRecord, putRecord } from "./storage.js";

export const KINDS = ["creatures", "worlds", "brains"];

// A short id made from a name, for example "big_legs_duck_3f2a"
export function newId(name) {
  const slug = String(name).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 32) || "item";
  const random = Array.from(crypto.getRandomValues(new Uint8Array(2)), (b) => b.toString(16).padStart(2, "0")).join("");
  return `${slug}_${random}`;
}

export function safeId(itemId) {
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(itemId || "")) throw new Error("That is not a valid name for a library item.");
  return itemId;
}

// The time now, as Python's isoformat(timespec="seconds") in local time
export function now() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function checkKind(kind) {
  if (!KINDS.includes(kind)) throw new Error("Unknown kind of library item.");
}

export async function listItems(kind) {
  checkKind(kind);
  const items = (await allRecords(kind)).filter((item) => item && typeof item === "object");
  return items.sort((a, b) => String(b.updated || "").localeCompare(String(a.updated || "")));
}

export async function getItem(kind, itemId) {
  checkKind(kind);
  return (await getRecord(kind, safeId(itemId))) || null;
}

// Saving over an existing item keeps its id; anything else (new, or a built-in template) gets a new one
export async function saveItem(kind, data) {
  checkKind(kind);
  data = { ...data };
  const name = String(data.name || "Untitled").trim().slice(0, 80) || "Untitled";
  data.name = name;
  let existing = null;
  if (data.id && !data.builtin) {
    try { existing = await getItem(kind, data.id); } catch { existing = null; }
  }
  if (!existing) {
    data.id = newId(name);
    data.created = now();
  }
  delete data.builtin;
  data.created = data.created || now();
  data.updated = now();
  await putRecord(kind, data);
  return data;
}

export async function deleteItem(kind, itemId) {
  checkKind(kind);
  await deleteRecord(kind, safeId(itemId));
}

// ---------- Facts about this computer ----------

export async function systemFacts() {
  const record = await getRecord("system", "facts");
  return (record && record.value) || {};
}

// Measured practice speed, so the page can estimate how long training takes
export async function rememberSpeed(worlds, stepsPerSecond) {
  const facts = await systemFacts();
  facts.speeds = { ...(facts.speeds || {}), [String(worlds)]: Math.round(stepsPerSecond) };
  await putRecord("system", { key: "facts", value: facts });
}
