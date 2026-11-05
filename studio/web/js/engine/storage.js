// Storage in the browser: IndexedDB, the browser's own database. Everything the user makes lives here, on their
// computer only. One store per kind of thing; every record is a plain object found by its key.
//   creatures, worlds, brains: { id, name, ... }       library items, by id
//   trainings: { id, setup, status, ... }               one record per training session, by id
//   files: { key: "<training id>/<name>", value }      bigger pieces of a training (replays, brain weights, charts)
//   system: { key, value }                              facts about this computer (measured practice speed)

const DB_NAME = "walk_studio";
const DB_VERSION = 1;
const STORES = { creatures: "id", worlds: "id", brains: "id", trainings: "id", files: "key", system: "key" };

let opening = null;

function openDatabase() {
  if (!opening) {
    opening = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        for (const [name, keyPath] of Object.entries(STORES)) {
          if (!request.result.objectStoreNames.contains(name)) request.result.createObjectStore(name, { keyPath });
        }
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new Error("The browser would not open its storage. Private windows sometimes block it."));
    });
  }
  return opening;
}

// Run one request on a store and wait for its answer
async function run(storeName, mode, makeRequest) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, mode);
    const request = makeRequest(transaction.objectStore(storeName));
    let result;
    request.onsuccess = () => { result = request.result; };
    transaction.oncomplete = () => resolve(result);
    transaction.onerror = () => reject(transaction.error || new Error("Saving in the browser failed."));
    transaction.onabort = () => reject(transaction.error || new Error("Saving in the browser was stopped (is the disk full?)."));
  });
}

export const getRecord = (store, key) => run(store, "readonly", (s) => s.get(key));
export const putRecord = (store, record) => run(store, "readwrite", (s) => s.put(record));
export const deleteRecord = (store, key) => run(store, "readwrite", (s) => s.delete(key));
export const allRecords = (store) => run(store, "readonly", (s) => s.getAll());

// ---------- Files of a training ----------

export async function readFile(trainingId, name, fallback = null) {
  const record = await getRecord("files", `${trainingId}/${name}`);
  return record ? record.value : fallback;
}

export function writeFile(trainingId, name, value) {
  return putRecord("files", { key: `${trainingId}/${name}`, value });
}

// The names of every file of a training
export async function listFiles(trainingId) {
  const db = await openDatabase();
  const prefix = `${trainingId}/`;
  return new Promise((resolve, reject) => {
    const range = IDBKeyRange.bound(prefix, `${prefix}￿`);
    const request = db.transaction("files", "readonly").objectStore("files").getAllKeys(range);
    request.onsuccess = () => resolve(request.result.map((key) => key.slice(prefix.length)));
    request.onerror = () => reject(request.error);
  });
}

export async function deleteFiles(trainingId) {
  const db = await openDatabase();
  const prefix = `${trainingId}/`;
  return new Promise((resolve, reject) => {
    const transaction = db.transaction("files", "readwrite");
    transaction.objectStore("files").delete(IDBKeyRange.bound(prefix, `${prefix}￿`));
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

// Ask the browser to keep this storage even when the disk gets full (it may still say no)
export async function askToKeepStorage() {
  try {
    if (navigator.storage && navigator.storage.persist) return await navigator.storage.persist();
  } catch { /* not supported: storage is still kept unless the user clears it */ }
  return false;
}
