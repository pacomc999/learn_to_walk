// The studio's fixed data: every setting (with its help text, range and default), the world templates and the
// presets. It lives in data/catalog.json (exported once from the Python studio) and is loaded at start.

let catalog = null;

// Hand the loaded catalog to the engine (the page fetches it, the tests read it from disk)
export function useCatalog(data) {
  catalog = data;
}

export function getCatalog() {
  if (!catalog) throw new Error("The catalog is not loaded yet.");
  return catalog;
}

// The setting groups of one animal, by its key
export function animalGroups(key) {
  const animal = getCatalog().animals.find((a) => a.key === key);
  if (!animal) throw new Error(`There is no animal called "${key}".`);
  return animal.groups;
}

export function worldGroups() { return getCatalog().world_groups; }
export function brainGroups() { return getCatalog().brain_groups; }
export function learningGroups() { return getCatalog().learning_groups; }
