// The helper worker: the heavy looks at trained walkers (test drives, races, report card levels), so the page
// never freezes. Messages in: { id, kind, catalog, args }. Out: { id, progress, message } while working, then
// { id, result } or { id, error }.

import { useCatalog } from "./catalog.js";
import { loadPhysics } from "./physics.js";
import { getRecord, readFile } from "./storage.js";
import { race, reportCard, runLevel, watch } from "./watching.js";

// A trained walker from the browser's storage: its setup and its best brain
async function trained(trainingId) {
  const record = await getRecord("trainings", trainingId);
  if (!record) throw new Error("That training was not found.");
  const best = await readFile(trainingId, "brain/best");
  if (!best) throw new Error("This walker has no trained brain yet. Wait for its first exam.");
  return { setup: record.setup, brain: best.brain };
}

self.onmessage = async (event) => {
  const { id, kind, catalog, args } = event.data;
  try {
    useCatalog(catalog);
    const mujoco = await loadPhysics();
    let result;
    if (kind === "watch") result = watch(mujoco, await trained(args.training), args.options);
    else if (kind === "race") result = race(mujoco, await Promise.all(args.ids.map(trained)), args.options);
    else if (kind === "level") result = runLevel(mujoco, await trained(args.training), args.challenge, args.level, true);
    else if (kind === "report") {
      result = reportCard(mujoco, await trained(args.training), (progress, message) => self.postMessage({ id, progress, message }));
    } else throw new Error(`Unknown job: ${kind}`);
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: String(error?.message || error) });
  }
};
