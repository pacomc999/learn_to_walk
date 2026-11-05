// A practice worker: a few practice worlds with their own MuJoCo, practising with the brain the trainer sends.
// Messages: start { catalog, spec }, difficulty { difficulty }, rollout { brain, steps }. Each is answered with
// "<type>:done" (and a result), or "error" with a message.

import { useCatalog } from "./catalog.js";
import { loadPhysics } from "./physics.js";
import { PracticeGroup } from "./practice.js";

let group = null;

self.onmessage = async (event) => {
  const { type } = event.data;
  try {
    let result = null;
    if (type === "start") {
      useCatalog(event.data.catalog);
      group = new PracticeGroup(await loadPhysics(), event.data.spec);
    } else if (type === "difficulty") {
      group.setDifficulty(event.data.difficulty);
    } else if (type === "rollout") {
      result = group.rollout(event.data.brain, event.data.steps, event.data.options || {});
    }
    self.postMessage({ type: `${type}:done`, result });
  } catch (error) {
    self.postMessage({ type: "error", message: String(error?.message || error) });
  }
};
