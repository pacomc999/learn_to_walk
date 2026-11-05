// Practice pools: where the practice worlds of a training run. All of them answer the same three calls:
//   setDifficulty(difficulty), rollout(brain, steps) -> merged practice, dispose()
// The inline pool runs every world in the calling thread. The worker pool spreads them over background workers,
// one per processor core, which is what makes training fast.

import { PracticeGroup } from "./practice.js";
import { mergeRollouts } from "./trainer.js";

export class InlinePool {
  constructor(mujoco, spec) {
    this.group = new PracticeGroup(mujoco, spec);
  }

  async setDifficulty(difficulty) { this.group.setDifficulty(difficulty); }
  async rollout(brain, steps, options = {}) { return this.group.rollout(brain, steps, options); }
  async dispose() { this.group.dispose(); }
}

// Split `worlds` worlds over at most `workers` groups, as evenly as possible
export function splitWorlds(worlds, workers) {
  const groups = Math.max(1, Math.min(worlds, workers));
  return Array.from({ length: groups }, (_, i) => Math.floor(worlds / groups) + (i < worlds % groups ? 1 : 0));
}

export class WorkerPool {
  // spec: { animal, values, world, senses, count (all worlds), seed }; catalog: the studio's data for the workers
  static async start(spec, catalog, workers) {
    const pool = new WorkerPool();
    pool.workers = [];
    let first = 0;
    for (const count of splitWorlds(spec.count, workers)) {
      const worker = new Worker(new URL("./practice.worker.js", import.meta.url), { type: "module" });
      pool.workers.push(worker);
      await pool.ask(worker, "start", { catalog, spec: { ...spec, count, seed: spec.seed + first } });
      first += count;
    }
    return pool;
  }

  // Send one message to a worker and wait for its answer
  ask(worker, type, data = {}) {
    return new Promise((resolve, reject) => {
      const listen = (event) => {
        if (event.data.type !== `${type}:done` && event.data.type !== "error") return;
        worker.removeEventListener("message", listen);
        if (event.data.type === "error") reject(new Error(event.data.message));
        else resolve(event.data.result);
      };
      worker.addEventListener("message", listen);
      worker.postMessage({ type, ...data });
    });
  }

  async setDifficulty(difficulty) {
    await Promise.all(this.workers.map((worker) => this.ask(worker, "difficulty", { difficulty })));
  }

  async rollout(brain, steps, options = {}) {
    const parts = await Promise.all(this.workers.map((worker) => this.ask(worker, "rollout", { brain, steps, options })));
    return mergeRollouts(parts);
  }

  async dispose() {
    for (const worker of this.workers) worker.terminate();
    this.workers = [];
  }
}
