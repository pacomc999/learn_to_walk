// The JavaScript networks answer exactly like the Python (PyTorch) brains: best moves, judge, wobble and cells.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { actionStd, brainCells, judgeValue, meanAction } from "../../studio/web/js/engine/network.js";

const cases = JSON.parse(readFileSync(new URL("./fixtures/brains.json", import.meta.url), "utf8"));

// The saved brain uses plain lists; the engine uses Float32Arrays
function load(brain) {
  const layer = (l) => ({ ...l, w: Float32Array.from(l.w), b: Float32Array.from(l.b) });
  const net = (n) => ({ layers: n.layers.map(layer), head: layer(n.head) });
  return { ...brain, actor: net(brain.actor), critic: net(brain.critic), logStd: Float32Array.from(brain.logStd) };
}

const close = (a, b, label) => assert.ok(Math.abs(a - b) <= 1e-4 * Math.max(1, Math.abs(b)), `${label}: ${a} vs ${b}`);

cases.forEach((item, index) => {
  test(`brain ${index} (${item.brain.kind}, ${item.brain.activation}) answers like PyTorch`, () => {
    const brain = load(item.brain);
    item.observations.forEach((observation, n) => {
      const scaled = Float32Array.from(observation);
      meanAction(brain, scaled).forEach((value, m) => close(value, item.actions[n][m], `obs ${n} motor ${m}`));
      close(judgeValue(brain, scaled), item.values[n], `obs ${n} judge`);
      assert.deepEqual(brainCells(brain, scaled), item.cells[n]);
    });
    actionStd(brain).forEach((value, m) => close(value, item.std[m], `wobble ${m}`));
  });
});
