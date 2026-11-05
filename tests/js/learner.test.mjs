// The JavaScript learner studies exactly like Stable-Baselines3: from the same brain and the same batch, one study
// session gives the same new weights and the same approx_kl (fixtures from make_fixtures.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadTensorflowForTests } from "./tf_node.mjs";
import { Learner, orthogonal } from "../../studio/web/js/engine/learner.js";
import { Random } from "../../studio/web/js/engine/random.js";

const tf = await loadTensorflowForTests();
const cases = JSON.parse(readFileSync(new URL("./fixtures/learners.json", import.meta.url), "utf8"));

test("orthogonal starting weights have orthonormal rows or columns, times the gain", () => {
  for (const [rows, columns] of [[8, 5], [5, 8]]) {
    const w = orthogonal(rows, columns, 2.0, new Random(1));
    const small = Math.min(rows, columns);
    for (let a = 0; a < small; a++) {
      for (let b = 0; b < small; b++) {
        let dot = 0;
        for (let k = 0; k < Math.max(rows, columns); k++) {
          dot += rows >= columns ? w[k * columns + a] * w[k * columns + b] : w[a * columns + k] * w[b * columns + k];
        }
        assert.ok(Math.abs(dot - (a === b ? 4 : 0)) < 1e-5);
      }
    }
  }
});

cases.forEach((item, index) => {
  test(`learner ${index} (${item.design.symmetric ? "symmetric" : "free"}) studies like Stable-Baselines3`, () => {
    const learner = new Learner(tf, {
      kind: item.design.symmetric ? "symmetric" : "free", activation: item.design.activation, inputSize: item.inputSize,
      motorCount: item.motorCount, actorLayers: item.design.layers, criticLayers: item.design.layers, mirror: item.mirror,
      learningRate: 1e-3,
    });
    try {
      learner.loadBrain(item.before);
      const count = item.returns.length;
      const stats = learner.update({ observations: Float32Array.from(item.observations), actions: Float32Array.from(item.actions),
        logProbs: Float32Array.from(item.logProbs), advantages: Float32Array.from(item.advantages),
        returns: Float32Array.from(item.returns), count }, { epochs: 1, batchSize: count, entropyWeight: item.entropy, learningRate: 1e-3 });
      assert.ok(Math.abs(stats.approx_kl - item.approxKl) < 1e-5 + 1e-3 * Math.abs(item.approxKl), `approx_kl ${stats.approx_kl} vs ${item.approxKl}`);
      const after = learner.toBrain();
      const compare = (mine, theirs, label) => {
        let worst = 0;
        mine.forEach((value, i) => { worst = Math.max(worst, Math.abs(value - theirs[i])); });
        assert.ok(worst < 2e-5, `${label}: largest difference ${worst}`);
      };
      for (const net of ["actor", "critic"]) {
        after[net].layers.forEach((layer, l) => { compare(layer.w, item.after[net].layers[l].w, `${net} layer ${l} weights`); compare(layer.b, item.after[net].layers[l].b, `${net} layer ${l} biases`); });
        compare(after[net].head.w, item.after[net].head.w, `${net} head weights`);
      }
      compare(after.logStd, item.after.logStd, "wobble");
    } finally {
      learner.dispose();
    }
  });
});

test("studying does not leave tensors behind, lesson after lesson", () => {
  const inputs = 12, motors = 4;
  const learner = new Learner(tf, { kind: "free", activation: "tanh", inputSize: inputs, motorCount: motors, actorLayers: [16], criticLayers: [16] });
  const count = 64;
  const batch = { count, observations: Float32Array.from({ length: count * inputs }, Math.random), actions: Float32Array.from({ length: count * motors }, Math.random),
    logProbs: new Float32Array(count).fill(-5), advantages: Float32Array.from({ length: count }, Math.random), returns: Float32Array.from({ length: count }, Math.random) };
  const lesson = (extra) => {
    learner.update(batch, { epochs: 2, batchSize: 16, entropyWeight: 0.01, learningRate: 3e-4 });
    learner.judgeValues(Float32Array.from({ length: (count + extra) * inputs }, Math.random), count + extra);
    learner.toBrain();
  };
  lesson(1);  // the first lesson creates Adam's running averages, which stay
  const before = tf.memory().numTensors;
  for (let n = 2; n <= 5; n++) lesson(n);
  try {
    assert.equal(tf.memory().numTensors, before, "tensors left over after four more lessons");
  } finally {
    learner.dispose();
  }
});

test("the judge's values do not depend on the padding of its batch", () => {
  const inputs = 6;
  const learner = new Learner(tf, { kind: "free", activation: "tanh", inputSize: inputs, motorCount: 2, actorLayers: [8], criticLayers: [8] });
  try {
    const rows = Float32Array.from({ length: 300 * inputs }, () => Math.random() - 0.5);
    const all = learner.judgeValues(rows, 300);
    const few = learner.judgeValues(rows.subarray(0, 7 * inputs), 7);
    assert.equal(all.length, 300);
    assert.equal(few.length, 7);
    few.forEach((value, i) => assert.ok(Math.abs(value - all[i]) < 1e-6));
  } finally {
    learner.dispose();
  }
});
