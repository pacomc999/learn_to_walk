// The learner: PPO's study step on the graphics card, with TensorFlow.js. A port of what Stable-Baselines3's PPO
// does with the studio's brains (studio/brain.py): the same starting weights, the clipped objective, the judge's
// loss, the curiosity bonus, gradient clipping and Adam. Practising (picking moves) happens in network.js.
//
// Weights live here as TensorFlow variables: w is [inputs, outputs] (x times w), unlike network.js, which keeps
// PyTorch's [outputs, inputs]. toBrain() and loadBrain() convert between the two.

import { Random } from "./random.js";

const HIDDEN_GAIN = Math.SQRT2;   // Stable-Baselines3's starting gains: hidden layers, actor output, critic output
const ACTOR_HEAD_GAIN = 0.01;
const CRITIC_HEAD_GAIN = 1.0;
const LOG_TWO_PI = Math.log(2 * Math.PI);
export const PPO_FIXED = { clipRange: 0.2, valueWeight: 0.5, maxGradNorm: 0.5, adamEpsilon: 1e-5, gaeLambda: 0.95 };
let learnersMade = 0;  // TensorFlow.js keeps one list of variable names, so each learner gets its own prefix

// ---------- Choosing where the maths runs ----------

// The fastest TensorFlow.js backend this browser has: WebGPU, else WebGL, else the processor
export async function startTensorflow(tf, prefer = ["webgpu", "webgl", "cpu"]) {
  for (const name of prefer) {
    try {
      if (name === "webgpu") {
        if (typeof navigator === "undefined" || !navigator.gpu) continue;
        await import("../../vendor/tfjs/tf-backend-webgpu.fesm.min.js");
      }
      if (await tf.setBackend(name)) {
        await tf.ready();
        // WebGL keeps freed textures for reuse and by default never gives them back; past 64 MB of them, let them go
        if (name === "webgl") tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 64 * 1024 * 1024);
        return name;
      }
    } catch { /* try the next one */ }
  }
  throw new Error("This browser cannot run the learning maths (no WebGPU, WebGL or processor backend).");
}

// ---------- Starting weights ----------

// An orthogonal matrix of rows x columns times the gain, as torch.nn.init.orthogonal_ makes them
// (row by row, PyTorch's [outputs, inputs] layout)
export function orthogonal(rows, columns, gain, random) {
  const tall = rows >= columns;
  const [n, m] = tall ? [rows, columns] : [columns, rows];  // n >= m: m orthonormal columns of length n
  const q = Array.from({ length: m }, () => Float64Array.from({ length: n }, () => random.normal()));
  for (let j = 0; j < m; j++) {  // Gram-Schmidt, which also fixes the signs the way QR with sign(diag(r)) does
    for (let k = 0; k < j; k++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += q[j][i] * q[k][i];
      for (let i = 0; i < n; i++) q[j][i] -= dot * q[k][i];
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += q[j][i] ** 2;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < n; i++) q[j][i] /= norm;
  }
  const out = new Float32Array(rows * columns);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) out[r * columns + c] = gain * (tall ? q[c][r] : q[r][c]);
  }
  return out;
}

// [outputs, inputs] row by row  <->  [inputs, outputs] row by row
function transpose(values, rows, columns) {
  const out = new Float32Array(values.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < columns; c++) out[c * rows + r] = values[r * columns + c];
  return out;
}

// ---------- The learner ----------

export class Learner {
  // options: kind ("free" | "symmetric"), activation, inputSize, motorCount, actorLayers, criticLayers,
  //          mirror (symmetric only), logStd (starting log of the wobble), seed, learningRate
  constructor(tf, options) {
    this.tf = tf;
    Object.assign(this, { kind: options.kind, activation: options.activation, inputSize: options.inputSize,
      motorCount: options.motorCount, mirror: options.mirror || null });
    this.learningRate = options.learningRate ?? 3e-4;
    learnersMade += 1;
    this.prefix = `learner${learnersMade}_`;
    const random = new Random(options.seed ?? 0);
    const symmetric = this.kind === "symmetric";
    const netInputs = symmetric ? 2 * this.inputSize : this.inputSize;
    this.actorOutputs = symmetric ? this.mirror.left_motors.length + this.mirror.middle_motors.length : this.motorCount;
    this.actor = this.makeNet("actor", netInputs, options.actorLayers, this.actorOutputs, ACTOR_HEAD_GAIN, random);
    this.critic = this.makeNet("critic", netInputs, options.criticLayers, 1, CRITIC_HEAD_GAIN, random);
    this.logStd = tf.variable(tf.fill([this.motorCount], options.logStd ?? -1.0), true, `${this.prefix}log_std`);
    if (symmetric) this.makeMirrorTensors();
    this.optimizer = tf.train.adam(this.learningRate, 0.9, 0.999, PPO_FIXED.adamEpsilon);
  }

  makeNet(name, inputs, sizes, outputs, headGain, random) {
    const tf = this.tf;
    const layer = (label, from, to, gain) => ({
      w: tf.variable(tf.tensor2d(transpose(orthogonal(to, from, gain, random), to, from), [from, to]), true, `${this.prefix}${label}_w`),
      b: tf.variable(tf.zeros([to]), true, `${this.prefix}${label}_b`), inputs: from, outputs: to,
    });
    const layers = [];
    let previous = inputs;
    sizes.forEach((size, i) => { layers.push(layer(`${name}_${i}`, previous, size, HIDDEN_GAIN)); previous = size; });
    return { layers, head: layer(`${name}_head`, previous, outputs, headGain) };
  }

  makeMirrorTensors() {
    const tf = this.tf, m = this.mirror;
    this.obsIndex = tf.tensor1d(m.observation_order, "int32");
    this.obsSign = tf.tensor1d(m.observation_signs, "float32");
    this.actIndex = tf.tensor1d(m.action_order, "int32");
    this.actSign = tf.tensor1d(m.action_signs, "float32");
    const place = new Array(this.motorCount).fill(this.actorOutputs);  // the extra zero at the end
    [...m.left_motors, ...m.middle_motors].forEach((motor, position) => { place[motor] = position; });
    this.placeIndex = tf.tensor1d(place, "int32");
  }

  variables() {
    const net = (n) => [...n.layers.flatMap((l) => [l.w, l.b]), n.head.w, n.head.b];
    return [...net(this.actor), ...net(this.critic), this.logStd];
  }

  // ---------- The networks on a batch ----------

  respond(x) {
    const tf = this.tf;
    if (this.activation === "relu") return tf.relu(x);
    if (this.activation === "elu") return tf.elu(x);
    if (this.activation === "silu") return tf.mul(x, tf.sigmoid(x));
    return tf.tanh(x);
  }

  runNet(net, x) {
    const tf = this.tf;
    let value = x;
    for (const layer of net.layers) value = this.respond(tf.add(tf.matMul(value, layer.w), layer.b));
    return tf.add(tf.matMul(value, net.head.w), net.head.b);
  }

  // Means of the tries [batch, motors] and the judge's values [batch], for a batch of scaled observations
  forward(observations) {
    const tf = this.tf;
    if (this.kind !== "symmetric") {
      return { mean: this.runNet(this.actor, observations), value: tf.reshape(this.runNet(this.critic, observations), [-1]) };
    }
    const batch = observations.shape[0];
    const mirrored = tf.mul(tf.gather(observations, this.obsIndex, 1), this.obsSign);
    // Both views stacked on the batch axis: real [s, Ms] first, then mirrored [Ms, s]
    const views = tf.concat([tf.concat([observations, mirrored], 1), tf.concat([mirrored, observations], 1)], 0);
    const outputs = this.runNet(this.actor, views);
    const padded = tf.concat([outputs, tf.zeros([2 * batch, 1])], 1);
    const placed = tf.gather(padded, this.placeIndex, 1);
    const [real, flipped] = tf.split(placed, 2, 0);
    const mean = tf.add(real, tf.mul(tf.gather(flipped, this.actIndex, 1), this.actSign));
    const values = tf.reshape(this.runNet(this.critic, views), [-1]);
    const [valueReal, valueFlipped] = tf.split(values, 2, 0);
    return { mean, value: tf.add(valueReal, valueFlipped) };
  }

  // The judge's values for many scaled observations at once (flat, row by row), read back in one go
  // The batch is padded with empty rows to a multiple of 256, so the graphics card sees the same few sizes every
  // lesson and can reuse its buffers (a new size each lesson made it keep more and more of them)
  judgeValues(observations, count) {
    if (!count) return new Float32Array(0);
    const tf = this.tf;
    const padded = Math.ceil(count / 256) * 256;
    const rows = new Float32Array(padded * this.inputSize);
    rows.set(observations.subarray(0, count * this.inputSize));
    return tf.tidy(() => this.forward(tf.tensor2d(rows, [padded, this.inputSize])).value.dataSync().slice(0, count));
  }

  // The log of the wobble per motor; the symmetric brain wobbles each motor like its mirror partner
  logStdUsed() {
    const tf = this.tf;
    if (this.kind !== "symmetric") return this.logStd;
    return tf.div(tf.add(this.logStd, tf.gather(this.logStd, this.actIndex)), 2);
  }

  // ---------- One study session (PPO's train()) ----------

  // batch: { observations, actions (both row by row), logProbs, advantages, returns, count }
  // settings: { epochs, batchSize, entropyWeight, learningRate }. Returns the average approx_kl and losses
  update(batch, settings, random = new Random(0)) {
    const tf = this.tf;
    this.setLearningRate(settings.learningRate ?? this.learningRate);
    const { count } = batch;
    const obsSize = this.inputSize, actSize = this.motorCount;
    // The numbers of each minibatch are added up on the graphics card and read once at the end: reading a number
    // back makes the graphics card stop and wait, which would cost more than the study itself
    let totals = null;
    let minibatches = 0;
    const order = Int32Array.from({ length: count }, (_, i) => i);
    const allObs = tf.tensor2d(batch.observations, [count, obsSize]);
    const allActions = tf.tensor2d(batch.actions, [count, actSize]);
    const allLogProbs = tf.tensor1d(batch.logProbs);
    const allAdvantages = tf.tensor1d(batch.advantages);
    const allReturns = tf.tensor1d(batch.returns);
    try {
      for (let epoch = 0; epoch < settings.epochs; epoch++) {
        for (let i = count - 1; i > 0; i--) {  // a fresh shuffle every round
          const j = Math.floor(random.random() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
        for (let start = 0; start < count; start += settings.batchSize) {
          const pick = order.slice(start, Math.min(start + settings.batchSize, count));
          const stats = this.studyMinibatch(pick, { allObs, allActions, allLogProbs, allAdvantages, allReturns }, settings);
          const added = totals ? tf.add(totals, stats) : stats;
          if (totals) tf.dispose([totals, stats]);
          totals = added;
          minibatches += 1;
        }
      }
      const [approxKl, policyLoss, valueLoss, entropy, clipFraction] = totals ? Array.from(totals.dataSync()) : [0, 0, 0, 0, 0];
      const n = Math.max(minibatches, 1);
      return { approx_kl: approxKl / n, policy_loss: policyLoss / n, value_loss: valueLoss / n, entropy: entropy / n, clip_fraction: clipFraction / n };
    } finally {
      tf.dispose([allObs, allActions, allLogProbs, allAdvantages, allReturns]);
      if (totals) tf.dispose(totals);
    }
  }

  studyMinibatch(pick, all, settings) {
    const tf = this.tf;
    const index = tf.tensor1d(pick, "int32");
    const observations = tf.gather(all.allObs, index);
    const actions = tf.gather(all.allActions, index);
    const oldLogProbs = tf.gather(all.allLogProbs, index);
    const returns = tf.gather(all.allReturns, index);
    // Normalised per minibatch, as Stable-Baselines3 does (PyTorch's std divides by n - 1). tidy() frees every
    // in-between tensor (numbers turned into tensors included); before, three were left behind per minibatch
    const advantages = tf.tidy(() => {
      const raw = tf.gather(all.allAdvantages, index);
      if (pick.length < 2) return raw;
      const moments = tf.moments(raw);
      const std = tf.sqrt(tf.mul(moments.variance, pick.length / (pick.length - 1)));
      return tf.div(tf.sub(raw, moments.mean), tf.add(std, 1e-8));
    });
    let kept = null;
    const lossFunction = () => {
      const { mean, value } = this.forward(observations);
      const logStd = this.logStdUsed();
      const z = tf.div(tf.sub(actions, mean), tf.exp(logStd));
      const logProbs = tf.sub(tf.mul(-0.5, tf.sum(tf.square(z), 1)), tf.add(tf.sum(logStd), 0.5 * this.motorCount * LOG_TWO_PI));
      const logRatio = tf.sub(logProbs, oldLogProbs);
      const ratio = tf.exp(logRatio);
      const clipped = tf.clipByValue(ratio, 1 - PPO_FIXED.clipRange, 1 + PPO_FIXED.clipRange);
      const policyLoss = tf.neg(tf.mean(tf.minimum(tf.mul(advantages, ratio), tf.mul(advantages, clipped))));
      const valueLoss = tf.mean(tf.square(tf.sub(returns, value)));
      const entropy = tf.add(tf.sum(logStd), 0.5 * this.motorCount * (1 + LOG_TWO_PI));  // the same for every sample
      const loss = tf.add(tf.add(policyLoss, tf.mul(-(settings.entropyWeight || 0), entropy)), tf.mul(PPO_FIXED.valueWeight, valueLoss));
      // approx_kl, policy loss, value loss, entropy and clip fraction, as one small tensor (read later, all at once)
      kept = tf.keep(tf.stack([
        tf.mean(tf.sub(tf.sub(ratio, 1), logRatio)), policyLoss, valueLoss, entropy,
        tf.mean(tf.cast(tf.greater(tf.abs(tf.sub(ratio, 1)), PPO_FIXED.clipRange), "float32")),
      ]));
      return loss;
    };
    const { value, grads } = tf.variableGrads(lossFunction, this.variables());
    const clippedGrads = this.clipGradients(grads);
    this.optimizer.applyGradients(clippedGrads);
    tf.dispose([value, index, observations, actions, oldLogProbs, returns, advantages, ...Object.values(grads), ...Object.values(clippedGrads)]);
    return kept;
  }

  // Scale all gradients down together when their total length is above maxGradNorm (torch's clip_grad_norm_)
  clipGradients(grads) {
    const tf = this.tf;
    return tf.tidy(() => {
      const names = Object.keys(grads);
      const total = tf.sqrt(tf.addN(names.map((name) => tf.sum(tf.square(grads[name])))));
      const scale = tf.minimum(tf.div(PPO_FIXED.maxGradNorm, tf.add(total, 1e-6)), 1.0);
      return Object.fromEntries(names.map((name) => [name, tf.mul(grads[name], scale)]));
    });
  }

  setLearningRate(rate) {
    this.learningRate = rate;
    this.optimizer.learningRate = rate;
  }

  // ---------- Handing the weights over ----------

  // The brain as plain data for network.js (PyTorch layout), without the sense scaling
  toBrain() {
    const net = (n) => {
      const layer = (l) => ({ w: transpose(l.w.dataSync(), l.inputs, l.outputs), b: Float32Array.from(l.b.dataSync()), inputs: l.inputs, outputs: l.outputs });
      return { layers: n.layers.map(layer), head: layer(n.head) };
    };
    const brain = { kind: this.kind, activation: this.activation, actor: net(this.actor), critic: net(this.critic), logStd: Float32Array.from(this.logStd.dataSync()) };
    if (this.mirror) brain.mirror = this.mirror;
    return brain;
  }

  // Load weights from plain brain data (a saved brain, or one brought over from the Python studio)
  loadBrain(brain) {
    const tf = this.tf;
    const net = (mine, theirs, label) => {
      const pairs = [...mine.layers.map((l, i) => [l, theirs.layers[i]]), [mine.head, theirs.head]];
      for (const [layer, saved] of pairs) {
        if (!saved || saved.inputs !== layer.inputs || saved.outputs !== layer.outputs) throw new Error(`The saved ${label} does not fit this brain.`);
        layer.w.assign(tf.tensor2d(transpose(Float32Array.from(saved.w), saved.outputs, saved.inputs), [layer.inputs, layer.outputs]));
        layer.b.assign(tf.tensor1d(Float32Array.from(saved.b)));
      }
    };
    net(this.actor, brain.actor, "brain");
    net(this.critic, brain.critic, "judge");
    this.logStd.assign(tf.tensor1d(Float32Array.from(brain.logStd)));
  }

  // The optimizer's memory (Adam's running averages), so a paused training carries on smoothly
  // (saved without this learner's name prefix, so another learner can load it)
  async optimizerState() {
    const weights = await this.optimizer.getWeights();
    return Promise.all(weights.map(async ({ name, tensor }) => ({ name: name.split(this.prefix).join(""), shape: tensor.shape,
      values: Float32Array.from(await tensor.data()) })));
  }

  async loadOptimizerState(saved) {
    if (!saved || !saved.length) return;
    const tf = this.tf;
    const withPrefix = (name) => (name === "iter" || name.includes("/") === false ? name : `${this.prefix}${name}`);
    await this.optimizer.setWeights(saved.map(({ name, shape, values }) => ({ name: withPrefix(name), tensor: tf.tensor(values, shape) })));
  }

  dispose() {
    this.tf.dispose(this.variables());
    this.optimizer.dispose();
    if (this.obsIndex) this.tf.dispose([this.obsIndex, this.obsSign, this.actIndex, this.actSign, this.placeIndex]);
  }
}
