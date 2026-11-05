// The brain's networks in plain JavaScript, for picking moves fast while practising and watching.
// Learning itself (gradients) happens in learner.js on the graphics card; it hands its weights to these.
//
// A brain is plain data (it can be saved and sent to workers):
//   { kind: "free" | "symmetric", activation, actor: { layers: [{ w, b, inputs, outputs }], head },
//     critic: { layers, head }, logStd: [...], mirror: tables (symmetric only), scaling: { mean, var, clip } }
// Weights are Float32Arrays, row by row: w[o * inputs + i] links input i to output o (as PyTorch stores them).
// The symmetric brain is learn_to_walk's NET method (a port of brain.MirrorExtractor): one network reads the
// situation and its mirror image; its answers drive the left side, its answers on the mirrored view the right side,
// and the middle motors get both answers combined.

export const CELL_SAMPLE = 24;  // how many cells of each hidden layer a replay keeps

const ACTIVATIONS = {
  tanh: Math.tanh,
  relu: (x) => (x > 0 ? x : 0),
  elu: (x) => (x > 0 ? x : Math.exp(x) - 1),
  silu: (x) => x / (1 + Math.exp(-x)),
};

// One linear layer: out = w x + b
function linear(layer, input) {
  const { w, b, inputs, outputs } = layer;
  const out = new Float32Array(outputs);
  for (let o = 0; o < outputs; o++) {
    let sum = b[o];
    const row = o * inputs;
    for (let i = 0; i < inputs; i++) sum += w[row + i] * input[i];
    out[o] = sum;
  }
  return out;
}

// The hidden layers (each followed by the cell response), keeping each layer's cells when asked
function trunk(layers, input, activation, keep = null) {
  const respond = ACTIVATIONS[activation] || Math.tanh;
  let value = input;
  for (const layer of layers) {
    value = linear(layer, value);
    for (let i = 0; i < value.length; i++) value[i] = respond(value[i]);
    if (keep) keep.push(value);
  }
  return value;
}

// ---------- The mirror ----------

// The observation as its mirror image sees it
function mirrorObservation(mirror, observation) {
  const order = mirror.observation_order, signs = mirror.observation_signs;
  const out = new Float32Array(order.length);
  for (let i = 0; i < order.length; i++) out[i] = observation[order[i]] * signs[i];
  return out;
}

function concat(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

// Each motor's command from one pass of the head: its own output (left motors, then middle ones), or 0
function place(mirror, outputs, motorCount) {
  const placed = new Float32Array(motorCount);
  const own = [...mirror.left_motors, ...mirror.middle_motors];
  own.forEach((motor, position) => { placed[motor] = outputs[position]; });
  return placed;
}

// ---------- Acting ----------

// The sense scaling learned in training: (x - mean) / sqrt(var + 1e-8), clipped
export function scaleObservation(brain, observation) {
  const { mean, var: variance, clip } = brain.scaling;
  const out = new Float32Array(observation.length);
  for (let i = 0; i < observation.length; i++) {
    const value = (observation[i] - mean[i]) / Math.sqrt(variance[i] + 1e-8);
    out[i] = Math.min(Math.max(value, -clip), clip);
  }
  return out;
}

// The brain's best move (the middle of its tries) for an already scaled observation
export function meanAction(brain, scaled) {
  if (brain.kind !== "symmetric") return linear(brain.actor.head, trunk(brain.actor.layers, scaled, brain.activation));
  const mirror = brain.mirror;
  const motorCount = mirror.action_order.length;
  const mirrored = mirrorObservation(mirror, scaled);
  const real = linear(brain.actor.head, trunk(brain.actor.layers, concat(scaled, mirrored), brain.activation));
  const flipped = linear(brain.actor.head, trunk(brain.actor.layers, concat(mirrored, scaled), brain.activation));
  const fromReal = place(mirror, real, motorCount), fromMirror = place(mirror, flipped, motorCount);
  const action = new Float32Array(motorCount);
  for (let m = 0; m < motorCount; m++) action[m] = fromReal[m] + fromMirror[mirror.action_order[m]] * mirror.action_signs[m];
  return action;
}

// The judge's guess of the points to come, for an already scaled observation
export function judgeValue(brain, scaled) {
  if (brain.kind !== "symmetric") return linear(brain.critic.head, trunk(brain.critic.layers, scaled, brain.activation))[0];
  const mirrored = mirrorObservation(brain.mirror, scaled);
  const real = linear(brain.critic.head, trunk(brain.critic.layers, concat(scaled, mirrored), brain.activation))[0];
  const flipped = linear(brain.critic.head, trunk(brain.critic.layers, concat(mirrored, scaled), brain.activation))[0];
  return real + flipped;
}

// The spread of tries per motor. The symmetric brain wobbles each motor like its mirror partner
export function actionStd(brain) {
  const logStd = brain.logStd;
  if (brain.kind !== "symmetric") return Float32Array.from(logStd, Math.exp);
  const order = brain.mirror.action_order;
  return Float32Array.from(logStd, (value, m) => Math.exp((value + logStd[order[m]]) / 2));
}

// A move with random wobble (for practising): the mean plus a bell curve try for each motor
export function sampleAction(brain, scaled, random, std = actionStd(brain)) {
  const mean = meanAction(brain, scaled);
  const action = new Float32Array(mean.length);
  for (let m = 0; m < mean.length; m++) action[m] = mean[m] + std[m] * random.normal();
  return { action, mean };
}

// A sample of every hidden layer's cells for one scaled observation (the "watch it think" panel)
export function brainCells(brain, scaled, sample = CELL_SAMPLE) {
  const input = brain.kind === "symmetric" ? concat(scaled, mirrorObservation(brain.mirror, scaled)) : scaled;
  const kept = [];
  trunk(brain.actor.layers, input, brain.activation, kept);
  return kept.map((row) => {
    const count = Math.min(sample, row.length);
    const picks = Array.from({ length: count }, (_, k) => Math.round((k * (row.length - 1)) / Math.max(count - 1, 1)));
    return picks.map((index) => Math.round(row[index] * 100) / 100);
  });
}

// A ready-to-use mover for recordings and exams: raw observation in, best move out
export function deterministicActor(brain) {
  return (observation) => meanAction(brain, scaleObservation(brain, observation));
}

export function cellReader(brain) {
  return { cells: (observation) => brainCells(brain, scaleObservation(brain, observation)) };
}
