// PPO's bookkeeping, in plain JavaScript: running averages for scaling the senses and the points (as
// Stable-Baselines3's VecNormalize does), and GAE, which turns the points of a practice run into "how much better
// than expected was each move". A port of what train.py gets from Stable-Baselines3.

export const CLIP_OBSERVATION = 10.0;
const CLIP_REWARD = 10.0;
const EPSILON = 1e-8;

// A running mean and variance over many batches (Stable-Baselines3's RunningMeanStd).
// order/signs (optional): learn from every observation and its mirror image together (mirror.MirroredRunningMeanStd)
export class RunningMeanStd {
  constructor(size, { order = null, signs = null, epsilon = 1e-4 } = {}) {
    this.size = size;
    this.mean = new Float64Array(size);
    this.var = new Float64Array(size).fill(1);
    this.count = epsilon;
    this.order = order;
    this.signs = signs;
  }

  // rows: an array of vectors (or one flat array with `size` numbers per row)
  update(rows) {
    const list = ArrayBuffer.isView(rows) ? splitRows(rows, this.size) : rows;
    const all = this.order ? [...list, ...list.map((row) => mirrorRow(row, this.order, this.signs))] : list;
    const n = all.length;
    if (!n) return;
    const batchMean = new Float64Array(this.size), batchVar = new Float64Array(this.size);
    for (const row of all) for (let i = 0; i < this.size; i++) batchMean[i] += row[i] / n;
    for (const row of all) for (let i = 0; i < this.size; i++) batchVar[i] += (row[i] - batchMean[i]) ** 2 / n;
    const total = this.count + n;
    for (let i = 0; i < this.size; i++) {
      const delta = batchMean[i] - this.mean[i];
      const m2 = this.var[i] * this.count + batchVar[i] * n + (delta ** 2 * this.count * n) / total;
      this.mean[i] += (delta * n) / total;
      this.var[i] = m2 / total;
    }
    this.count = total;
  }

  toData() {
    return { mean: Array.from(this.mean), var: Array.from(this.var), count: this.count };
  }

  static fromData(data, options = {}) {
    const rms = new RunningMeanStd(data.mean.length, options);
    rms.mean = Float64Array.from(data.mean);
    rms.var = Float64Array.from(data.var);
    rms.count = data.count;
    return rms;
  }
}

function splitRows(flat, size) {
  const rows = [];
  for (let start = 0; start < flat.length; start += size) rows.push(flat.subarray(start, start + size));
  return rows;
}

function mirrorRow(row, order, signs) {
  const out = new Float64Array(order.length);
  for (let i = 0; i < order.length; i++) out[i] = row[order[i]] * signs[i];
  return out;
}

// The sense scaling as brains use it (network.scaleObservation)
export function scalingOf(obsRms) {
  return { mean: Float32Array.from(obsRms.mean), var: Float32Array.from(obsRms.var), clip: CLIP_OBSERVATION };
}

// ---------- Scaling the points ----------

// Points are divided by the spread of the running discounted return, as VecNormalize does, step by step:
// rewards[t][e] for every step t and world e. Returns the scaled points (same shape) and updates `state`
export function normalizeRewards(rewards, dones, state, gamma) {
  const scaled = [];
  for (let t = 0; t < rewards.length; t++) {
    const row = rewards[t];
    for (let e = 0; e < row.length; e++) state.returns[e] = state.returns[e] * gamma + row[e];
    state.retRms.update([Float64Array.from(state.returns)]);
    const spread = Math.sqrt(state.retRms.var[0] + EPSILON);
    scaled.push(row.map((r) => Math.min(Math.max(r / spread, -CLIP_REWARD), CLIP_REWARD)));
    dones[t].forEach((done, e) => { if (done) state.returns[e] = 0; });
  }
  return scaled;
}

// The running return per world and its running spread (a one-number RunningMeanStd)
export function rewardState(worlds, saved = null) {
  return {
    returns: new Float64Array(worlds),
    retRms: saved ? RunningMeanStd.fromData(saved) : new RunningMeanStd(1),
  };
}

// ---------- GAE ----------

// Advantages and returns for [step][world] arrays (Stable-Baselines3's compute_returns_and_advantage).
// episodeStarts[t][e]: world e began a new attempt at step t. lastValues/lastDones: after the last step
export function advantagesAndReturns({ rewards, values, episodeStarts, lastValues, lastDones, gamma, lambda }) {
  const steps = rewards.length, worlds = rewards[0].length;
  const advantages = Array.from({ length: steps }, () => new Float64Array(worlds));
  const returns = Array.from({ length: steps }, () => new Float64Array(worlds));
  for (let e = 0; e < worlds; e++) {
    let last = 0;
    for (let t = steps - 1; t >= 0; t--) {
      const nextNonTerminal = t === steps - 1 ? 1 - lastDones[e] : 1 - episodeStarts[t + 1][e];
      const nextValue = t === steps - 1 ? lastValues[e] : values[t + 1][e];
      const delta = rewards[t][e] + gamma * nextValue * nextNonTerminal - values[t][e];
      last = delta + gamma * lambda * nextNonTerminal * last;
      advantages[t][e] = last;
      returns[t][e] = last + values[t][e];
    }
  }
  return { advantages, returns };
}

// The batch size PPO uses: it must divide one lesson's practice (learning.ppo_settings)
export function fittedBatchSize(worlds, steps, batch) {
  while ((worlds * steps) % batch && batch > 32) batch = Math.floor(batch / 2);
  return batch;
}

// ---------- The adaptive learning speed (learn_to_walk's rule) ----------

export const STEP_SIZE_RULE = { targetChange: 0.01, factor: 1.5, min: 1e-5, max: 1e-2 };

export function adjustedLearningRate(rate, approxKl) {
  if (approxKl === null || approxKl === undefined) return rate;
  if (approxKl > 2 * STEP_SIZE_RULE.targetChange) rate /= STEP_SIZE_RULE.factor;
  else if (approxKl < STEP_SIZE_RULE.targetChange / 2) rate *= STEP_SIZE_RULE.factor;
  return Math.min(Math.max(rate, STEP_SIZE_RULE.min), STEP_SIZE_RULE.max);
}
