// Brain designs (layers, cell response, senses, starting wobble) and the left/right mirror tables.
// A port of the design part of studio/brain.py and of studio/mirror.py. The networks are in network.js.

import { brainGroups } from "./catalog.js";
import { GROUND_SCAN } from "./env.js";
import { cleanValues, defaultValues } from "./kit.js";

export const DEFAULT_LAYERS = [256, 256];
// The starting wobble as the log of the spread of random tries per motor
export const START_WOBBLE = { tiny: -3.0, calm: -1.0, lively: -0.5, wild: 0.0 };
// "Matched to the muscles": Duck 4's wobble for direct muscles (0.05), learn_to_walk's -1 for springy ones
export const AUTO_WOBBLE = { force: -3.0, position: -1.0 };

function cleanLayers(layers, fallback) {
  if (!Array.isArray(layers) || !layers.length) return [...fallback];
  const cleaned = [];
  for (const size of layers.slice(0, 6)) {
    const n = Math.trunc(Number(size));
    if (Number.isFinite(n)) cleaned.push(Math.min(Math.max(n, 4), 1024));
  }
  return cleaned.length ? cleaned : [...fallback];
}

// A brain design with every value checked
export function cleanDesign(design) {
  design = design || {};
  const values = cleanValues(brainGroups(), design.values || {});
  const layers = cleanLayers(design.layers, DEFAULT_LAYERS);
  const judge = values.judge_same ? layers : cleanLayers(design.judge_layers, layers);
  return { values, layers, judge_layers: judge };
}

export function defaultDesign() {
  return { values: defaultValues(brainGroups()), layers: [...DEFAULT_LAYERS], judge_layers: [...DEFAULT_LAYERS] };
}

// A body with unpaired parts off the middle line cannot have the symmetric brain, so it gets the free one
export function designForBody(design, build) {
  if (build.symmetricOk || !design.values.symmetric) return design;
  return { ...design, values: { ...design.values, symmetric: false } };
}

// The senses switched on by a design, in the environment's words
export function designSenses(design) {
  const v = design.values;
  return { feet: v.feel_feet, ground: v.see_ground, turn: v.feel_turn };
}

// How many numbers the brain learns (both networks), for the Brain screen
export function countWeights(observationSize, motorCount, design) {
  const inputs = observationSize * (design.values.symmetric ? 2 : 1);
  const stack = (sizes, outputs) => {
    let total = 0, previous = inputs;
    for (const size of sizes) { total += previous * size + size; previous = size; }
    return total + previous * outputs + outputs;
  };
  const brain = stack(design.layers, motorCount);
  const judge = stack(design.judge_layers, 1);
  return { brain, judge, total: brain + judge };
}

// The log of the starting spread of tries, from the design and the kind of muscle
export function startLogStd(design, motorType) {
  const wobble = design.values.start_wobble;
  return wobble === "auto" ? AUTO_WOBBLE[motorType] ?? -1.0 : START_WOBBLE[wobble];
}

// ---------- Mirror tables ----------

// Torso senses that keep their place: height (same), sideways position (flips),
// orientation (w, -x, y, -z), linear velocity (vx, -vy, vz), angular velocity (-wx, wy, -wz)
const BODY_SIGNS = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1];

// Which sense or motor command of the real walker each one of its mirror image is, and with which sign
export function mirrorTables(build, senses) {
  const names = build.motorNames;
  const actionOrder = build.motors.map((m) => names.indexOf(m.mirror));
  const actionSigns = build.motors.map((m) => m.mirrorSign);
  const count = names.length;
  const order = [...Array(12).keys()], signs = [...BODY_SIGNS];
  order.push(...actionOrder.map((i) => 12 + i)); signs.push(...actionSigns);
  order.push(...actionOrder.map((i) => 12 + count + i)); signs.push(...actionSigns);
  let position = 12 + 2 * count;
  if (senses.feet ?? true) {
    const feet = build.feet.map((f) => f.name);
    order.push(...build.feet.map((f) => position + feet.indexOf(f.mirror)));
    signs.push(...feet.map(() => 1));
    position += feet.length;
  }
  if (senses.ground ?? false) {  // the ground ahead lies on the middle line
    order.push(...GROUND_SCAN.map((_, i) => position + i));
    signs.push(...GROUND_SCAN.map(() => 1));
    position += GROUND_SCAN.length;
  }
  if (senses.turn ?? false) {  // a turn to the left is a turn to the right in the mirror
    order.push(position, position + 1, position + 2);
    signs.push(-1, 1, -1);
    position += 3;
  }
  order.push(position);  // the asked speed stays
  signs.push(1);
  return {
    action_order: actionOrder, action_signs: actionSigns, observation_order: order, observation_signs: signs,
    left_motors: build.motors.flatMap((m, i) => (m.side === "left" ? [i] : [])),
    middle_motors: build.motors.flatMap((m, i) => (m.side === "middle" ? [i] : [])),
  };
}
