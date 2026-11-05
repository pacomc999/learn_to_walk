// The shape of the ground: height maps for every kind of terrain. A port of studio/terrain.py.
// A height map is a Float64Array of rows x columns heights in metres (rows along y, columns along x, row by row).
// The walker starts at x = 0 and walks toward +x, so every floor reaches far enough ahead for the fastest speed.

import { pyRound } from "./random.js";

// The cobblestones: rolling bumps with square stones on top (learn_to_walk's FLOOR_SHAPE)
const COBBLE_SHAPE = { waves: 8, wavelengthRange: [0.2, 0.5], stoneSize: 0.06, stoneHeightRange: [0.35, 1.0], gapShare: 0.07, bumpShare: 0.5 };

// How finely each terrain is sampled, in metres. Small stones need small cells, hills do not
const CELL_SIZES = { cobblestones: 0.005, blocks: 0.01, stairs: 0.01, craters: 0.01, hills: 0.02, hurdles: 0.01, slopes: 0.02 };

export const SLOPE_LENGTH = 5.0;      // climbs and descents this long; their height is terrain height per metre
export const MOST_STEPS = 6;          // the most steps a flight of stairs climbs
const STAIRS_RUN_UP = 1.0;            // the flat run-up before the first step or hurdle
const STEP_DEPTH_RANGE = [0.25, 0.4];

// ---------- Where a floor lies ----------

export class FloorLayout {
  constructor(xMin, xMax, yHalf, cell) {
    Object.assign(this, { xMin, xMax, yHalf, cell });
    this.columns = pyRound((xMax - xMin) / cell) + 1;
    this.rows = pyRound((2 * yHalf) / cell) + 1;
    this.columnX = linspace(xMin, xMax, this.columns);
    this.rowY = linspace(-yHalf, yHalf, this.rows);
  }

  // The (row, column) of the cell under a point, kept inside the grid
  index(x, y) {
    const column = pyRound((x - this.xMin) / this.cell);
    const row = pyRound((y + this.yHalf) / this.cell);
    return [clamp(row, 0, this.rows - 1), clamp(column, 0, this.columns - 1)];
  }
}

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);

function linspace(start, stop, count) {
  const values = new Float64Array(count);
  const step = count > 1 ? (stop - start) / (count - 1) : 0;
  for (let i = 0; i < count; i++) values[i] = start + i * step;
  if (count > 1) values[count - 1] = stop;
  return values;
}

// The floor for a terrain, long enough for the fastest asked speed over a whole attempt
export function floorLayout(terrain, fastestSpeed, seconds, size) {
  const reach = Math.max(16.0, fastestSpeed * seconds * 1.15 + 3.0);
  return new FloorLayout(-2.0, Math.round(reach * 10) / 10, Math.max(1.0, size), CELL_SIZES[terrain] ?? 0.01);
}

// The tallest point a terrain can reach when its bumps are `tallest` metres (stairs pile up)
export function capacity(terrain, tallest) {
  if (terrain === "stairs") return tallest * MOST_STEPS;
  if (terrain === "slopes") return tallest * SLOPE_LENGTH;
  return tallest;
}

// ---------- Helpers on grids ----------

// Scale values so the lowest becomes 0 and the highest becomes 1 (in place)
function toUnit(values) {
  let low = Infinity, high = -Infinity;
  for (const value of values) { if (value < low) low = value; if (value > high) high = value; }
  const spread = high - low;
  for (let i = 0; i < values.length; i++) values[i] = spread > 0 ? (values[i] - low) / spread : 0;
  return values;
}

// One profile along x, the same in every row
function tile(profile, rows) {
  const grid = new Float64Array(rows * profile.length);
  for (let r = 0; r < rows; r++) grid.set(profile, r * profile.length);
  return grid;
}

// ---------- Cobblestones (from learn_to_walk) ----------

// A sum of plane waves: cos(2 pi (x cos a + y sin a) / wavelength + phase)
function rollingBumps(rng, layout, [shortest, longest], waves, maxAngle = Math.PI) {
  const grid = new Float64Array(layout.rows * layout.columns);
  for (let w = 0; w < waves; w++) {
    const wavelength = rng.uniform(shortest, longest);
    const angle = rng.uniform(0, maxAngle) - (maxAngle < Math.PI ? maxAngle / 2 : 0);
    const phase = rng.uniform(0, 2 * Math.PI);
    const kx = (2 * Math.PI * Math.cos(angle)) / wavelength, ky = (2 * Math.PI * Math.sin(angle)) / wavelength;
    const alongX = Float64Array.from(layout.columnX, (x) => kx * x + phase);
    for (let r = 0; r < layout.rows; r++) {
      const alongY = ky * layout.rowY[r];
      const offset = r * layout.columns;
      for (let c = 0; c < layout.columns; c++) grid[offset + c] += Math.cos(alongY + alongX[c]);
    }
  }
  return grid;
}

function stoneIndexAndGap(positions, start, stone, gapShare) {
  const index = new Int32Array(positions.length), gap = new Uint8Array(positions.length);
  positions.forEach((position, i) => {
    const place = (position - start) / stone;
    index[i] = Math.floor(place);
    const inside = place - index[i];
    gap[i] = inside < gapShare || inside > 1 - gapShare ? 1 : 0;
  });
  return [index, gap];
}

function squareStones(rng, layout, stone, [lowest, highest], gapShare) {
  const [columnStone, columnGap] = stoneIndexAndGap(layout.columnX, layout.xMin, stone, gapShare);
  const [rowStone, rowGap] = stoneIndexAndGap(layout.rowY, -layout.yHalf, stone, gapShare);
  const stoneColumns = Math.max(...columnStone) + 1, stoneRows = Math.max(...rowStone) + 1;
  const heights = Float64Array.from({ length: stoneRows * stoneColumns }, () => rng.uniform(lowest, highest));
  const grid = new Float64Array(layout.rows * layout.columns);
  for (let r = 0; r < layout.rows; r++) {
    for (let c = 0; c < layout.columns; c++) {
      grid[r * layout.columns + c] = rowGap[r] || columnGap[c] ? 0 : heights[rowStone[r] * stoneColumns + columnStone[c]];
    }
  }
  return grid;
}

function cobblestones(rng, layout) {
  const shape = COBBLE_SHAPE;
  const bumps = toUnit(rollingBumps(rng, layout, shape.wavelengthRange, shape.waves));
  const stones = toUnit(squareStones(rng, layout, shape.stoneSize, shape.stoneHeightRange, shape.gapShare));
  return toUnit(bumps.map((bump, i) => shape.bumpShare * bump + (1 - shape.bumpShare) * stones[i]));
}

// ---------- The other terrains ----------

// Long smooth hills, mostly across the walking line so the walker goes up and down
function hills(rng, layout) {
  return toUnit(rollingBumps(rng, layout, [2.0, 6.0], 4, 0.7));
}

// Set profile[x >= position] = value
function fillFrom(profile, xs, position, value) {
  for (let i = 0; i < xs.length; i++) if (xs[i] >= position) profile[i] = value;
}

// Flights of steps up and down, with flat landings between them, the same all across
function stairs(rng, layout) {
  const xs = layout.columnX;
  const profile = new Float64Array(xs.length);
  let position = STAIRS_RUN_UP, level = 0.0;
  while (position < layout.xMax) {
    const steps = rng.integers(3, MOST_STEPS + 1);
    const rises = Array.from({ length: steps }, () => rng.uniform(0.6, 1.0));
    const depth = rng.uniform(...STEP_DEPTH_RANGE);
    for (const direction of [1, -1]) {
      for (const rise of rises) {
        level = Math.max(level + direction * rise, 0.0);
        fillFrom(profile, xs, position, level);
        position += depth;
      }
      position += rng.uniform(0.6, 1.2);  // landing
    }
  }
  return tile(profile.map((value) => value / MOST_STEPS), layout.rows);  // a share of what the floor holds
}

// The first index whose value is not below `value` (NumPy's searchsorted)
function searchSorted(values, value) {
  let low = 0, high = values.length;
  while (low < high) { const mid = (low + high) >> 1; if (values[mid] < value) low = mid + 1; else high = mid; }
  return low;
}

// A dusty plain with round craters (bowls with a raised rim) and a little roughness
function craters(rng, layout) {
  const xs = layout.columnX, ys = layout.rowY;
  const ground = new Float64Array(layout.rows * layout.columns);
  const area = (layout.xMax - layout.xMin) * 2 * layout.yHalf;
  for (let n = 0; n < Math.floor(area / 2.5); n++) {
    const centreX = rng.uniform(layout.xMin, layout.xMax);
    const centreY = rng.uniform(-layout.yHalf, layout.yHalf);
    const radius = rng.uniform(0.25, 0.8);
    const depth = rng.uniform(0.4, 1.0);
    const reach = 1.6 * radius;  // only the cells near this crater change
    const [c0, c1] = [searchSorted(xs, centreX - reach), searchSorted(xs, centreX + reach)];
    const [r0, r1] = [searchSorted(ys, centreY - reach), searchSorted(ys, centreY + reach)];
    for (let r = r0; r < r1; r++) {
      for (let c = c0; c < c1; c++) {
        const distance = Math.hypot(xs[c] - centreX, ys[r] - centreY);
        const bowl = distance < radius ? depth * (1 - (distance / radius) ** 2) : 0.0;
        const rim = 0.25 * depth * Math.exp(-(((distance - radius) / (0.18 * radius)) ** 2));
        ground[r * layout.columns + c] += rim - bowl;
      }
    }
  }
  const dust = toUnit(rollingBumps(rng, layout, [0.3, 0.8], 6));
  return toUnit(ground.map((value, i) => value + 0.15 * dust[i]));
}

// Big flat topped rocks of random heights, side by side
function blocks(rng, layout) {
  const size = rng.uniform(0.18, 0.32);
  const columnBlock = Int32Array.from(layout.columnX, (x) => Math.floor((x - layout.xMin) / size));
  const rowBlock = Int32Array.from(layout.rowY, (y) => Math.floor((y + layout.yHalf) / size));
  const blockColumns = Math.max(...columnBlock) + 1, blockRows = Math.max(...rowBlock) + 1;
  const heights = Float64Array.from({ length: blockRows * blockColumns }, () => rng.uniform(0.0, 1.0));
  for (let i = 0; i < heights.length; i++) if (rng.random() < 0.25) heights[i] = 0.0;
  const grid = new Float64Array(layout.rows * layout.columns);
  for (let r = 0; r < layout.rows; r++) {
    for (let c = 0; c < layout.columns; c++) grid[r * layout.columns + c] = heights[rowBlock[r] * blockColumns + columnBlock[c]];
  }
  return grid;
}

// Low bars across the path to step over, about a metre apart, after a flat run-up
function hurdles(rng, layout) {
  const xs = layout.columnX;
  const profile = new Float64Array(xs.length);
  let position = STAIRS_RUN_UP;
  while (position < layout.xMax) {
    const width = rng.uniform(0.03, 0.06);
    const height = rng.uniform(0.6, 1.0);
    for (let i = 0; i < xs.length; i++) if (xs[i] >= position && xs[i] < position + width) profile[i] = height;
    position += rng.uniform(0.8, 1.3);
  }
  return tile(profile, layout.rows);
}

// Long even climbs and descents with flat tops, like a mountain path. The height setting is the rise per metre
function slopes(rng, layout) {
  const xs = layout.columnX;
  const profile = new Float64Array(xs.length);
  const between = (start, end, value) => { for (let i = 0; i < xs.length; i++) if (xs[i] >= start && xs[i] < end) profile[i] = value(xs[i]); };
  let position = STAIRS_RUN_UP, level = 0.0;
  while (position < layout.xMax) {
    const length = rng.uniform(0.6, 1.0) * SLOPE_LENGTH;
    const climbStart = position, climbLevel = level;
    between(position, position + length, (x) => climbLevel + (x - climbStart) / SLOPE_LENGTH);
    position += length;
    level += length / SLOPE_LENGTH;
    const flat = rng.uniform(0.8, 1.5);
    const topLevel = level;
    between(position, position + flat, () => topLevel);
    position += flat;
    const descentStart = position;
    between(position, position + length, (x) => topLevel - (x - descentStart) / SLOPE_LENGTH);
    position += length;
    level = Math.max(level - length / SLOPE_LENGTH, 0.0);
    fillFrom(profile, xs, position, level);
    position += rng.uniform(0.8, 1.5);
  }
  return tile(profile.map((value) => clamp(value, 0.0, 1.0)), layout.rows);
}

const SHAPES = { cobblestones, hills, stairs, craters, blocks, hurdles, slopes };

// Heights in metres for one attempt: the terrain's shape scaled to this attempt's tallest bumps
export function groundHeights(terrain, rng, layout, height) {
  if (height <= 0 || !(terrain in SHAPES)) return new Float64Array(layout.rows * layout.columns);
  const scale = capacity(terrain, height);
  return SHAPES[terrain](rng, layout).map((value) => value * scale);
}

// The highest ground in the square of half size `radius` around (x, y)
export function highestNear(heights, layout, x, y, radius) {
  const [firstRow, firstColumn] = layout.index(x - radius, y - radius);
  const [lastRow, lastColumn] = layout.index(x + radius, y + radius);
  let highest = -Infinity;
  for (let r = firstRow; r <= lastRow; r++) {
    for (let c = firstColumn; c <= lastColumn; c++) highest = Math.max(highest, heights[r * layout.columns + c]);
  }
  return highest;
}

// The ground height under one point
export function heightAt(heights, layout, x, y) {
  const [row, column] = layout.index(x, y);
  return heights[row * layout.columns + column];
}
