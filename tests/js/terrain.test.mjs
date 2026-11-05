// The ground: every terrain makes the same ground as the recorded one (terrain_snapshot.mjs), in standard floats
// (half the memory of doubles), and a cobblestone floor of a real attempt is made quickly.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { floorLayout, groundHeights } from "../../studio/web/js/engine/terrain.js";
import { Random } from "../../studio/web/js/engine/random.js";
import { TERRAINS, smallLayout } from "./terrain_snapshot.mjs";

const snapshot = JSON.parse(readFileSync(new URL("./fixtures/terrain.json", import.meta.url), "utf8"));

for (const terrain of TERRAINS) {
  test(`${terrain} makes the recorded ground, in standard floats`, () => {
    const heights = groundHeights(terrain, new Random(42), smallLayout(), 0.05);
    assert.ok(heights instanceof Float32Array, "heights should be a Float32Array");
    assert.equal(heights.length, snapshot[terrain].length);
    let worst = 0;
    heights.forEach((value, i) => { worst = Math.max(worst, Math.abs(value - snapshot[terrain][i])); });
    assert.ok(worst < 1e-5, `largest difference ${worst} m`);
  });
}

test("a full cobblestone floor (5 mm cells, 2 million heights) is made in well under a second", () => {
  const layout = floorLayout("cobblestones", 0.7, 20, 1.0);
  const start = performance.now();
  groundHeights("cobblestones", new Random(1), layout, 0.03);
  const seconds = (performance.now() - start) / 1000;
  assert.ok(seconds < 0.25, `took ${seconds.toFixed(2)} s`);
});
