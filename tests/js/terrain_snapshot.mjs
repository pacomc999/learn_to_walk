// Records the ground every terrain makes on a small floor with a fixed seed (tests/js/fixtures/terrain.json),
// so a faster or leaner terrain.js can be checked to make the same ground. Run: node tests/js/terrain_snapshot.mjs
import { writeFileSync } from "node:fs";
import { FloorLayout, groundHeights } from "../../studio/web/js/engine/terrain.js";
import { Random } from "../../studio/web/js/engine/random.js";

export const TERRAINS = ["cobblestones", "hills", "stairs", "craters", "blocks", "hurdles", "slopes"];
export const smallLayout = () => new FloorLayout(-2.0, 6.0, 1.0, 0.02);  // 401 columns x 101 rows

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, "/")}`) {
  const snapshot = {};
  for (const terrain of TERRAINS) snapshot[terrain] = Array.from(groundHeights(terrain, new Random(42), smallLayout(), 0.05));
  writeFileSync(new URL("./fixtures/terrain.json", import.meta.url), JSON.stringify(snapshot));
  console.log("Wrote", Object.keys(snapshot).length, "terrains of", snapshot.hills.length, "heights");
}
