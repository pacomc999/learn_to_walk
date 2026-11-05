// The JavaScript scenes match the Python studio: the same MuJoCo file, stand height, torso top and mass.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { useCatalog } from "../../studio/web/js/engine/catalog.js";
import { buildCreature } from "../../studio/web/js/engine/creatures/index.js";
import { loadPhysics } from "../../studio/web/js/engine/physics.js";
import { buildScene, disposeScene } from "../../studio/web/js/engine/scene.js";
import { makeWorld } from "../../studio/web/js/engine/worlds.js";

const read = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
useCatalog(read("../../studio/web/data/catalog.json"));
const cases = read("./fixtures/scenes.json");
const mujoco = await loadPhysics();

cases.forEach((item, index) => {
  test(`scene ${index} (${item.animal}, ${item.world.terrain || "flat"}) matches Python`, () => {
    const build = buildCreature(item.animal, item.values);
    const scene = buildScene(mujoco, build, makeWorld({ values: item.world }));
    try {
      assert.equal(scene.xml, item.xml);
      assert.ok(Math.abs(scene.standHeight - item.standHeight) < 1e-9, `stand height ${scene.standHeight} vs ${item.standHeight}`);
      assert.ok(Math.abs(scene.torsoTop - item.torsoTop) < 1e-9, "torso top");
      assert.ok(Math.abs(scene.totalMass - item.totalMass) < 1e-9, "total mass");
    } finally {
      disposeScene(scene);
    }
  });
});
