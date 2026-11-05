// The JavaScript practice world matches the Python one step by step: senses, points and falls.
// Both start from the exact standing pose (no random noise) and get the same motor commands.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { useCatalog } from "../../studio/web/js/engine/catalog.js";
import { buildCreature } from "../../studio/web/js/engine/creatures/index.js";
import { WalkEnv } from "../../studio/web/js/engine/env.js";
import { loadPhysics, setStandPose } from "../../studio/web/js/engine/physics.js";
import { creaturePreview, dropTest } from "../../studio/web/js/engine/previews.js";
import { makeWorld } from "../../studio/web/js/engine/worlds.js";

const read = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
useCatalog(read("../../studio/web/data/catalog.json"));
const cases = read("./fixtures/envs.json");
const mujoco = await loadPhysics();

const close = (a, b, tolerance, label) => assert.ok(Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b)), `${label}: ${a} vs ${b}`);

cases.forEach((item, index) => {
  test(`environment ${index} (${item.animal}) steps like Python`, () => {
    const env = new WalkEnv(mujoco, buildCreature(item.animal, item.values), makeWorld({ values: item.world }), { senses: { feet: true, ground: true } });
    try {
      env.reset(1);
      setStandPose(mujoco, env.model, env.data, env.build, env.scene.standHeight);
      mujoco.mj_forward(env.model, env.data);
      env.contactCache = null;
      env.targetSpeed = 0.45;
      item.steps.forEach((expected, t) => {
        const action = env.build.motors.map((_, m) => Math.sin(0.15 * t + m));
        const result = env.step(action);
        assert.equal(result.observation.length, expected.observation.length);
        result.observation.forEach((value, i) => close(value, expected.observation[i], 1e-5, `step ${t} sense ${i}`));
        close(result.reward, expected.reward, 1e-6, `step ${t} reward`);
        assert.equal(result.terminated, expected.terminated, `step ${t} fell`);
        for (const [name, value] of Object.entries(expected.parts)) close(result.info.reward_parts[name], value, 1e-6, `step ${t} ${name}`);
      });
    } finally {
      env.dispose();
    }
  });
});

test("the creature preview and the drop test give replays the 3D view can show", () => {
  const preview = creaturePreview(mujoco, "duck", {});
  assert.equal(preview.facts.motors, 12);  // the standard duck: neck, legs and toes
  assert.equal(preview.joints.filter((mark) => mark.motor).length, 12);
  assert.ok(preview.shapes.length > 10 && preview.frames[0].length === preview.bodies.length * 7);
  const drop = dropTest(mujoco, "puppy", {}, { values: { terrain: "stairs", terrain_height: 5 } }, 2);
  assert.equal(drop.frames.length, 51);
  assert.equal(drop.ground.kind, "heights");
  assert.equal(typeof drop.stats.stood, "boolean");
});
