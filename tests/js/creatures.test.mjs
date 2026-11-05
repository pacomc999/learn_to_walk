// The JavaScript creatures build exactly what the Python studio builds (fixtures from make_fixtures.py).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { useCatalog } from "../../studio/web/js/engine/catalog.js";
import { buildCreature } from "../../studio/web/js/engine/creatures/index.js";
import { num } from "../../studio/web/js/engine/kit.js";

const read = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8"));
useCatalog(read("../../studio/web/data/catalog.json"));
const cases = read("./fixtures/creatures.json");

test("numbers are written as Python writes them", () => {
  assert.equal(num(0.0078125), "0.007812");  // an exact tie keeps the even digit
  assert.equal(num(0.0234375), "0.023438");
  assert.equal(num(-0.0), "0");
  assert.equal(num(-0.0000001), "0");
  assert.equal(num(1.5), "1.5");
  assert.equal(num(20), "20");
});

cases.forEach((item, index) => {
  test(`${item.animal} case ${index} builds like Python`, () => {
    const build = buildCreature(item.animal, item.values);
    const expected = item.expected;
    assert.equal(build.bodyXml, expected.bodyXml);
    assert.equal(build.actuatorXml, expected.actuatorXml);
    assert.deepEqual(build.motors, expected.motors);
    assert.deepEqual(build.feet, expected.feet);
    assert.deepEqual(build.standPose, expected.standPose);
    assert.deepEqual(build.stridePairs, expected.stridePairs);
    assert.deepEqual(build.materials, expected.materials);
    assert.deepEqual(build.torsoGeoms, expected.torsoGeoms);
    for (const key of ["jointDamping", "jointArmature", "size"]) assert.ok(Math.abs(build[key] - expected[key]) < 1e-12, key);
    assert.equal(build.symmetricOk, expected.symmetricOk);
    assert.equal(build.motorType, expected.motorType);
  });
});
