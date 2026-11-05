// Loads the vendored TensorFlow.js in Node for the tests. Its code notices Node and calls require("util"),
// which plain ES modules do not have, so this hands it one first (browsers never take that path).
import { createRequire } from "node:module";

globalThis.require ??= createRequire(import.meta.url);

export async function loadTensorflowForTests() {
  const tf = await import("../../studio/web/vendor/tfjs/tf.fesm.min.js");
  await tf.setBackend("cpu");
  await tf.ready();
  return tf;
}
