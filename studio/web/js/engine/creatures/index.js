// The animals that can learn to walk, by key. A port of studio/creatures/__init__.py.

import * as custom from "./custom.js";
import * as duck from "./duck.js";
import * as puppy from "./puppy.js";

export const ANIMALS = { duck, puppy, custom };

// An animal built from its settings, with a plain message for an unknown one
export function buildCreature(key, values = {}) {
  const animal = ANIMALS[key];
  if (!animal) throw new Error(`There is no animal called "${key}" (known: ${Object.keys(ANIMALS).join(", ")}).`);
  return animal.build(values);
}
