// Worlds: the ground, the physics and the task a walker trains in. A port of studio/worlds.py.
// The settings and the built-in worlds come from catalog.json.

import { getCatalog, worldGroups } from "./catalog.js";
import { cleanValues, defaultValues } from "./kit.js";

export function worldDefaults() {
  return defaultValues(worldGroups());
}

// One world with its values cleaned, and the shortcuts the simulation reads often
export function makeWorld({ id = "world", name = "World", description = "", values = {}, builtin = false } = {}) {
  const cleaned = cleanValues(worldGroups(), values);
  const tallest = cleaned.terrain !== "flat" ? cleaned.terrain_height / 100.0 : 0.0;  // metres
  const terrain = tallest > 0 ? cleaned.terrain : "flat";
  const low = cleaned.slowest_speed, high = cleaned.fastest_speed;
  return {
    id: String(id), name: String(name), description: String(description), values: cleaned, builtin: Boolean(builtin),
    tallest, terrain,
    speedRange: [Math.min(low, high), Math.max(low, high)],
    turning: Boolean(cleaned.turning) && terrain === "flat",  // only on flat ground (bumpy floors are narrow strips)
    grip: cleaned.ground_grip / 100.0,
  };
}

// A world from saved data, every missing value filled with its default
export function worldFromDict(data = {}) {
  data = data || {};
  return makeWorld({ id: data.id ?? "world", name: data.name ?? "World", description: data.description ?? "",
    values: data.values || {}, builtin: data.builtin });
}

export function worldToDict(world) {
  return { id: world.id, name: world.name, description: world.description, builtin: world.builtin, values: { ...world.values } };
}

// The built-in worlds, in the order the page shows them
export function builtinWorlds() {
  return getCatalog().world_templates.map((data) => worldFromDict({ ...data, builtin: true }));
}

// A plain world with every setting at its default, for previews
export function plainWorld(id = "preview") {
  return makeWorld({ id, name: id, values: worldDefaults() });
}
