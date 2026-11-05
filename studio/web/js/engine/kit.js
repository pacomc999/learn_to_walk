// The building kit every animal uses: cleaning settings, the finished build, and writing MuJoCo XML.
// A port of studio/creatures/kit.py. The setting descriptions themselves come from catalog.json.

// ---------- Settings ----------

// Every setting of a list of groups, by key
export function allSettings(groups) {
  const settings = {};
  for (const group of groups) for (const setting of group.settings) settings[setting.key] = setting;
  return settings;
}

export function defaultValues(groups) {
  const values = {};
  for (const [key, setting] of Object.entries(allSettings(groups))) values[key] = setting.default;
  return values;
}

// The settings with every missing or broken value replaced by its default, numbers kept in range
export function cleanValues(groups, values) {
  values = values || {};
  const cleaned = {};
  for (const [key, setting] of Object.entries(allSettings(groups))) {
    cleaned[key] = cleanValue(setting, key in values ? values[key] : setting.default);
  }
  return cleaned;
}

function cleanValue(setting, value) {
  if (setting.kind === "number") {
    const number = typeof value === "boolean" ? Number(value) : Number(value);
    if (value === null || value === "" || Number.isNaN(number)) return setting.default;
    return Math.min(Math.max(number, setting.minimum), setting.maximum);
  }
  if (setting.kind === "bool") return Boolean(value);
  if (setting.kind === "choice") return setting.choices.some((c) => c.value === value) ? value : setting.default;
  if (setting.kind === "colour") return /^#[0-9a-fA-F]{6}$/.test(String(value)) ? String(value) : setting.default;
  return value;
}

// ---------- The finished build ----------

// One motor: its joint, its group, its mirror partner (itself on the middle line), the mirror sign and its side
export function motor(name, group, mirror, mirrorSign = 1, side = "middle") {
  return { name, group, mirror, mirrorSign, side };
}

export function foot(name, geoms, mirror) {
  return { name, geoms, mirror };
}

// Everything the simulation needs to know about one built animal (defaults as in kit.py)
export function makeBuild(fields) {
  return {
    stridePairs: [], materials: {}, jointDamping: 0.1, jointArmature: 0.005, size: 1.0, torsoGeoms: [],
    symmetricOk: true, motorType: "position",
    ...fields,
    get motorNames() { return this.motors.map((m) => m.name); },
  };
}

// ---------- Writing XML ----------

// Numbers written short but exact enough for the simulation, exactly as Python's f"{value:.6f}" does.
// Python rounds an exact tie to the even digit, JavaScript's toFixed rounds it up, so ties are checked
export function num(value) {
  let text = value.toFixed(6);
  const exact = Math.abs(value).toFixed(20);
  const point = exact.indexOf(".");
  const after = exact.slice(point + 7);
  if (/^50*$/.test(after)) {  // an exact tie: Python keeps the even last digit
    const kept = Number(exact.slice(point + 6, point + 7));
    if (kept % 2 === 0) text = (value < 0 ? "-" : "") + exact.slice(0, point + 7);
  }
  text = text.replace(/0+$/, "").replace(/\.$/, "");
  return text === "-0" || text === "" ? "0" : text;
}

export function vec(...values) {
  return values.map(num).join(" ");
}

// A hex colour such as "#f5c430" as the "r g b a" text MuJoCo uses
export function rgba(hexColour, alpha = 1.0) {
  const channel = (i) => parseInt(hexColour.slice(i, i + 2), 16) / 255;
  return vec(channel(1), channel(3), channel(5), alpha);
}

// A shape line; extra holds any further attributes, already written, in order
export function geom(name, kind, size, mass = null, material = null, extra = {}) {
  const parts = ["<geom"];
  if (name) parts.push(`name="${name}"`);
  parts.push(`type="${kind}"`);
  parts.push(`size="${vec(...size)}"`);
  for (const [key, value] of Object.entries(extra)) parts.push(`${key}="${value}"`);
  if (mass !== null && mass !== undefined) parts.push(`mass="${num(mass)}"`);
  if (material) parts.push(`material="${material}"`);
  return parts.join(" ") + "/>";
}

// A hinge joint around one axis, with its range in radians
export function hinge(name, axis, limit) {
  return `<joint name="${name}" type="hinge" axis="${vec(...axis)}" range="${vec(-limit, limit)}" limited="true"/>`;
}

// A position motor (a spring toward the asked angle) or a force motor (the brain sets the force)
export function motorLine(name, limit, stiffness, force, motorType) {
  if (motorType === "force") {
    return `<motor name="${name}" joint="${name}" ctrlrange="${vec(-force, force)}" forcerange="${vec(-force, force)}" forcelimited="true"/>`;
  }
  return `<position name="${name}" joint="${name}" kp="${num(stiffness)}" ctrlrange="${vec(-limit, limit)}" forcerange="${vec(-force, force)}" forcelimited="true"/>`;
}

// Indent every line of a block of XML by some spaces (blank lines stay blank)
export function indent(text, spaces) {
  const pad = " ".repeat(spaces);
  return splitLines(text).map((line) => (line.trim() ? pad + line : line)).join("\n");
}

// The lines of a text, like Python's str.splitlines(): no empty last line after a final newline
export function splitLines(text) {
  const lines = text.split(/\r?\n/);
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}
