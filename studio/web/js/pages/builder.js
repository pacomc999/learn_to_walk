// Build your own: put parts together (sticks, balls and plates), join them with motors, and watch it in 3D.
// Parts are added and moved by clicking on the creature in the 3D view; the list and the editor fine tune them.

import { h, toast, settingControl, confirmDialog, fmtNumber } from "../ui.js";
import { icon } from "../icons.js";
import { JOINT_COLOURS } from "../viewer.js";

const PART_COLOURS = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"];
const MOST_PARTS = 24;

// The kinds of joint, and the motors each one gets (the colours match the rings in the 3D view)
const JOINT_MOTORS = { fixed: [], swing: ["swing"], side: ["side"], twist: ["twist"], hip: ["swing", "side"] };
const JOINT_CHOICES = [
  { value: "fixed", label: "Glued (no motor)" }, { value: "swing", label: "Swing (1 motor)" },
  { value: "side", label: "Sideways (1 motor)" }, { value: "twist", label: "Twist (1 motor)" },
  { value: "hip", label: "Hip (2 motors)" },
];
const JOINT_HELP = {
  fixed: "Glued on: no motor, it moves with its parent.",
  swing: "Swings forward and back, like a knee. One motor.",
  side: "Swings out to the side and back in. One motor.",
  twist: "Turns around the up and down line. One motor.",
  hip: "Swings forward and back and out to the side, like a hip. Two motors.",
};
const MOTOR_WORDS = { swing: "forward and back", side: "sideways", twist: "twist", glued: "glued, no motor" };

const clone = (value) => JSON.parse(JSON.stringify(value));
const degrees = (radians) => (radians * 180) / Math.PI;
const radians = (deg) => (deg * Math.PI) / 180;
const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const roundTo = (value, step) => Math.round(value / step) * step;

// The direction a part points to, from its tilt down (pitch) and turn to the left (yaw), as custom.py does
function pointing(pitch, yaw) {
  const p = radians(pitch), y = radians(yaw);
  return { x: Math.cos(p) * Math.cos(y), y: Math.cos(p) * Math.sin(y), z: -Math.sin(p) };
}

// Half sizes of the torso and of ball or plate parts (what attach points are shares of)
function torsoHalf(torso) {
  if (torso.shape === "ball") return { x: torso.length / 2, y: torso.length / 2, z: torso.length / 2 };
  return { x: torso.length / 2, y: torso.width / 2, z: torso.height / 2 };
}
function partHalf(part) {
  if (part.shape === "plate") return { x: part.length / 2, y: part.width / 2, z: part.thickness / 2 };
  return { x: part.thickness, y: part.thickness, z: part.thickness };
}
function shapeCentre(part) {
  const d = pointing(part.pitch, part.yaw);
  const reach = part.shape === "ball" ? part.length : part.length / 2;
  return { x: d.x * reach, y: d.y * reach, z: d.z * reach };
}

// The builder's panel. creature.values.body holds the design; onChange() redraws the 3D view
export function builderPanel(creature, animal, viewer, onChange) {
  const body = creature.values.body;
  let selected = "torso";
  let placing = null;     // null, { kind: "add" } or { kind: "move", id }
  let addAsPair = false;  // the "One part / Mirrored pair" switch
  const history = [];
  const future = [];
  let lastSnapshot = JSON.stringify(body);

  const panel = h("div", { class: "stack", style: { gap: "12px" } });
  const tree = h("div", { class: "parts-tree" });
  const editor = h("div");
  const undoButton = h("button", { class: "btn small ghost", title: "Undo", onclick: () => undo(), html: `${icon.restart}<span>Undo</span>` });
  const redoButton = h("button", { class: "btn small ghost", title: "Redo", onclick: () => redo() }, "Redo");

  // ---------- Undo and redo ----------

  const remember = () => {
    const now = JSON.stringify(body);
    if (now === lastSnapshot) return;
    history.push(lastSnapshot);
    if (history.length > 80) history.shift();
    future.length = 0;
    lastSnapshot = now;
    paintUndo();
  };
  const restore = (snapshot) => {
    const data = JSON.parse(snapshot);
    body.template = data.template;
    body.torso = data.torso;
    body.parts = data.parts;
    lastSnapshot = snapshot;
    if (selected !== "torso" && !partById(selected)) selected = "torso";
    paintAll();
    onChange();
  };
  const undo = () => { if (!history.length) return; future.push(JSON.stringify(body)); restore(history.pop()); paintUndo(); };
  const redo = () => { if (!future.length) return; history.push(JSON.stringify(body)); restore(future.pop()); paintUndo(); };
  const paintUndo = () => { undoButton.disabled = !history.length; redoButton.disabled = !future.length; };

  // A change from the editor: remember it for undo (after a pause) and redraw
  let rememberTimer = null;
  const changed = (structure = false) => {
    clearTimeout(rememberTimer);
    rememberTimer = setTimeout(remember, 400);
    if (structure) paintTree();
    onChange();
  };
  // A change made in one go (adding, moving, removing): remembered straight away
  const changedNow = () => { clearTimeout(rememberTimer); remember(); paintAll(); onChange(); };

  // ---------- Looking things up ----------

  const partById = (id) => body.parts.find((p) => p.id === id);
  const childrenOf = (id) => body.parts.filter((p) => p.parent === id);
  const descendants = (id) => childrenOf(id).flatMap((child) => [child.id, ...descendants(child.id)]);
  // True when the part, or something it hangs on, is a mirrored pair: then it has a twin already
  const hasTwin = (id) => {
    let part = partById(id);
    while (part) { if (part.mirror) return true; part = partById(part.parent); }
    return false;
  };
  const insideMirrored = (part) => hasTwin(part.parent);
  const newId = (taken = []) => {
    let n = body.parts.length + taken.length + 1;
    while (partById(`p${n}`) || taken.some((p) => p.id === `p${n}`)) n += 1;
    return `p${n}`;
  };

  // ---------- Selecting ----------

  const select = (id) => {
    selected = id || "torso";
    paintAll();
  };
  const highlight = () => {
    if (selected === "torso") viewer.highlight((name) => name === "torso");
    else viewer.highlight((name) => name === selected || name === `left_${selected}` || name === `right_${selected}`);
  };

  // ---------- Placing parts by clicking on the creature ----------

  // Turns a spot clicked in the 3D view into an attach point on that part, or null if it cannot go there
  const placementFrom = (hit) => {
    if (!hit || !hit.name) return null;
    const flip = hit.name.startsWith("right_") ? -1 : 1;  // the right twin is the left one mirrored
    const parentId = hit.name.replace(/^(left|right)_/, "");
    const parent = parentId === "torso" ? null : partById(parentId);
    if (parentId !== "torso" && !parent) return null;
    if (placing && placing.kind === "move" && (parentId === placing.id || descendants(placing.id).includes(parentId))) return null;

    // The spot and the surface's outward direction, in the parent's own frame (the left side for twins)
    const local = viewer.toBodyFrame(hit.body, hit.point, hit.normal);
    const spot = { x: local.point.x, y: local.point.y * flip, z: local.point.z };
    const out = { x: local.direction.x, y: local.direction.y * flip, z: local.direction.z };

    // The new part points outward and a little down, so a leg on the side ends up on the ground.
    // From the front or back end it points straight out, like a neck or a tail
    const down = out.z < -0.7 || Math.abs(out.x) > 0.7 ? 0 : 1;
    const length = Math.hypot(out.x, out.y, out.z - down) || 1;
    const aim = { x: out.x / length, y: out.y / length, z: (out.z - down) / length };
    const place = {
      parent: parentId,
      pitch: roundTo(clamp(degrees(Math.asin(clamp(-aim.z, -1, 1))), -90, 90), 5),
      yaw: roundTo(degrees(Math.atan2(aim.y, aim.x)), 5),
      x: 0, y: 0, z: 0, along: 1,
    };
    if (!parent) {
      const half = torsoHalf(body.torso);
      Object.assign(place, { x: clamp(spot.x / half.x, -1, 1), y: clamp(spot.y / half.y, -1, 1), z: clamp(spot.z / half.z, -1, 1) });
    } else if (parent.shape === "stick") {
      const d = pointing(parent.pitch, parent.yaw);
      place.along = clamp((spot.x * d.x + spot.y * d.y + spot.z * d.z) / Math.max(parent.length, 0.001), 0, 1);
      // Near the far end (its last quarter, or its rounded tip): the new part carries the stick on,
      // pointing the same way (a chain, a longer leg)
      const facesAlong = out.x * d.x + out.y * d.y + out.z * d.z > 0.5;
      if (place.along >= 0.75 || facesAlong) {
        Object.assign(place, { along: 1, pitch: parent.pitch, yaw: parent.yaw, continues: true });
        Object.assign(aim, pointing(parent.pitch, parent.yaw));
      }
    } else {
      const half = partHalf(parent), centre = shapeCentre(parent);
      Object.assign(place, { x: clamp((spot.x - centre.x) / half.x, -1, 1), y: clamp((spot.y - centre.y) / half.y, -1, 1),
        z: clamp((spot.z - centre.z) / half.z, -1, 1) });
    }
    for (const key of ["x", "y", "z", "along"]) place[key] = roundTo(place[key], 0.05);

    // One part or a pair: a part on a mirrored parent always has a twin; a pair needs to be off the middle line
    const parentHasTwin = parentId !== "torso" && hasTwin(parentId);
    // Off the middle line: measured in the world, against the middle of the body
    const middle = viewer.bodyGroups[0] ? viewer.bodyGroups[0].position.y : 0;
    place.offMiddle = Math.abs(hit.point.y - middle) > 0.005;
    const canPair = !parentHasTwin && Math.abs(spot.y) > 0.005 && !(parent && parent.shape === "stick");
    place.mirror = addAsPair && canPair;
    if (place.mirror && place.y < 0) { place.y = -place.y; place.yaw = -place.yaw; }  // the design keeps the left one
    place.twin = parentHasTwin || place.mirror;
    place.hit = hit;
    place.aimWorld = { x: aim.x, y: aim.y * flip, z: aim.z };
    return place;
  };

  // The see-through preview of the new (or moved) part while the mouse moves over the creature
  const showGhost = (place) => {
    if (!place) { viewer.setGhost([]); return; }
    // A moved part keeps its size; a part that carries a stick on copies that stick
    const sized = placing.kind === "move" ? partById(placing.id) : place.continues ? partById(place.parent) : null;
    const length = sized ? Math.max(sized.length, 0.02) : 0.12;
    const radius = sized ? sized.thickness : 0.012;
    const from = place.hit.point.clone();
    const to = from.clone();
    to.x += place.aimWorld.x * length; to.y += place.aimWorld.y * length; to.z += place.aimWorld.z * length;
    const sticks = [{ from, to, radius }];
    if (place.twin) {
      // The twin is the mirror image across the creature's middle line
      const middle = viewer.bodyGroups[0] ? viewer.bodyGroups[0].position.y : 0;
      const mirror = (point) => { const copy = point.clone(); copy.y = 2 * middle - copy.y; return copy; };
      sticks.push({ from: mirror(from), to: mirror(to), radius });
    }
    viewer.setGhost(sticks);
  };

  // The next free name in a series: "Segment 5" gives "Segment 6", "Shin" gives "Shin 2"
  const nextName = (name) => {
    const base = name.replace(/\s*\d+$/, "");
    const numbers = body.parts.map((p) => numberAfter(p.name, base)).filter((n) => n !== null);
    return `${base} ${Math.max(1, ...numbers) + 1}`;
  };
  // "Segment 4" with base "Segment" gives 4, "Segment" gives 1, anything else null
  const numberAfter = (name, base) => {
    if (name === base) return 1;
    if (!name.startsWith(`${base} `)) return null;
    const rest = name.slice(base.length + 1);
    return /^\d+$/.test(rest) ? Number(rest) : null;
  };

  // A copy of a part's look and joint (not its place), for chains and longer legs
  const lookOf = (part) => ({ shape: part.shape, length: part.length, thickness: part.thickness, width: part.width,
    joint: part.joint, motion: part.motion, strength: part.strength, heaviness: part.heaviness, colour: part.colour });

  // Who feels the ground when a part gets a new part on its end: a leg pointing down hands it to the new tip,
  // a chain lying on the ground (a worm) keeps it on every segment
  const passOnFeel = (parent, part) => {
    if (!parent || parent.shape !== "stick" || part.along < 0.8) return;
    part.foot = parent.foot || part.foot;
    if (parent.pitch > 45) parent.foot = false;
  };

  const addPartAt = (place) => {
    if (body.parts.length >= MOST_PARTS) { toast("That is a lot of parts", `A creature can have up to ${MOST_PARTS} parts.`); return; }
    const parent = place.parent === "torso" ? null : partById(place.parent);
    const id = newId();
    const legs = body.parts.filter((p) => p.parent === "torso").length;
    const part = {
      id, name: parent ? `${parent.name} part` : `Leg ${legs + 1}`, parent: place.parent, shape: "stick", length: 0.12,
      thickness: 0.012, width: 0.05, x: place.x, y: place.y, z: place.z, along: place.along, pitch: place.pitch, yaw: place.yaw,
      joint: parent ? "swing" : "hip", motion: 60, strength: 10, mirror: place.mirror, foot: true, heaviness: 100,
      colour: parent ? parent.colour : PART_COLOURS[body.parts.length % PART_COLOURS.length],
    };
    if (place.continues) Object.assign(part, lookOf(parent), { name: nextName(parent.name), foot: false });
    passOnFeel(parent, part);
    body.parts.push(part);
    selected = id;
    const what = place.twin ? "A mirrored pair was added" : "One part was added";
    const lopsided = !place.twin && place.offMiddle;
    toast(what, lopsided ? "It has no twin on the other side, so this creature will get a free brain instead of the symmetric one." : "Change it on the left, or click Add part again for another.");
  };

  // Extend: the same part again on the end of the selected one, pointing the same way. Press again for a chain
  const extendPart = (id) => {
    const parent = partById(id);
    if (!parent) return;
    if (body.parts.length >= MOST_PARTS) { toast("That is a lot of parts", `A creature can have up to ${MOST_PARTS} parts.`); return; }
    const part = { ...lookOf(parent), id: newId(), name: nextName(parent.name), parent: id, x: 0, y: 0, z: 0, along: 1,
      pitch: parent.pitch, yaw: parent.yaw, mirror: false, foot: false };
    passOnFeel(parent, part);
    body.parts.push(part);
    selected = part.id;
    changedNow();
  };

  const movePartTo = (id, place) => {
    const part = partById(id);
    if (!part) return;
    const wasPair = part.mirror || hasTwin(part.parent);
    Object.assign(part, { parent: place.parent, x: place.x, y: place.y, z: place.z, along: place.along, pitch: place.pitch, yaw: place.yaw });
    // A pair stays a pair where it can be one; on a mirrored parent the twin comes from the parent
    const parent = partById(place.parent);
    const canPair = !hasTwin(place.parent) && Math.abs(place.y) > 0.001 && !(parent && parent.shape === "stick");
    part.mirror = wasPair && canPair;
    if (part.mirror && part.y < 0) { part.y = -part.y; part.yaw = -part.yaw; }
    selected = id;
  };

  const startPlacing = (mode) => {
    placing = mode;
    paintBar();
  };
  const stopPlacing = () => {
    placing = null;
    viewer.setGhost([]);
    paintBar();
  };

  viewer.onHover = (hit) => { if (placing) showGhost(placementFrom(hit)); };
  viewer.onPick = (name, hit) => {
    if (placing) {
      const place = placementFrom(hit);      if (!place) { if (!name) stopPlacing(); return; }
      if (placing.kind === "add") addPartAt(place); else movePartTo(placing.id, place);
      stopPlacing();
      changedNow();
      return;
    }
    if (!name) return;
    const id = name.replace(/^(left|right)_/, "");
    if (id === "torso" || partById(id)) select(id);
  };
  const onKey = (event) => {
    if (viewer.disposed) { window.removeEventListener("keydown", onKey); return; }
    if (event.key === "Escape" && placing) stopPlacing();
  };
  window.addEventListener("keydown", onKey);

  // ---------- The bar above the 3D view ----------

  const bar = h("div", { class: "build-bar" });
  const paintBar = () => {
    const part = selected !== "torso" ? partById(selected) : null;
    const adding = placing && placing.kind === "add";
    const moving = placing && placing.kind === "move";
    const segment = (pair, label) => h("button", { class: pair === addAsPair ? "on" : "", onclick: () => { addAsPair = pair; paintBar(); } }, label);
    const hint = adding ? `Click where ${addAsPair ? "the pair" : "it"} goes. The end of a stick carries it on. Esc cancels.`
      : moving ? `Click where ${part ? part.name : "the part"} should go. Esc cancels.`
      : part ? `Extend adds another ${part.name} on its end, for chains like a worm.`
      : "Add part, then click on the creature where it goes.";
    bar.replaceChildren(
      h("div", { class: "build-bar-row" },
        h("button", { class: `btn small ${adding ? "primary" : ""}`, onclick: () => (adding ? stopPlacing() : startPlacing({ kind: "add" })),
          html: `${icon.plus}<span>${adding ? "Cancel" : "Add part"}</span>` }),
        h("div", { class: "seg", title: "A mirrored pair gets a twin on the other side, so the brain can be left/right symmetric" },
          segment(false, "One part"), segment(true, "Mirrored pair")),
        h("button", { class: "btn small ghost", disabled: !part || Boolean(placing),
          title: part ? `Add another ${part.name} on its end, pointing the same way` : "Select a part first",
          onclick: () => extendPart(part.id), html: `${icon.plus}<span>Extend</span>` }),
        h("button", { class: `btn small ${moving ? "primary" : "ghost"}`, disabled: !part,
          title: part ? `Attach ${part.name} somewhere else` : "Select a part first", onclick: () => (moving ? stopPlacing() : startPlacing({ kind: "move", id: part.id })),
          html: `${icon.target}<span>${moving ? "Cancel" : "Move"}</span>` }),
        h("label", { class: "build-toggle" },
          h("input", { type: "checkbox", checked: viewer.showJoints, onchange: (event) => viewer.setJointsVisible(event.target.checked) }),
          "Show motors")),
      h("div", { class: `build-hint ${placing ? "on" : ""}` }, hint),
      legend());
    viewer.container.classList.toggle("placing", Boolean(placing));
  };
  const legend = () => h("div", { class: "motor-legend" },
    ...["swing", "side", "twist", "glued"].map((kind) => h("span", {}, motorDot(kind), MOTOR_WORDS[kind])));
  const motorDot = (kind) => h("i", { class: `motor-dot ${kind === "glued" ? "glued" : ""}`, style: { borderColor: JOINT_COLOURS[kind], background: kind === "glued" ? JOINT_COLOURS[kind] : "transparent" } });
  if (viewer.container.parentElement) viewer.container.parentElement.insertBefore(bar, viewer.container);

  // ---------- Removing and duplicating ----------

  const removePart = async (id) => {
    const gone = [id, ...descendants(id)];
    if (gone.length > 1 && !(await confirmDialog("Remove this part?", `It takes ${gone.length - 1} part${gone.length > 2 ? "s" : ""} hanging on it with it.`, "Remove", "danger"))) return;
    body.parts = body.parts.filter((p) => !gone.includes(p.id));
    if (gone.includes(selected)) selected = "torso";
    changedNow();
  };

  const duplicatePart = (id) => {
    const original = partById(id);
    const ids = [id, ...descendants(id)];
    const mapping = {};
    const copies = [];
    for (const oldId of ids) {
      const copy = clone(partById(oldId));
      const fresh = newId(copies);
      mapping[oldId] = fresh;
      copy.id = fresh;
      copy.parent = oldId === id ? original.parent : mapping[copy.parent];
      if (oldId === id) { copy.name = `${original.name} copy`; copy.x = clamp((copy.x || 0) - 0.5, -1, 1); }
      copies.push(copy);
    }
    if (body.parts.length + copies.length > MOST_PARTS) { toast("Too many parts", `A creature can have up to ${MOST_PARTS} parts.`); return; }
    body.parts.push(...copies);
    selected = copies[0].id;
    toast("Copied", "The copy sits a little further back. Use Move to put it where you want.");
    changedNow();
  };

  // ---------- The parts list ----------

  // How many motors a part has, counting its twin
  const motorCount = (part) => JOINT_MOTORS[part.joint].length * (hasTwin(part.id) ? 2 : 1);
  const totalMotors = () => body.parts.reduce((sum, part) => sum + motorCount(part), 0);

  const paintTree = () => {
    const rows = [row("torso", "Body", `${body.torso.shape}, ${fmtNumber(body.torso.weight, 2)} kg`, 0, body.torso.colour, [])];
    const walk = (parentId, depth) => {
      for (const part of childrenOf(parentId)) {
        const tags = [hasTwin(part.id) ? "pair" : "one", part.shape];
        if (part.foot) tags.push("feels ground");
        const kinds = JOINT_MOTORS[part.joint].length ? JOINT_MOTORS[part.joint] : ["glued"];
        rows.push(row(part.id, part.name, tags.join(", "), depth, part.colour, kinds));
        walk(part.id, depth + 1);
      }
    };
    walk("torso", 1);
    tree.replaceChildren(...rows);
    treeSummary.textContent = `${body.parts.length + 1} parts, ${totalMotors()} motors`;
  };

  const row = (id, name, meta, depth, colour, kinds) => h("div", {
    class: `part-row ${selected === id ? "on" : ""}`, style: { paddingLeft: `${10 + Math.min(depth, 4) * 12}px` }, onclick: () => select(id) },
    h("span", { class: "swatch", style: { background: colour } }),
    h("div", { class: "part-text" }, h("div", { class: "part-name" }, name), h("div", { class: "part-meta" }, meta)),
    h("div", { class: "part-motors", title: kinds.map((k) => MOTOR_WORDS[k]).join(", ") }, ...kinds.map(motorDot)),
    id !== "torso" ? h("div", { class: "part-actions" },
      h("button", { class: "reset-link", title: "Duplicate", html: icon.copy, onclick: (event) => { event.stopPropagation(); duplicatePart(id); } }),
      h("button", { class: "reset-link", title: "Remove", html: icon.trash, onclick: (event) => { event.stopPropagation(); removePart(id); } })) : null);
  const treeSummary = h("div", { class: "desc" });

  // ---------- The editor of the selected part ----------

  // A slider bound to a field, shown in other units (cm, degrees or percent)
  const slider = (target, key, label, { min, max, step, unit, scale = 1, help = "", structure = false }) => settingControl(
    { key, label, kind: "number", minimum: min, maximum: max, step, unit, help },
    Math.round(target[key] * scale / step) * step,
    (value) => { target[key] = value / scale; changed(structure); },
    { defaultValue: Math.round(target[key] * scale / step) * step });

  const toggle = (target, key, label, help, structure = true) => settingControl(
    { key, label, kind: "bool", help }, target[key], (value) => { target[key] = value; changed(structure); }, { defaultValue: target[key] });

  const choice = (target, key, label, choices, help = "", after = null) => settingControl(
    { key, label, kind: "choice", choices, help }, target[key], (value) => { target[key] = value; changed(true); if (after) after(); },
    { defaultValue: target[key] });

  const colourField = (target, key, label) => settingControl(
    { key, label, kind: "colour" }, target[key], (value) => { target[key] = value; changed(true); }, { defaultValue: target[key] });

  const section = (title, ...children) => h("div", { class: "edit-section" }, h("div", { class: "edit-title" }, title), ...children);

  const torsoEditor = () => {
    const torso = body.torso;
    return h("div", { class: "card card-pad" },
      h("h3", { style: { marginBottom: "4px" } }, "Body"),
      h("div", { class: "faint", style: { fontSize: "12.5px" } }, "The middle of the creature. Everything else hangs on it."),
      section("Shape",
        choice(torso, "shape", "Shape", [{ value: "box", label: "Box" }, { value: "ellipsoid", label: "Egg" }, { value: "capsule", label: "Tube" }, { value: "ball", label: "Ball" }]),
        slider(torso, "length", "Length (front to back)", { min: 4, max: 120, step: 1, unit: "cm", scale: 100 }),
        torso.shape !== "ball" ? slider(torso, "width", "Width", { min: 4, max: 100, step: 1, unit: "cm", scale: 100 }) : null,
        torso.shape !== "ball" ? slider(torso, "height", "Height", { min: 2, max: 100, step: 1, unit: "cm", scale: 100 }) : null),
      section("Weight and colour",
        slider(torso, "weight", "Weight", { min: 0.05, max: 20, step: 0.05, unit: "kg" }),
        colourField(torso, "colour", "Colour")));
  };

  // The motors of the selected part in words, with the ring colours
  const motorSummary = (part) => {
    const kinds = JOINT_MOTORS[part.joint];
    const twin = hasTwin(part.id);
    if (!kinds.length) return h("div", { class: "motor-summary" }, motorDot("glued"), "Glued on: no motor.");
    return h("div", { class: "motor-summary" }, ...kinds.map(motorDot),
      `${kinds.length} motor${kinds.length > 1 ? "s" : ""} (${kinds.map((k) => MOTOR_WORDS[k]).join(" and ")})${twin ? ", and the same on its twin" : ""}.`);
  };

  const partEditor = (part) => {
    const parent = part.parent === "torso" ? null : partById(part.parent);
    const nameInput = h("input", { class: "input", value: part.name });
    nameInput.addEventListener("input", () => { part.name = nameInput.value; changed(true); });
    const where = parent && parent.shape === "stick"
      ? [slider(part, "along", `Where on ${parent.name}`, { min: 0, max: 100, step: 5, unit: "%", scale: 100, help: "0% is where the parent starts, 100% its far end." })]
      : [slider(part, "x", "Front or back", { min: -100, max: 100, step: 5, unit: "%", scale: 100, help: `Along ${parent ? parent.name : "the body"}: -100% the back, +100% the front.` }),
        slider(part, "y", "Left or right", { min: -100, max: 100, step: 5, unit: "%", scale: 100, help: "+100% the left side. A pair gets its twin on the other side." }),
        slider(part, "z", "Down or up", { min: -100, max: 100, step: 5, unit: "%", scale: 100, help: "-100% underneath, +100% on top." })];
    const sizes = part.shape === "ball"
      ? [slider(part, "thickness", "Radius", { min: 0.5, max: 20, step: 0.5, unit: "cm", scale: 100 }),
        slider(part, "length", "Distance from the joint", { min: 0, max: 60, step: 1, unit: "cm", scale: 100 })]
      : part.shape === "plate"
        ? [slider(part, "length", "Length", { min: 1, max: 60, step: 1, unit: "cm", scale: 100 }),
          slider(part, "width", "Width", { min: 1, max: 50, step: 1, unit: "cm", scale: 100 }),
          slider(part, "thickness", "Thickness", { min: 0.5, max: 10, step: 0.5, unit: "cm", scale: 100 })]
        : [slider(part, "length", "Length", { min: 1, max: 80, step: 1, unit: "cm", scale: 100 }),
          slider(part, "thickness", "Thickness (radius)", { min: 0.5, max: 10, step: 0.1, unit: "cm", scale: 100 })];
    return h("div", { class: "card card-pad" },
      h("div", { class: "spread", style: { marginBottom: "6px", gap: "8px" } }, nameInput,
        h("button", { class: "btn small ghost", title: "Remove this part", onclick: () => removePart(part.id), html: `${icon.trash}<span>Remove</span>` })),
      h("div", { class: "faint", style: { fontSize: "12.5px" } }, `Hangs on ${parent ? parent.name : "the body"}. Use Move above the 3D view to attach it somewhere else.`),
      section("Joint and motors",
        motorSummary(part),
        choice(part, "joint", "Joint", JOINT_CHOICES, JOINT_HELP[part.joint], paintAll),
        part.joint !== "fixed" ? slider(part, "motion", "How far it can move", { min: 5, max: 170, step: 5, unit: "°", help: "The whole swing, from one end stop to the other." }) : null,
        part.joint !== "fixed" ? slider(part, "strength", "Muscle strength", { min: 0.5, max: 100, step: 0.5, unit: "N m", help: "The strongest push of its motor." }) : null),
      section("Shape and size",
        choice(part, "shape", "Shape", [{ value: "stick", label: "Stick" }, { value: "ball", label: "Ball" }, { value: "plate", label: "Plate" }], "", paintEditor),
        ...sizes),
      section("Where it attaches", ...where,
        !insideMirrored(part)
          ? toggle(part, "mirror", "Mirrored pair", "Adds a twin on the other side, moving like a mirror image. Pairs let the brain be left/right symmetric.")
          : h("div", { class: "field-help" }, "Part of a mirrored pair: its twin follows it.")),
      section("Which way it points",
        slider(part, "pitch", "Tilt down", { min: -90, max: 90, step: 5, unit: "°", help: "90° straight down, 0° level, -90° straight up." }),
        slider(part, "yaw", "Turn", { min: -180, max: 180, step: 5, unit: "°", help: "0° forward, 90° out to the left, 180° backward." })),
      section("More",
        toggle(part, "foot", "Feels the ground", "The brain is told when this part touches the ground. Give it to feet and tips."),
        slider(part, "heaviness", "Heaviness", { min: 10, max: 1000, step: 10, unit: "%", help: "100% weighs like water; the weight follows the size." }),
        colourField(part, "colour", "Colour")));
  };

  const paintEditor = () => {
    const part = selected === "torso" ? null : partById(selected);
    if (selected !== "torso" && !part) selected = "torso";
    editor.replaceChildren(part ? partEditor(part) : torsoEditor());
  };

  // ---------- Templates ----------

  // Picking one swaps the body straight away; Undo brings the old one back
  const useTemplate = (template) => {
    clearTimeout(rememberTimer);
    history.push(JSON.stringify(body));
    future.length = 0;
    body.template = template.key;
    body.torso = clone(template.torso);
    body.parts = clone(template.parts);
    lastSnapshot = JSON.stringify(body);
    selected = "torso";
    stopPlacing();
    paintAll();
    paintTemplates();
    onChange(true);
  };
  const templates = h("div", { class: "btn-row" });
  const paintTemplates = () => templates.replaceChildren(...animal.templates.map((template) =>
    h("button", { class: `btn small ${body.template === template.key ? "primary" : ""}`, title: template.description, onclick: () => useTemplate(template) }, template.name)));

  const paintAll = () => { paintTree(); paintEditor(); highlight(); paintBar(); paintUndo(); };

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "10px" } },
      h("div", {}, h("h3", {}, "Start from"), h("div", { class: "faint", style: { fontSize: "12.5px" } }, "A ready-made body to change. Undo brings back the one you had.")),
      templates),
    h("div", { class: "card" },
      h("div", { class: "card-head" }, h("div", {}, h("h3", {}, "Parts"), treeSummary),
        h("div", { class: "row", style: { gap: "2px" } }, undoButton, redoButton)),
      h("div", { class: "card-body", style: { paddingTop: "10px" } }, tree)),
    editor);
  paintTemplates();
  paintAll();
  return { element: panel, highlight };
}
