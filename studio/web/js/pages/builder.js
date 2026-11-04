// Build your own: put parts together (sticks, balls and plates), join them with motors, and watch it in 3D.

import { h, toast, settingControl, confirmDialog, fmtNumber } from "../ui.js";
import { icon } from "../icons.js";

const PART_COLOURS = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6"];

const JOINT_CHOICES = [
  { value: "fixed", label: "Fixed" }, { value: "swing", label: "Swing" }, { value: "side", label: "Sideways" },
  { value: "twist", label: "Twist" }, { value: "hip", label: "Hip (2)" },
];
const JOINT_HELP = {
  fixed: "Glued on: no motor, it moves with its parent.",
  swing: "Swings forward and back, like a knee. One motor.",
  side: "Swings out to the side and back in. One motor.",
  twist: "Turns around the up and down line. One motor.",
  hip: "Swings forward and back and out to the side, like a hip. Two motors.",
};

const clone = (value) => JSON.parse(JSON.stringify(value));

// The builder's panel. creature.values.body holds the design; onChange() redraws the 3D view
export function builderPanel(creature, animal, viewer, onChange) {
  const body = creature.values.body;
  let selected = "torso";
  const history = [];
  const future = [];
  let lastSnapshot = JSON.stringify(body);

  const panel = h("div", { class: "stack", style: { gap: "12px" } });
  const tree = h("div", { class: "parts-tree" });
  const editor = h("div");
  const undoButton = h("button", { class: "btn small ghost", title: "Undo", onclick: () => undo(), html: `${icon.restart}<span>Undo</span>` });
  const redoButton = h("button", { class: "btn small ghost", title: "Redo", onclick: () => redo() }, "Redo");

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
    body.torso = data.torso;
    body.parts = data.parts;
    lastSnapshot = snapshot;
    if (selected !== "torso" && !body.parts.some((p) => p.id === selected)) selected = "torso";
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

  const partById = (id) => body.parts.find((p) => p.id === id);
  const childrenOf = (id) => body.parts.filter((p) => p.parent === id);
  const insideMirrored = (part) => {
    let parent = partById(part.parent);
    while (parent) { if (parent.mirror) return true; parent = partById(parent.parent); }
    return false;
  };
  const descendants = (id) => childrenOf(id).flatMap((child) => [child.id, ...descendants(child.id)]);

  const select = (id) => {
    selected = id || "torso";
    paintTree();
    paintEditor();
    highlight();
  };
  const highlight = () => {
    if (selected === "torso") viewer.highlight((name) => name === "torso");
    else viewer.highlight((name) => name === selected || name === `left_${selected}` || name === `right_${selected}`);
  };
  viewer.onPick = (name) => {
    if (!name) return;
    const id = name.replace(/^(left|right)_/, "");
    if (id === "torso" || partById(id)) select(id);
  };

  const newId = () => {
    let n = body.parts.length + 1;
    while (partById(`p${n}`)) n += 1;
    return `p${n}`;
  };

  const addPart = (parentId) => {
    if (body.parts.length >= 24) { toast("That is a lot of parts", "A creature can have up to 24 parts."); return; }
    const parent = parentId === "torso" ? null : partById(parentId);
    const id = newId();
    const onTorso = !parent;
    const part = {
      id, name: onTorso ? "Leg" : `${parent.name} part`, parent: parentId, shape: "stick", length: 0.12, thickness: 0.014,
      width: 0.05, x: onTorso ? 0 : 0, y: onTorso ? 1 : 0, z: onTorso ? -1 : 0, along: 1, pitch: 90, yaw: 0,
      joint: onTorso ? "hip" : "swing", motion: 60, strength: 10, mirror: onTorso, foot: true, heaviness: 100,
      colour: PART_COLOURS[body.parts.length % PART_COLOURS.length],
    };
    if (parent) parent.foot = false; // the new tip feels the ground instead
    body.parts.push(part);
    select(id);
    changed(true);
  };

  const removePart = async (id) => {
    const gone = [id, ...descendants(id)];
    if (gone.length > 1 && !(await confirmDialog("Remove this part?", `It takes ${gone.length - 1} part${gone.length > 2 ? "s" : ""} hanging on it with it.`, "Remove", "danger"))) return;
    body.parts = body.parts.filter((p) => !gone.includes(p.id));
    if (gone.includes(selected)) selected = "torso";
    paintAll();
    changed(true);
  };

  const duplicatePart = (id) => {
    const original = partById(id);
    const ids = [id, ...descendants(id)];
    const mapping = {};
    for (const oldId of ids) mapping[oldId] = null;
    const copies = [];
    for (const oldId of ids) {
      const copy = clone(partById(oldId));
      const fresh = newIdFrom(copies);
      mapping[oldId] = fresh;
      copy.id = fresh;
      copy.parent = oldId === id ? original.parent : mapping[copy.parent];
      if (oldId === id) { copy.name = `${original.name} copy`; copy.x = Math.max(-1, Math.min(1, (copy.x || 0) - 0.5)); }
      copies.push(copy);
    }
    if (body.parts.length + copies.length > 24) { toast("Too many parts", "A creature can have up to 24 parts."); return; }
    body.parts.push(...copies);
    select(copies[0].id);
    changed(true);
  };
  const newIdFrom = (extra) => {
    let n = body.parts.length + extra.length + 1;
    while (partById(`p${n}`) || extra.some((p) => p.id === `p${n}`)) n += 1;
    return `p${n}`;
  };

  // ---------- The parts tree ----------

  const paintTree = () => {
    const rows = [row("torso", "Body", `${body.torso.shape} · ${fmtNumber(body.torso.weight, 2)} kg`, 0, body.torso.colour)];
    const walk = (parentId, depth) => {
      for (const part of childrenOf(parentId)) {
        const tags = [part.shape, part.joint === "fixed" ? "fixed" : part.joint === "hip" ? "hip joint" : `${part.joint} joint`];
        if (part.mirror) tags.push("pair");
        if (part.foot) tags.push("feels ground");
        rows.push(row(part.id, part.name, tags.join(" · "), depth, part.colour));
        walk(part.id, depth + 1);
      }
    };
    walk("torso", 1);
    tree.replaceChildren(...rows);
  };

  const row = (id, name, meta, depth, colour) => h("div", {
    class: `part-row ${selected === id ? "on" : ""}`, style: { paddingLeft: `${10 + depth * 18}px` }, onclick: () => select(id) },
    h("span", { class: "swatch", style: { background: colour } }),
    h("div", { class: "part-text" }, h("div", { class: "part-name" }, name), h("div", { class: "part-meta" }, meta)),
    h("div", { class: "part-actions" },
      h("button", { class: "reset-link", title: "Attach a new part here", html: icon.plus, onclick: (event) => { event.stopPropagation(); addPart(id); } }),
      id !== "torso" ? h("button", { class: "reset-link", title: "Duplicate", html: icon.copy, onclick: (event) => { event.stopPropagation(); duplicatePart(id); } }) : null,
      id !== "torso" ? h("button", { class: "reset-link", title: "Remove", html: icon.trash, onclick: (event) => { event.stopPropagation(); removePart(id); } }) : null));

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

  const paintEditor = () => {
    if (selected === "torso") {
      const torso = body.torso;
      editor.replaceChildren(h("div", { class: "card card-pad" },
        h("div", { class: "spread", style: { marginBottom: "4px" } }, h("h3", {}, "Body"),
          h("button", { class: "btn small", onclick: () => addPart("torso"), html: `${icon.plus}<span>Attach a part</span>` })),
        h("div", { class: "faint", style: { fontSize: "12.5px" } }, "The middle of the creature. Everything else hangs on it."),
        section("Shape",
          choice(torso, "shape", "Shape", [{ value: "box", label: "Box" }, { value: "ellipsoid", label: "Egg" }, { value: "capsule", label: "Tube" }, { value: "ball", label: "Ball" }]),
          slider(torso, "length", "Length (front to back)", { min: 4, max: 120, step: 1, unit: "cm", scale: 100 }),
          torso.shape !== "ball" ? slider(torso, "width", "Width", { min: 4, max: 100, step: 1, unit: "cm", scale: 100 }) : null,
          torso.shape !== "ball" ? slider(torso, "height", "Height", { min: 2, max: 100, step: 1, unit: "cm", scale: 100 }) : null),
        section("Weight and colour",
          slider(torso, "weight", "Weight", { min: 0.05, max: 20, step: 0.05, unit: "kg" }),
          colourField(torso, "colour", "Colour"))));
      return;
    }
    const part = partById(selected);
    if (!part) { selected = "torso"; paintEditor(); return; }
    const parent = part.parent === "torso" ? null : partById(part.parent);
    const nameInput = h("input", { class: "input", value: part.name });
    nameInput.addEventListener("input", () => { part.name = nameInput.value; changed(true); });
    const mirroredByParent = insideMirrored(part);
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
    const jointHelp = h("div", { class: "field-help" }, JOINT_HELP[part.joint]);
    editor.replaceChildren(h("div", { class: "card card-pad" },
      h("div", { class: "spread", style: { marginBottom: "8px", gap: "8px" } }, nameInput,
        h("button", { class: "btn small", onclick: () => addPart(part.id), title: "Attach a part to this one", html: `${icon.plus}<span>Attach</span>` })),
      h("div", { class: "faint", style: { fontSize: "12.5px" } }, `Hangs on ${parent ? parent.name : "the body"}.`),
      section("Shape and size",
        choice(part, "shape", "Shape", [{ value: "stick", label: "Stick" }, { value: "ball", label: "Ball" }, { value: "plate", label: "Plate" }], "", paintEditor),
        ...sizes),
      section("Where it attaches", ...where,
        part.parent === "torso" || !mirroredByParent
          ? toggle(part, "mirror", "Mirrored pair", "Adds a twin on the other side, moving like a mirror image. Pairs let the brain be left/right symmetric.")
          : h("div", { class: "field-help" }, "Part of a mirrored pair: its twin follows it.")),
      section("Which way it points",
        slider(part, "pitch", "Tilt down", { min: -90, max: 90, step: 5, unit: "°", help: "90° straight down, 0° level, -90° straight up." }),
        slider(part, "yaw", "Turn", { min: -180, max: 180, step: 5, unit: "°", help: "0° forward, 90° out to the left, 180° backward." })),
      section("Joint and muscle",
        choice(part, "joint", "Joint", JOINT_CHOICES, "", () => { jointHelp.textContent = JOINT_HELP[part.joint]; paintEditor(); }),
        jointHelp,
        part.joint !== "fixed" ? slider(part, "motion", "How far it can move", { min: 5, max: 170, step: 5, unit: "°", help: "The whole swing, from one end stop to the other." }) : null,
        part.joint !== "fixed" ? slider(part, "strength", "Muscle strength", { min: 0.5, max: 100, step: 0.5, unit: "N m", help: "The strongest push of its motor." }) : null),
      section("More",
        toggle(part, "foot", "Feels the ground", "The brain is told when this part touches the ground. Give it to feet and tips."),
        slider(part, "heaviness", "Heaviness", { min: 10, max: 1000, step: 10, unit: "%", help: "100% weighs like water; the weight follows the size." }),
        colourField(part, "colour", "Colour"))));
  };

  // ---------- Templates ----------

  const templates = h("div", { class: "btn-row" }, ...animal.templates.map((template) =>
    h("button", { class: `btn small ${body.template === template.key ? "primary" : ""}`, title: template.description, onclick: async () => {
      const ok = await confirmDialog(`Start from "${template.name}"?`, `${template.description} It replaces the parts you have now (Undo brings them back).`, "Use it");
      if (!ok) return;
      history.push(JSON.stringify(body));
      body.template = template.key;
      body.torso = clone(template.torso);
      body.parts = clone(template.parts);
      lastSnapshot = JSON.stringify(body);
      selected = "torso";
      paintAll();
      paintUndo();
      onChange(true);
      [...templates.children].forEach((child) => child.classList.toggle("primary", child.textContent === template.name));
    } }, template.name)));

  const paintAll = () => { paintTree(); paintEditor(); highlight(); };

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "10px" } },
      h("div", {}, h("h3", {}, "Start from"), h("div", { class: "faint", style: { fontSize: "12.5px" } }, "A ready-made body to change, or the bare minimum.")),
      templates),
    h("div", { class: "card" },
      h("div", { class: "card-head" }, h("div", {}, h("h3", {}, "Parts"), h("div", { class: "desc" }, "Click a part here or in the 3D view to change it.")),
        h("div", { class: "row", style: { gap: "2px" } }, undoButton, redoButton)),
      h("div", { class: "card-body", style: { paddingTop: "10px" } }, tree)),
    editor);
  paintAll();
  paintUndo();
  return { element: panel, highlight };
}
