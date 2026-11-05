// New walker: four steps (creature, world, brain, training plan) and then training starts.

import { h, api, toast, debounce, settingGroups, fmtNumber, fmtDuration, askName, clear, confirmDialog } from "../ui.js";
import { icon, animalArt, worldArt } from "../icons.js";
import { Viewer } from "../viewer.js";
import { store, go, askForNotifications } from "../app.js";
import { networkDiagram } from "./network.js";
import { builderPanel } from "./builder.js";
import { scoringGlance } from "./scoring.js";

const STEPS = [
  { key: "creature", title: "Creature", sub: "Body, legs and muscles" },
  { key: "world", title: "World", sub: "Ground, physics and task" },
  { key: "brain", title: "Brain", sub: "Network and senses" },
  { key: "plan", title: "Training", sub: "How long and how" },
];

// Quick ideas for each animal: a name and the settings they change
const IDEAS = {
  duck: [
    { name: "Stilt walker", values: { leg_length: 175, leg_thickness: 80 } },
    { name: "Big feet", values: { foot_size: 190 } },
    { name: "Chunky", values: { body_width: 140, body_weight: 170 } },
    { name: "Strongman", values: { hip_strength: 250, knee_strength: 250, ankle_strength: 250 } },
    { name: "Duckling", values: { overall_size: 60, head_size: 140 } },
    { name: "Long neck", values: { neck_length: 200, head_size: 90 } },
  ],
  puppy: [
    { name: "Dachshund", values: { body_length: 175, front_leg_length: 55, back_leg_length: 55, ear_size: 170 } },
    { name: "Greyhound", values: { front_leg_length: 150, back_leg_length: 160, body_width: 80, tail_length: 160 } },
    { name: "Bulldog", values: { body_width: 150, body_length: 85, front_leg_length: 75, back_leg_length: 70, head_size: 130 } },
    { name: "Giraffe pup", values: { front_leg_length: 200, back_leg_length: 170, neck_length: 220 } },
    { name: "Big pup", values: { overall_size: 150 } },
    { name: "Strong legs", values: { hip_strength: 250, knee_strength: 250 } },
  ],
};

// Which shapes light up while a setting is being changed
const HIGHLIGHT = {
  overall_size: [""], body_length: ["body", "tail", "wing"], body_width: ["body"], neck_length: ["neck"],
  head_size: ["head", "beak", "snout"], tail_size: ["tail"], tail_length: ["tail"], wing_size: ["wing"],
  leg_length: ["thigh", "shin"], leg_thickness: ["thigh", "shin", "upper", "lower"], leg_position: ["thigh", "shin", "foot", "toes"],
  leg_spread: ["thigh", "shin", "foot", "toes", "upper", "lower", "paw"], foot_size: ["foot", "toes"], foot_grip: ["foot", "toes", "paw"],
  has_toes: ["toes"], body_weight: ["body", "tail", "wing"], head_weight: ["head", "neck", "beak", "snout"],
  leg_weight: ["thigh", "shin", "foot", "toes", "upper", "lower", "paw"], hip_strength: ["thigh", "upper"],
  knee_strength: ["shin", "lower"], ankle_strength: ["foot"], toe_strength: ["toes"], neck_strength: ["neck"],
  tail_strength: ["tail"], neck_moves: ["neck"], tail_moves: ["tail"], front_leg_length: ["front"],
  back_leg_length: ["back"], paw_size: ["paw"], body_colour: ["body", "tail", "neck", "head"], wing_colour: ["wing"],
  leg_colour: ["thigh", "shin", "foot", "toes"], beak_colour: ["beak"], fur_colour: ["body", "upper", "lower", "neck", "head", "tail", "snout"],
  paw_colour: ["paw"], stiffness: ["thigh", "shin", "foot", "toes", "upper", "lower", "neck"],
  flexibility: ["thigh", "shin", "foot", "toes", "upper", "lower", "neck", "tail"],
};

export async function render(main, params) {
  const stepKey = params[0] || "creature";
  const stepIndex = Math.max(0, STEPS.findIndex((step) => step.key === stepKey));
  const draft = store.draft;
  const page = h("div", { class: "page page-wide" });
  main.append(page);

  const head = h("div", { class: "page-head" },
    h("div", {}, h("div", { class: "eyebrow" }, "New walker"), h("h1", {}, pageTitle(stepIndex)),
      h("div", { class: "sub" }, pageSubtitle(stepIndex))),
    h("div", { class: "btn-row" },
      h("button", { class: "btn ghost", onclick: async () => {
        const ok = await confirmDialog("Start over?", "This clears the creature, world, brain and plan you are working on.", "Start over", "danger");
        if (ok) { store.resetDraft(draft.creature.animal); go("#/new/creature"); }
      }, html: `${icon.restart}<span>Start over</span>` })));
  page.append(head, stepper(stepIndex));

  const body = h("div");
  page.append(body);
  const steps = [creatureStep, worldStep, brainStep, planStep];
  const cleanup = await steps[stepIndex](body, draft);

  page.append(h("div", { class: "wizard-foot" },
    stepIndex > 0 ? h("button", { class: "btn", onclick: () => go(`#/new/${STEPS[stepIndex - 1].key}`), html: `${icon.arrowLeft}<span>Back: ${STEPS[stepIndex - 1].title}</span>` }) : h("span"),
    stepIndex < STEPS.length - 1
      ? h("button", { class: "btn primary big", onclick: () => go(`#/new/${STEPS[stepIndex + 1].key}`), html: `<span>Next: ${STEPS[stepIndex + 1].title}</span>${icon.arrowRight}` })
      : h("span")));
  return cleanup;
}

function pageTitle(index) {
  return ["Design your creature", "Choose its world", "Shape its brain", "Plan the training"][index];
}

function pageSubtitle(index) {
  return [
    "Pick an animal and shape it with the sliders. The 3D view updates as you go. Try the drop test to see if it can stand.",
    "Where it practises: the ground, gravity, the speeds it is asked to walk at, and what earns it points.",
    "The neural network that will learn to move the muscles. The standard brain works well; change it to experiment.",
    "How long it practises and how it studies. Then press Start and watch it learn.",
  ][index];
}

function stepper(active) {
  const draft = store.draft;
  const subs = [draft.creature.name, draft.world.name, brainSummary(draft.brain), planSummary(draft.learning)];
  const items = [];
  STEPS.forEach((step, index) => {
    if (index) items.push(h("div", { class: "step-line" }));
    items.push(h("div", { class: `step ${index === active ? "on" : ""} ${index < active ? "done" : ""}`, onclick: () => go(`#/new/${step.key}`) },
      h("span", { class: "num", html: index < active ? icon.check : String(index + 1) }),
      h("div", { style: { minWidth: 0 } }, h("div", {}, step.title), h("div", { class: "step-sub" }, subs[index] || step.sub))));
  });
  return h("div", { class: "stepper" }, ...items);
}

function brainSummary(brain) {
  return `${brain.layers.length} layer${brain.layers.length > 1 ? "s" : ""} of ${brain.layers.join(", ")}`;
}

function planSummary(learning) {
  return `${fmtNumber(learning.million_steps, 1)} million steps`;
}

function persist() {
  store.saveDraft();
}

// ---------- Step 1: the creature ----------

async function creatureStep(body, draft) {
  const panel = h("div", { class: "panel" });
  const stageBox = h("div", { class: "viewer-tall" });
  const stage = h("div", { class: "stage" }, stageBox,
    h("div", { class: "btn-row", style: { marginTop: "12px", justifyContent: "space-between" } },
      h("div", { class: "btn-row" },
        h("button", { class: "btn primary", onclick: () => dropTest(), html: `${icon.play}<span>Drop test: will it stand?</span>` }),
        h("button", { class: "btn", onclick: () => refresh(true), html: `${icon.camera}<span>Standing pose</span>` })),
      h("span", { class: "faint" }, "Drag to turn the view, scroll to zoom.")));
  body.append(h("div", { class: "workbench" }, panel, stage));

  const viewer = new Viewer(stageBox, { autoplay: true, loop: false });
  const animal = store.animal(draft.creature.animal);

  let latest = 0;  // only the newest preview is shown, even if an older one answers later
  const refresh = async (reframe = false) => {
    const request = ++latest;
    viewer.setLoading(true);
    try {
      const preview = await api("preview/creature", { method: "POST", body: { animal: draft.creature.animal, values: draft.creature.values } });
      if (request !== latest || viewer.disposed) return;
      viewer.load(preview, { keepCamera: !reframe && viewer.lastAnimal === draft.creature.animal });
      viewer.lastAnimal = draft.creature.animal;
      const facts = preview.facts;
      const chips = [["Weight", `${fmtNumber(facts.mass_kg, 2)} kg`], ["Height", `${fmtNumber(facts.height_cm, 0)} cm`],
        ["Motors", String(facts.motors)], ["Feels the ground with", `${facts.feet} part${facts.feet === 1 ? "" : "s"}`],
        ["Strongest muscle", `${fmtNumber(facts.strongest_motor_nm, 1)} N m`]];
      if (draft.creature.animal === "custom") chips.push(["Symmetric brain", facts.symmetric_ok ? "possible" : "not possible (lopsided)"]);
      if (preview.support && preview.support.length >= 3) chips.push(["Balance", balanceWord(preview)]);
      viewer.setChips(chips);
      if (!facts.motors) viewer.banner("No motors yet: give a part a joint so the brain has something to move.", "bad", 0);
      viewer.banner("");
    } catch (error) {
      toast("Could not build this creature", error.message, "bad");
    } finally {
      viewer.setLoading(false);
    }
  };
  const refreshSoon = debounce(() => refresh(false), 140);

  const dropTest = async () => {
    const request = ++latest;
    viewer.setLoading(true);
    try {
      const replay = await api("preview/drop", { method: "POST", body: { animal: draft.creature.animal, values: draft.creature.values, seconds: 4 } });
      if (request !== latest || viewer.disposed) return;
      viewer.load(replay, { keepCamera: true });
      viewer.lastAnimal = draft.creature.animal;
      const stats = replay.stats;
      if (stats.stood) viewer.banner("It stands: its muscles are strong enough to hold its pose.", "good", 6);
      else viewer.banner(`It falls over after ${fmtNumber(stats.seconds, 1)} s holding its pose. Learning may still fix that, or try stronger muscles or a wider stance.`, "bad", 8);
    } catch (error) {
      toast("The drop test failed", error.message, "bad");
    } finally {
      viewer.setLoading(false);
    }
  };

  // Animal picker
  const picker = h("div", { class: "picker" });
  for (const option of store.catalog.animals) {
    const on = option.key === draft.creature.animal;
    const colours = on ? draft.creature.values : option.defaults;
    picker.append(h("button", { class: `pick ${on ? "on" : ""}`, onclick: () => {
      if (option.key === draft.creature.animal) return;
      draft.creature = { animal: option.key, name: option.key === "custom" ? "My creation" : `My ${option.name.toLowerCase()}`,
        values: JSON.parse(JSON.stringify(option.defaults)), id: null };
      persist();
      go("#/new/creature");
    } },
      h("span", { class: "check", html: icon.check }),
      h("div", { class: "art", style: { background: "var(--surface-2)" }, html: animalArt(option.key, colours) }),
      h("div", { class: "pick-name" }, option.name), h("div", { class: "pick-desc" }, option.description)));
  }

  const nameInput = h("input", { class: "input", value: draft.creature.name, placeholder: "Name your creature" });
  nameInput.addEventListener("input", () => { draft.creature.name = nameInput.value; persist(); });

  // Saved creatures of this animal
  const saved = (await api("creatures")).filter((item) => item.animal === draft.creature.animal);
  const savedSelect = h("select", { class: "input" },
    h("option", { value: "" }, saved.length ? "Load a saved creature..." : "No saved creatures yet"),
    ...saved.map((item) => h("option", { value: item.id, selected: item.id === draft.creature.id }, item.name)));
  savedSelect.addEventListener("change", () => {
    const item = saved.find((entry) => entry.id === savedSelect.value);
    if (!item) return;
    draft.creature = { animal: item.animal, name: item.name, values: { ...JSON.parse(JSON.stringify(animal.defaults)), ...item.values }, id: item.id };
    persist();
    go("#/new/creature");
  });

  const ideas = h("div", { class: "btn-row" }, ...(IDEAS[animal.key] || []).map((idea) =>
    h("button", { class: "btn small", onclick: () => {
      Object.assign(draft.creature.values, idea.values);
      for (const [key, value] of Object.entries(idea.values)) groups.controls[key] && groups.controls[key].setValue(value);
      if (!draft.creature.name || draft.creature.name.startsWith("My ")) { draft.creature.name = idea.name; nameInput.value = idea.name; }
      persist();
      refresh(false);
      toast(`Applied "${idea.name}"`, "Every slider can still be changed.");
    } }, idea.name)));

  const isCustom = animal.key === "custom";
  const builder = isCustom ? builderPanel(draft.creature, animal, viewer, (now) => { persist(); if (now) refresh(true); else refreshSoon(); }) : null;
  const groups = settingGroups(animal.groups, draft.creature.values, () => { persist(); refreshSoon(); }, {
    open: ["size", "legs"], defaults: animal.defaults,
    onHover: (key) => {
      const parts = key ? HIGHLIGHT[key] : null;
      viewer.highlight(parts ? (name) => parts.some((part) => (part === "" ? true : name.includes(part))) : null);
    },
  });

  const saveCreature = async (asNew) => {
    const name = await askName(asNew || !draft.creature.id ? "Save this creature" : "Save changes", "It goes into your library, so you can use it again later.", draft.creature.name);
    if (!name) return;
    const item = await api("creatures", { method: "POST", body: { id: asNew ? null : draft.creature.id, name, animal: draft.creature.animal,
      values: draft.creature.values, thumbnail: viewer.thumbnailUrl() } });
    draft.creature.id = item.id;
    draft.creature.name = item.name;
    nameInput.value = item.name;
    persist();
    toast("Saved to your library", item.name, "good");
  };

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "14px", marginBottom: "12px" } },
      h("div", {}, h("h3", {}, "Animal"), h("div", { class: "faint", style: { fontSize: "12.5px" } }, "Each animal has its own body plan.")),
      picker,
      h("div", { class: "grid grid-2", style: { gap: "10px" } },
        h("div", {}, h("div", { class: "field-label", style: { marginBottom: "6px" } }, "Name"), nameInput),
        h("div", {}, h("div", { class: "field-label", style: { marginBottom: "6px" } }, "From your library"), savedSelect)),
      isCustom ? null : h("div", {}, h("div", { class: "field-label", style: { marginBottom: "8px" } }, "Quick ideas"), ideas),
      h("div", { class: "btn-row" },
        h("button", { class: "btn small", onclick: () => saveCreature(false), html: `${icon.save}<span>${draft.creature.id ? "Save changes" : "Save to library"}</span>` }),
        draft.creature.id ? h("button", { class: "btn small ghost", onclick: () => saveCreature(true), html: `${icon.copy}<span>Save as new</span>` }) : null,
        isCustom ? null : h("button", { class: "btn small ghost", onclick: () => {
          draft.creature.values = JSON.parse(JSON.stringify(animal.defaults));
          persist();
          go("#/new/creature");
        }, html: `${icon.reset}<span>All standard</span>` }))),
    ...(builder ? [builder.element, h("div", { style: { height: "12px" } })] : []),  // native append would write "null" as text
    groups);

  await refresh(true);
  return () => viewer.dispose();
}

// ---------- Step 2: the world ----------

async function worldStep(body, draft) {
  const panel = h("div", { class: "panel" });
  const stageBox = h("div", { class: "viewer-mid" });
  const glance = h("div");
  const stage = h("div", { class: "stage" }, stageBox,
    h("div", { class: "btn-row", style: { marginTop: "12px", justifyContent: "space-between" } },
      h("button", { class: "btn", onclick: () => show(), html: `${icon.restart}<span>New ground</span>` }),
      h("span", { class: "faint" }, "Your creature stands in the world with its muscles holding still.")),
    glance);
  const paintGlance = () => glance.replaceChildren(scoringGlance(draft.world.values));
  body.append(h("div", { class: "workbench" }, panel, stage));
  const viewer = new Viewer(stageBox, { autoplay: true, loop: false });
  const catalog = store.catalog;
  const worldDefaults = Object.fromEntries(catalog.world_groups.flatMap((group) => group.settings.map((s) => [s.key, s.default])));

  let latest = 0;
  const show = async () => {
    const request = ++latest;
    viewer.setLoading(true);
    try {
      const replay = await api("preview/drop", { method: "POST", body: { animal: draft.creature.animal, values: draft.creature.values, world: draft.world, seconds: 3 } });
      if (request !== latest || viewer.disposed) return;
      viewer.follow = false;  // the world stays in view while the creature stands
      viewer.load(replay);
      viewer.frameWide(1.2);
      const v = draft.world.values;
      const terrainName = (catalog.world_groups[0].settings[0].choices.find((c) => c.value === v.terrain) || {}).label;
      viewer.setChips([["Ground", v.terrain === "flat" || !v.terrain_height ? "Flat" : `${terrainName}, up to ${fmtNumber(v.terrain_height, 1)} cm`],
        ["Gravity", `${fmtNumber(v.gravity, 1)} m/s²`], ["Asked speeds", `${fmtNumber(v.slowest_speed, 2)} to ${fmtNumber(v.fastest_speed, 2)} m/s`],
        ["Grip", `${fmtNumber(v.ground_grip)}%`]]);
    } catch (error) {
      toast("Could not build this world", error.message, "bad");
    } finally {
      viewer.setLoading(false);
    }
  };
  const showSoon = debounce(show, 350);

  const templates = catalog.world_templates;
  const mine = await api("worlds");
  const gallery = (items, title) => h("div", { class: "stack", style: { gap: "8px" } },
    h("div", { class: "field-label" }, title),
    h("div", { class: "picker", style: { gridTemplateColumns: "repeat(auto-fill, minmax(124px, 1fr))" } },
      ...items.map((world) => h("button", { class: `pick ${world.id === draft.world.id ? "on" : ""}`, title: world.description, onclick: () => {
        draft.world = JSON.parse(JSON.stringify({ ...world, thumbnail: undefined, values: { ...worldDefaults, ...world.values } }));
        if (!templates.includes(world)) draft.world.builtin = false;
        persist();
        go("#/new/world");
      } },
        h("span", { class: "check", html: icon.check }),
        h("div", { class: "art", style: { height: "58px" }, html: worldArt({ ...worldDefaults, ...world.values }) }),
        h("div", { class: "pick-name", style: { fontSize: "13px" } }, world.name)))));

  const description = h("p", { class: "muted", style: { fontSize: "13px" } }, draft.world.description || "");
  const groupOf = (key) => (catalog.world_groups.find((group) => group.settings.some((s) => s.key === key)) || {}).key;
  const groups = settingGroups(catalog.world_groups, draft.world.values, (key) => {
    if (draft.world.builtin) { draft.world.builtin = false; draft.world.id = null; draft.world.name = `${draft.world.name} (changed)`; nameText.textContent = draft.world.name; }
    persist();
    paintGlance();
    if (!["reward", "rules"].includes(groupOf(key))) showSoon();
  }, { open: ["ground", "task", "reward"], defaults: worldDefaults });

  const nameText = h("h3", {}, draft.world.name);
  const saveWorld = async () => {
    const name = await askName("Save this world", "It goes into your library, so you can use it again.", draft.world.name);
    if (!name) return;
    const item = await api("worlds", { method: "POST", body: { ...draft.world, name, thumbnail: viewer.thumbnailUrl() } });
    draft.world = item;
    persist();
    nameText.textContent = item.name;
    toast("World saved", item.name, "good");
  };

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "14px", marginBottom: "12px" } },
      gallery(templates, "Ready-made worlds"),
      mine.length ? gallery(mine, "Your worlds") : null),
    h("div", { class: "card card-pad", style: { marginBottom: "12px" } },
      h("div", { class: "spread" }, nameText, h("button", { class: "btn small", onclick: saveWorld, html: `${icon.save}<span>Save world</span>` })),
      description,
      h("div", { style: { marginTop: "14px" } }, h("div", { class: "field-label", style: { marginBottom: "8px" } }, "Scoring style"),
        h("div", { class: "btn-row" }, ...SCORING_STYLES.map((style) => h("button", { class: "btn small", title: style.description, onclick: () => {
          const changes = { ...SCORING_RESET, ...style.values };
          Object.assign(draft.world.values, changes);
          for (const [key, value] of Object.entries(changes)) if (groups.controls[key]) groups.controls[key].setValue(value);
          if (draft.world.builtin) { draft.world.builtin = false; draft.world.id = null; draft.world.name = `${draft.world.name} (${style.name.toLowerCase()})`; nameText.textContent = draft.world.name; }
          groups.querySelectorAll(".group").forEach((card) => card.dispatchEvent(new Event("refresh")));
          persist();
          paintGlance();
          toast(`Scoring style: ${style.name}`, style.description);
        } }, style.name))))),
    groups);

  paintGlance();
  await show();
  return () => viewer.dispose();
}

// Ready-made scoring styles: the main points and the style rules together
const SCORING_RESET = {
  speed_shape: "target", speed_points: 3.0, alive_points: 0.2, energy_cost: 1.5, straight_cost: 0.5, fall_cost: 0,
  end_stops: true, end_stop_cost: 0.05, level_body: false, no_crouch: false, smooth_moves: false, lift_feet: false,
  face_forward: false, steady_ride: false,
};
const SCORING_STYLES = [
  { name: "Classic", description: "learn_to_walk's scoring: the asked speed, staying up, energy and walking straight, with a gentle joint limit penalty.", values: {} },
  { name: "Graceful", description: "Smooth, level and steady: calm walking that looks natural.", values: { smooth_moves: true, smooth_cost: 0.3, level_body: true, level_cost: 0.5, steady_ride: true, bounce_cost: 0.5, no_crouch: true, crouch_cost: 1.0 } },
  { name: "Athletic", description: "Big steps and plenty of speed: cheaper energy and points for lifting the feet.", values: { speed_points: 4.0, energy_cost: 0.8, lift_feet: true, lift_points: 1.0 } },
  { name: "Thrifty", description: "Every joule counts: expensive energy and smooth moves. Slow but efficient.", values: { energy_cost: 2.5, smooth_moves: true, smooth_cost: 0.2 } },
  { name: "Anything goes", description: "Only forward points and staying up: no costs at all. Expect strange gaits.", values: { energy_cost: 0, straight_cost: 0, end_stops: false } },
];

// ---------- Step 3: the brain ----------

async function brainStep(body, draft) {
  const catalog = store.catalog;
  const panel = h("div", { class: "panel" });
  const diagram = h("div", { class: "card card-pad" });
  const facts = h("div", { class: "stats", style: { marginTop: "14px" } });
  const explain = h("div", { style: { marginTop: "14px" } });
  const stage = h("div", { class: "stage" }, diagram, facts, explain);
  body.append(h("div", { class: "workbench" }, panel, stage));
  const brain = draft.brain;
  const brainDefaults = catalog.brain_default.values;

  const update = async () => {
    persist();
    try {
      const info = await api("preview/brain", { method: "POST", body: { animal: draft.creature.animal, values: draft.creature.values, design: brain } });
      clear(diagram).append(h("div", { class: "spread", style: { marginBottom: "8px" } },
        h("h3", {}, "The network"), h("span", { class: "pill accent" }, `${fmtNumber(info.weights.total)} numbers to learn`)),
        networkDiagram(info, brain));
      facts.replaceChildren(
        statCard("Senses in", String(info.inputs), "numbers it feels each moment"),
        statCard("Thinking cells", String(brain.layers.reduce((a, b) => a + b, 0)), `${brain.layers.length} layer${brain.layers.length > 1 ? "s" : ""}`),
        statCard("Commands out", String(info.outputs), "one per motor"),
        statCard("Judge", fmtNumber(info.weights.judge), "numbers in the judge network"));
      if (!info.symmetric_ok && brain.values.symmetric) {
        explain.replaceChildren(h("div", { class: "callout warn", html: `${icon.info}<div><b>This creature is lopsided</b>: it has parts off the middle line without a mirrored twin, so its brain cannot be left/right symmetric. It will get a free brain instead. Make those parts mirrored pairs (or put them on the middle line) to use the symmetric one.</div>` }));
        return;
      }
      explain.replaceChildren(h("div", { class: `callout ${brain.values.symmetric ? "good" : ""}`, html: `${icon.info}<div>${brain.values.symmetric
        ? `<b>Symmetric brain.</b> One network thinks for the left side and, looking at the world mirrored, for the right side. The ${info.left_motors} motors of each side share what they learn${info.middle_motors ? `, and the ${info.middle_motors} middle motor${info.middle_motors > 1 ? "s" : ""} (neck, tail) combine both views` : ""}. Walking gaits come much sooner.`
        : "<b>Free brain.</b> Every motor has its own outputs and the two sides learn separately. Slower to learn, but it can find lopsided ways to move."}</div>` }));
    } catch (error) {
      toast("Could not draw the brain", error.message, "bad");
    }
  };

  const presets = h("div", { class: "picker", style: { gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" } },
    ...catalog.brain_presets.map((preset) => h("button", {
      class: `pick ${draft.brainPreset === preset.id ? "on" : ""}`, onclick: () => {
        brain.layers = [...preset.layers];
        if (brain.values.judge_same) brain.judge_layers = [...preset.layers];
        draft.brainPreset = preset.id;
        persist();
        go("#/new/brain");
      } },
      h("span", { class: "check", html: icon.check }),
      h("div", { class: "pick-name" }, preset.name),
      h("div", { class: "pick-desc" }, preset.description))));

  const layersBox = h("div");
  const paintLayers = () => {
    layersBox.replaceChildren(
      layerEditor("Brain layers", brain.layers, (layers) => { brain.layers = layers; if (brain.values.judge_same) brain.judge_layers = [...layers]; draft.brainPreset = null; paintLayers(); update(); }),
      brain.values.judge_same ? null : h("div", { style: { marginTop: "14px" } },
        layerEditor("Judge layers", brain.judge_layers, (layers) => { brain.judge_layers = layers; paintLayers(); update(); })));
  };
  paintLayers();

  const groups = settingGroups(catalog.brain_groups, brain.values, (key) => {
    if (key === "judge_same" && brain.values.judge_same) brain.judge_layers = [...brain.layers];
    paintLayers();
    update();
  }, { open: ["cells", "senses"], defaults: brainDefaults });

  const savedDesigns = await api("brains");
  const loadSelect = h("select", { class: "input" }, h("option", { value: "" }, savedDesigns.length ? "Load a saved brain..." : "No saved brains yet"),
    ...savedDesigns.map((item) => h("option", { value: item.id }, item.name)));
  loadSelect.addEventListener("change", () => {
    const item = savedDesigns.find((entry) => entry.id === loadSelect.value);
    if (!item) return;
    draft.brain = { values: { ...brainDefaults, ...item.design.values }, layers: item.design.layers, judge_layers: item.design.judge_layers };
    draft.brainPreset = null;
    persist();
    go("#/new/brain");
  });

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "14px", marginBottom: "12px" } },
      h("div", {}, h("h3", {}, "Starting point"), h("div", { class: "faint", style: { fontSize: "12.5px" } }, "Pick a ready-made size, then fine tune below.")),
      presets,
      h("div", { class: "row" }, loadSelect, h("button", { class: "btn", onclick: async () => {
        const name = await askName("Save this brain design", "Keep the layers and settings to use again.", `${brain.layers.join(" x ")} brain`);
        if (!name) return;
        await api("brains", { method: "POST", body: { name, design: brain } });
        toast("Brain design saved", name, "good");
      }, html: `${icon.save}<span>Save</span>` }))),
    h("div", { class: "card card-pad", style: { marginBottom: "12px" } }, layersBox),
    groups);
  await update();
  return null;
}

const LAYER_SIZES = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

function layerEditor(title, layers, onChange) {
  const nextSize = (size, direction) => {
    const index = LAYER_SIZES.findIndex((s) => s >= size);
    const at = index < 0 ? LAYER_SIZES.length - 1 : index;
    return LAYER_SIZES[Math.max(0, Math.min(LAYER_SIZES.length - 1, at + direction))];
  };
  const boxes = layers.map((size, index) => h("div", { class: "layer-box" },
    h("div", { class: "spread" }, h("span", { class: "title" }, `Layer ${index + 1}`),
      layers.length > 1 ? h("button", { class: "reset-link", title: "Remove this layer", html: icon.close, onclick: () => onChange(layers.filter((_, i) => i !== index)) }) : null),
    h("div", { class: "size" }, String(size), h("small", {}, " cells")),
    h("div", { class: "btn-row" },
      h("button", { class: "btn small", onclick: () => onChange(layers.map((s, i) => (i === index ? nextSize(s, -1) : s))) }, "Fewer"),
      h("button", { class: "btn small", onclick: () => onChange(layers.map((s, i) => (i === index ? nextSize(s, 1) : s))) }, "More"))));
  const add = layers.length < 6 ? h("button", { class: "add-layer", onclick: () => onChange([...layers, layers[layers.length - 1] || 64]), html: `${icon.plus}<span>Add layer</span>` }) : null;
  return h("div", {}, h("div", { style: { marginBottom: "10px" } }, h("h3", {}, title),
    h("div", { class: "faint", style: { fontSize: "12.5px" } }, "More layers and cells can learn more, but need more practice.")),
    h("div", { class: "layer-editor" }, ...boxes, add));
}

function statCard(label, value, note) {
  return h("div", { class: "stat" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), h("div", { class: "note" }, note));
}

// ---------- Step 4: the training plan ----------

async function planStep(body, draft) {
  const catalog = store.catalog;
  const learning = draft.learning;
  const left = h("div", { class: "panel" });
  const right = h("div", { class: "stage stack" });
  body.append(h("div", { class: "workbench" }, left, right));
  const learningDefaults = catalog.learning_default;

  const nameInput = h("input", { class: "input", value: draft.name || defaultName(draft), placeholder: "Name this walker" });
  nameInput.addEventListener("input", () => { draft.name = nameInput.value; persist(); });

  const estimate = h("div");
  const paintEstimate = () => {
    const worlds = learning.parallel_worlds;
    const measured = (catalog.system.speeds || {})[String(worlds)];
    const cores = catalog.system.cores || 4;
    const speed = measured || 260 * Math.min(worlds, cores) + 60;
    const totalMillion = learning.million_steps + (draft.course || []).reduce((sum, stage) => sum + stage.million_steps, 0);
    const seconds = totalMillion * 1e6 / speed;
    const busy = store.trainings.filter((t) => ["starting", "training", "examining"].includes(t.status.state));
    const warn = [
      worlds > cores ? h("div", { class: "callout warn", style: { marginTop: "10px" }, html: `${icon.info}<div>This computer has ${cores} processor cores. More worlds than cores will not go faster.</div>` }) : null,
      busy.length ? h("div", { class: "callout warn", style: { marginTop: "10px" }, html: `${icon.info}<div>${busy.map((t) => `"${t.name}"`).join(", ")} ${busy.length > 1 ? "are" : "is"} training right now. Both will share the computer and each will go about half as fast. You can pause it on its page.</div>` }) : null,
    ];
    estimate.replaceChildren(h("div", { class: "stats" },
      statCard("Practice", `${fmtNumber(totalMillion, 1)}M`, (draft.course || []).length ? `steps in ${draft.course.length + 1} stages` : "steps in total"),
      statCard("Practice time", fmtDuration(totalMillion * 1e6 * 0.02), "of simulated walking"),
      statCard("Estimated time", fmtDuration(seconds), measured ? "about, at the speed measured on this computer" : "a first guess, better after one run")), ...warn);
  };

  const presets = h("div", { class: "picker", style: { gridTemplateColumns: "repeat(3, 1fr)" } },
    ...catalog.learning_presets.map((preset) => h("button", { class: `pick ${draft.learningPreset === preset.id ? "on" : ""}`, onclick: () => {
      Object.assign(learning, preset.values);
      draft.learningPreset = preset.id;
      persist();
      go("#/new/plan");
    } }, h("span", { class: "check", html: icon.check }), h("div", { class: "pick-name" }, preset.name), h("div", { class: "pick-desc" }, preset.description))));

  const groups = settingGroups(catalog.learning_groups, learning, () => { draft.learningPreset = null; persist(); paintEstimate(); paintCourse(); }, { open: ["length"], defaults: learningDefaults });

  // A course: more worlds to practise in after this one, in order
  const worldDefaults = Object.fromEntries(catalog.world_groups.flatMap((g) => g.settings.map((s) => [s.key, s.default])));
  const courseWorlds = [...catalog.world_templates, ...(await api("worlds"))];
  draft.course = draft.course || [];
  const courseBox = h("div", { class: "stack", style: { gap: "8px" } });
  const paintCourse = () => {
    const rows = draft.course.map((stage, index) => {
      const select = h("select", { class: "input" }, ...courseWorlds.map((w) => h("option", { value: w.id, selected: w.id === stage.world.id }, w.name)));
      select.addEventListener("change", () => {
        const w = courseWorlds.find((item) => item.id === select.value);
        stage.world = { ...JSON.parse(JSON.stringify(w)), values: { ...worldDefaults, ...w.values } };
        persist();
      });
      const steps = h("input", { class: "input", type: "number", min: 0.1, max: 100, step: 0.5, value: stage.million_steps, style: { width: "92px" } });
      steps.addEventListener("change", () => { stage.million_steps = Math.max(0.1, Number(steps.value) || 2); persist(); paintEstimate(); });
      return h("div", { class: "row" }, h("span", { class: "faint", style: { width: "64px", flex: "none" } }, `Stage ${index + 2}`), select, steps,
        h("span", { class: "faint" }, "million"),
        h("button", { class: "reset-link", title: "Remove this stage", html: icon.trash, onclick: () => { draft.course.splice(index, 1); persist(); paintCourse(); paintEstimate(); } }));
    });
    courseBox.replaceChildren(
      h("div", { class: "row" }, h("span", { class: "faint", style: { width: "64px", flex: "none" } }, "Stage 1"),
        h("span", { style: { fontWeight: 600 } }, `${draft.world.name}, ${fmtNumber(learning.million_steps, 1)} million steps`)),
      ...rows,
      draft.course.length < 5 ? h("button", { class: "btn small", style: { alignSelf: "flex-start" }, onclick: () => {
        const w = courseWorlds[1] || courseWorlds[0];
        draft.course.push({ world: { ...JSON.parse(JSON.stringify(w)), values: { ...worldDefaults, ...w.values } }, million_steps: 2 });
        persist(); paintCourse(); paintEstimate();
      }, html: `${icon.plus}<span>Add a stage</span>` }) : null);
  };
  paintCourse();

  // Starting from a brain that is already trained
  const trainings = (await api("trainings")).filter((t) => t.has_brain && t.creature.animal === draft.creature.animal);
  const startSelect = h("select", { class: "input" },
    h("option", { value: "" }, "A brand new brain"),
    ...trainings.map((t) => h("option", { value: t.id, selected: draft.start_from === t.id }, `The best brain of "${t.name}"`)));
  startSelect.addEventListener("change", () => { draft.start_from = startSelect.value || null; persist(); });

  left.append(
    h("div", { class: "card card-pad stack", style: { gap: "14px", marginBottom: "12px" } },
      h("div", {}, h("div", { class: "field-label", style: { marginBottom: "6px" } }, "Name"), nameInput),
      h("div", {}, h("div", { class: "field-label", style: { marginBottom: "8px" } }, "How long"), presets),
      h("div", {}, h("div", { class: "field-label", style: { marginBottom: "6px" } }, "Start with"), startSelect,
        h("div", { class: "field-help" }, "Teach an old walker a new world: start from a trained brain of the same animal. It needs the same motors, senses and brain layers."))),
    h("div", { class: "card card-pad stack", style: { gap: "10px", marginBottom: "12px" } },
      h("div", {}, h("h3", {}, "Course"), h("div", { class: "faint", style: { fontSize: "12.5px" } },
        "Optional: after its world, it moves on to more worlds, one after the other, with the same brain. Like school: flat ground first, then cobblestones, then stairs.")),
      courseBox),
    groups);

  const animal = store.animal(draft.creature.animal);
  const brain = draft.brain;
  const values = draft.world.values;
  const startButton = h("button", { class: "btn primary big", style: { width: "100%" }, html: `${icon.play}<span>Start training</span>` });
  startButton.addEventListener("click", async () => {
    askForNotifications();
    startButton.disabled = true;
    startButton.innerHTML = `<span class="spinner"></span><span>Starting...</span>`;
    try {
      const setup = await api("trainings", { method: "POST", body: {
        name: nameInput.value.trim() || defaultName(draft), creature: draft.creature, world: draft.world, brain: draft.brain,
        learning: draft.learning, start_from: draft.start_from, course: draft.course || [] } });
      toast("Training started", "It first builds its practice worlds, then starts learning.", "good");
      draft.name = "";
      persist();
      go(`#/walker/${setup.id}`);
    } catch (error) {
      toast("Could not start", error.message, "bad");
      startButton.disabled = false;
      startButton.innerHTML = `${icon.play}<span>Start training</span>`;
    }
  });

  right.append(
    h("div", { class: "card card-pad summary-card" },
      h("div", { class: "icon-head" }, h("div", { class: "art", html: animalArt(animal.key, draft.creature.values) }),
        h("div", {}, h("div", { class: "faint", style: { fontSize: "12px" } }, "Creature"), h("h3", {}, draft.creature.name || animal.name)),
        h("button", { class: "btn small ghost", style: { marginLeft: "auto" }, onclick: () => go("#/new/creature") }, "Change")),
      h("dl", { class: "kv" }, h("dt", {}, "Animal"), h("dd", {}, animal.name),
        ...(animal.key === "custom"
          ? [h("dt", {}, "Parts"), h("dd", {}, String(draft.creature.values.body.parts.length + 1))]
          : [h("dt", {}, "Changed settings"), h("dd", {}, String(changedCount(draft.creature.values, animal.defaults)))]))),
    h("div", { class: "card card-pad summary-card" },
      h("div", { class: "icon-head" }, h("div", { class: "art", style: { overflow: "hidden" }, html: worldArt(values) }),
        h("div", {}, h("div", { class: "faint", style: { fontSize: "12px" } }, "World"), h("h3", {}, draft.world.name)),
        h("button", { class: "btn small ghost", style: { marginLeft: "auto" }, onclick: () => go("#/new/world") }, "Change")),
      h("dl", { class: "kv" }, h("dt", {}, "Ground"), h("dd", {}, values.terrain === "flat" || !values.terrain_height ? "Flat" : `${values.terrain}, up to ${values.terrain_height} cm`),
        h("dt", {}, "Asked speeds"), h("dd", {}, `${values.slowest_speed} to ${values.fastest_speed} m/s`),
        h("dt", {}, "Gravity"), h("dd", {}, `${values.gravity} m/s²`))),
    h("div", { class: "card card-pad summary-card" },
      h("div", { class: "icon-head" }, h("div", { class: "art", html: `<div style="width:24px;height:24px;color:var(--accent)">${icon.brain}</div>` }),
        h("div", {}, h("div", { class: "faint", style: { fontSize: "12px" } }, "Brain"), h("h3", {}, brainSummary(brain))),
        h("button", { class: "btn small ghost", style: { marginLeft: "auto" }, onclick: () => go("#/new/brain") }, "Change")),
      h("dl", { class: "kv" }, h("dt", {}, "Symmetric"), h("dd", {}, brain.values.symmetric ? "Yes" : "No"),
        h("dt", {}, "Sees the ground ahead"), h("dd", {}, brain.values.see_ground ? "Yes" : "No"))),
    estimate,
    startButton);
  paintEstimate();
  return null;
}

// Whether the centre of mass sits over the area the feet stand on
function balanceWord(preview) {
  const [x, y] = preview.centre_of_mass;
  let inside = false;
  const polygon = preview.support;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside ? "weight over its feet" : "weight outside its feet";
}

function defaultName(draft) {
  return `${draft.creature.name || store.animal(draft.creature.animal).name} in ${draft.world.name}`;
}

function changedCount(values, defaults) {
  return Object.keys(defaults).filter((key) => key !== "body" && values[key] !== defaults[key]).length;
}
