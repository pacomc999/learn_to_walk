// Compare walkers: their learning curves on one chart and their recipes side by side.

import { h, api, fmtNumber, fmtSteps } from "../ui.js";
import { animalArt } from "../icons.js";
import { lineChart, legend } from "../charts.js";
import { store, go } from "../app.js";

const COLOURS = ["#2f7d4f", "#c28a2c", "#3f7fb5", "#b5533c", "#6b7280", "#2a8f86"];

export async function render(main) {
  const page = h("div", { class: "page" });
  main.append(page);
  const trainings = await api("trainings");
  page.append(h("div", { class: "page-head" }, h("div", {}, h("div", { class: "eyebrow" }, "Compare"), h("h1", {}, "Compare walkers"),
    h("div", { class: "sub" }, "Pick up to six walkers to see how fast each learned and what was different about them. Handy for experiments: did longer legs help? Was the bigger brain worth it?"))));
  if (trainings.length < 1) {
    page.append(h("div", { class: "empty" }, h("div", { class: "t" }, "Nothing to compare yet"),
      h("div", { class: "d" }, "Train a walker or two first."), h("button", { class: "btn primary", onclick: () => go("#/new/creature") }, "Make a walker")));
    return null;
  }
  const chosen = new Set(trainings.slice(0, 3).map((t) => t.id));
  const picker = h("div", { class: "picker", style: { gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))" } });
  const results = h("div", { class: "stack", style: { marginTop: "18px" } });
  const cache = {};

  const paintPicker = () => picker.replaceChildren(...trainings.map((t) => h("button", {
    class: `pick ${chosen.has(t.id) ? "on" : ""}`, onclick: () => {
      if (chosen.has(t.id)) chosen.delete(t.id); else if (chosen.size < 6) chosen.add(t.id);
      paintPicker(); paintResults();
    } },
    h("span", { class: "check", html: "✓" }),
    h("div", { class: "row" }, h("div", { style: { width: "42px", height: "32px", flex: "none" }, html: animalArt(t.creature.animal, t.creature_values).replace('class="animal"', 'style="width:42px;height:32px"') }),
      h("div", { style: { minWidth: 0 } }, h("div", { class: "pick-name", style: { fontSize: "13px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" } }, t.name),
        h("div", { class: "pick-desc" }, `${t.world.name} · ${fmtSteps(t.status.steps || 0)}`))))));

  const paintResults = async () => {
    const ids = [...chosen];
    await Promise.all(ids.filter((id) => !cache[id]).map(async (id) => { cache[id] = await api(`trainings/${id}`); }));
    const items = ids.map((id, index) => ({ id, colour: COLOURS[index % COLOURS.length], details: cache[id] }));
    if (!items.length) { results.replaceChildren(h("div", { class: "faint" }, "Pick at least one walker.")); return; }
    const quality = items.map((item) => ({ name: item.details.setup.name, colour: item.colour, dots: true,
      points: item.details.exams.map((exam) => [exam.steps, exam.quality]) }));
    const staying = items.map((item) => ({ name: item.details.setup.name, colour: item.colour,
      points: item.details.progress.map((row) => [row.steps, row.seconds]) }));
    results.replaceChildren(
      h("div", { class: "grid grid-2" },
        card("How well it walks", "Each exam: the share of the asked distance covered.", quality, { yMin: 0, yMax: 100, yFormat: (v) => `${fmtNumber(v)}%` }),
        card("Staying up", "Seconds per practice attempt.", staying, { yMin: 0 })),
      recipeTable(items));
  };

  page.append(picker, results);
  paintPicker();
  paintResults();
  return null;
}

function card(title, description, series, options) {
  return h("div", { class: "card" },
    h("div", { class: "card-head" }, h("div", {}, h("h3", {}, title), h("div", { class: "desc" }, description))),
    h("div", { class: "card-body" }, lineChart(series, { height: 220, ...options }), h("div", { style: { marginTop: "8px" } }, legend(series))));
}

// The differences that matter, one column per walker; rows where all agree are left out
function recipeTable(items) {
  const catalog = store.catalog;
  const rows = [
    ["Best walk", (d) => d.status.best_quality !== null && d.status.best_quality !== undefined ? `${fmtNumber(d.status.best_quality)}%` : "-"],
    ["Practice steps", (d) => fmtSteps(d.status.steps || 0)],
    ["Energy per metre", (d) => { const last = d.exams[d.exams.length - 1]; return last && last.joules_per_metre ? `${fmtNumber(last.joules_per_metre, 1)} J` : "-"; }],
    ["Animal", (d) => store.animal(d.setup.creature.animal).name],
    ["World", (d) => d.setup.world.name],
    ["Brain layers", (d) => d.setup.brain.layers.join(" x ")],
    ["Symmetric", (d) => (d.setup.brain.values.symmetric ? "yes" : "no")],
    ["Sees the ground", (d) => (d.setup.brain.values.see_ground ? "yes" : "no")],
  ];
  // Creature and world settings that differ between the chosen walkers
  const animalKeys = new Set(items.map((item) => item.details.setup.creature.animal));
  if (animalKeys.size === 1) {
    const animal = store.animal([...animalKeys][0]);
    for (const group of animal.groups) for (const setting of group.settings) {
      const values = items.map((item) => item.details.setup.creature.values[setting.key] ?? animal.defaults[setting.key]);
      if (new Set(values.map(String)).size > 1) rows.push([setting.label, (d) => show(setting, d.setup.creature.values[setting.key] ?? animal.defaults[setting.key])]);
    }
  }
  for (const group of catalog.world_groups) for (const setting of group.settings) {
    const values = items.map((item) => item.details.setup.world.values[setting.key]);
    if (new Set(values.map(String)).size > 1) rows.push([`World: ${setting.label}`, (d) => show(setting, d.setup.world.values[setting.key])]);
  }
  for (const group of catalog.learning_groups) for (const setting of group.settings) {
    const values = items.map((item) => item.details.setup.learning[setting.key]);
    if (new Set(values.map(String)).size > 1) rows.push([`Training: ${setting.label}`, (d) => show(setting, d.setup.learning[setting.key])]);
  }
  return h("div", { class: "card", style: { overflowX: "auto" } },
    h("table", { class: "compare" },
      h("thead", {}, h("tr", {}, h("th", {}, ""), ...items.map((item) => h("th", {}, h("span", { class: "swatch", style: { background: item.colour } }),
        h("a", { href: `#/walker/${item.id}` }, item.details.setup.name))))),
      h("tbody", {}, ...rows.map(([label, value]) => h("tr", {}, h("td", { class: "faint" }, label), ...items.map((item) => h("td", {}, value(item.details))))))));
}

function show(setting, value) {
  if (value === undefined || value === null) return "-";
  if (setting.kind === "number") return `${fmtNumber(value, setting.step < 1 ? 2 : 0)}${setting.unit === "%" ? "%" : setting.unit ? ` ${setting.unit}` : ""}`;
  if (setting.kind === "bool") return value ? "on" : "off";
  if (setting.kind === "choice") return (setting.choices.find((choice) => choice.value === value) || {}).label || String(value);
  return String(value);
}
