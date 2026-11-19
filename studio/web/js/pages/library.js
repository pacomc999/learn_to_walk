// Library: the trained walkers, creatures, worlds and brain designs you saved.

import { h, api, toast, confirmDialog, fmtWhen, fmtSteps, fmtNumber } from "../ui.js";
import { icon, animalArt, worldArt } from "../icons.js";
import { store, go } from "../app.js";
import { downloadFromLibrary, trainFromLibrary } from "../engine/sharing.js";

// Each tab: what it holds, how things get into it, and where to go to add one
const KINDS = [
  { key: "walkers", title: "Trained walkers", how: "Open a trained walker in My walkers and press Save to library. It keeps a copy with its trained brain.",
    goLabel: "Go to My walkers", goTo: "#/walkers" },
  { key: "creatures", title: "Creatures", how: "In New walker, design a creature and press Save to library on the Creature step.",
    goLabel: "Open the designer", goTo: "#/new/creature" },
  { key: "worlds", title: "Worlds", how: "In New walker, change a world's settings and press Save world on the World step.",
    goLabel: "Open the designer", goTo: "#/new/world" },
  { key: "brains", title: "Brain designs", how: "In New walker, save a brain's layers and settings on the Brain step. These are designs, not trained brains.",
    goLabel: "Open the designer", goTo: "#/new/brain" },
];

export async function render(main, params) {
  const kind = KINDS.find((k) => k.key === params[0]) || KINDS[0];
  const page = h("div", { class: "page" });
  main.append(page);
  const all = Object.fromEntries(await Promise.all(KINDS.map(async (k) => [k.key, await api(k.key)])));

  // Heading and tabs
  page.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Library"),
    h("div", { class: "sub" }, "Everything you saved to use again, kept in this browser on this computer."))));
  page.append(h("div", { class: "tabs" }, ...KINDS.map((k) => h("button", { class: k === kind ? "on" : "", onclick: () => go(`#/library/${k.key}`) },
    k.title, h("span", { class: "count" }, String(all[k.key].length))))));

  // How things get into this tab, always shown
  page.append(h("div", { class: "faint", style: { margin: "0 0 16px" } }, kind.how));

  const items = all[kind.key];
  if (!items.length) {
    page.append(h("div", { class: "empty" }, h("div", { class: "t" }, `No saved ${kind.title.toLowerCase()} yet`), h("div", { class: "d" }, kind.how),
      h("button", { class: "btn primary", onclick: () => go(kind.goTo) }, kind.goLabel)));
    return null;
  }
  const grid = h("div", { class: "grid grid-auto" });
  for (const item of items) grid.append(kind.key === "walkers" ? walkerCard(item) : card(kind.key, item));
  page.append(grid);
  return null;
}

// ---------- Saved trained walkers ----------

function walkerCard(item) {
  // Put a fresh copy in My walkers and open it there, ready to continue or train more
  const trainAgain = async (event) => {
    event.stopPropagation();
    try {
      const id = await trainFromLibrary(item.id);
      toast("In My walkers", `${item.name}: continue it, train it longer or teach it a new world.`, "good");
      go(`#/walker/${id}`);
    } catch (error) {
      toast("Could not open it", error.message, "bad");
    }
  };
  const download = async (event) => {
    event.stopPropagation();
    try {
      const filename = await downloadFromLibrary(item.id);
      toast("Downloaded", `${filename}: open it anywhere with Open a shared walker.`, "good");
    } catch (error) {
      toast("Could not download it", error.message, "bad");
    }
  };
  const remove = async (event) => {
    event.stopPropagation();
    if (!(await confirmDialog(`Delete "${item.name}" from the library?`, "Walkers in My walkers are not touched.", "Delete", "danger"))) return;
    await api(`walkers/${item.id}`, { method: "DELETE" });
    toast("Deleted", item.name);
    go("#/library/walkers");
  };

  const picture = typeof item.thumbnail === "string" && item.thumbnail.startsWith("data:image/jpeg") ? item.thumbnail : null;
  const animalName = store.animal(item.animal) ? store.animal(item.animal).name : "Walker";
  const quality = item.best_quality === null || item.best_quality === undefined ? "" : ` · best walk ${fmtNumber(item.best_quality)}%`;
  return h("div", { class: "card walker-card", onclick: trainAgain },
    picture
      ? h("div", { class: "top", style: { background: `center / cover no-repeat url(${picture})` } })
      : h("div", { class: "top", style: { background: "var(--surface-2)", overflow: "hidden" }, html: animalArt(item.animal, item.creature_values) }),
    h("div", { class: "bottom" },
      h("div", { class: "spread" }, h("div", { class: "name" }, item.name),
        h("button", { class: "btn ghost icon-btn small danger", title: "Delete", onclick: remove, html: icon.trash })),
      h("div", { class: "meta" }, h("span", {}, `${item.creature_name || animalName} · trained in ${item.world_name}`)),
      h("div", { class: "meta" }, h("span", {}, `${fmtSteps(item.steps)} steps${quality} · saved ${fmtWhen(item.updated)}`)),
      h("div", { class: "btn-row", style: { marginTop: "10px" } },
        h("button", { class: "btn small primary", onclick: trainAgain, html: `${icon.play}<span>Train it again</span>` }),
        h("button", { class: "btn small", title: "Download as a .walker file", onclick: download, html: `${icon.share}<span>Download</span>` }))));
}

// ---------- Creatures, worlds and brain designs ----------

function card(kind, item) {
  const catalog = store.catalog;
  let art = "", meta = "";
  if (kind === "creatures") {
    art = animalArt(item.animal, item.values);
    meta = store.animal(item.animal).name;
  } else if (kind === "worlds") {
    const defaults = Object.fromEntries(catalog.world_groups.flatMap((g) => g.settings.map((s) => [s.key, s.default])));
    art = worldArt({ ...defaults, ...item.values });
    meta = item.description || "Your world";
  } else {
    art = `<div style="width:48px;height:48px;color:var(--accent)">${icon.brain}</div>`;
    meta = `${item.design.layers.join(" x ")} cells`;
  }
  const use = () => {
    const draft = store.draft;
    if (kind === "creatures") {
      draft.creature = { animal: item.animal, name: item.name, values: { ...store.animal(item.animal).defaults, ...item.values }, id: item.id };
      store.saveDraft(); go("#/new/creature");
    } else if (kind === "worlds") {
      draft.world = JSON.parse(JSON.stringify(item));
      store.saveDraft(); go("#/new/world");
    } else {
      draft.brain = { values: { ...catalog.brain_default.values, ...item.design.values }, layers: item.design.layers, judge_layers: item.design.judge_layers };
      draft.brainPreset = null;
      store.saveDraft(); go("#/new/brain");
    }
  };
  const remove = async (event) => {
    event.stopPropagation();
    if (!(await confirmDialog(`Delete "${item.name}"?`, "Walkers already trained with it keep their own copy.", "Delete", "danger"))) return;
    await api(`${kind}/${item.id}`, { method: "DELETE" });
    toast("Deleted", item.name);
    go(`#/library/${kind}`);
  };
  const picture = typeof item.thumbnail === "string" && item.thumbnail.startsWith("data:image/jpeg") ? item.thumbnail : null;
  return h("div", { class: "card walker-card", onclick: use },
    picture
      ? h("div", { class: "top", style: { background: `center / cover no-repeat url(${picture})` } })
      : h("div", { class: "top", style: { background: "var(--surface-2)", overflow: "hidden" }, html: art }),
    h("div", { class: "bottom" },
      h("div", { class: "spread" }, h("div", { class: "name" }, item.name),
        h("button", { class: "btn ghost icon-btn small danger", title: "Delete", onclick: remove, html: icon.trash })),
      h("div", { class: "meta" }, h("span", {}, meta)),
      h("div", { class: "meta" }, h("span", {}, `Saved ${fmtWhen(item.updated)}`)),
      h("div", { class: "btn-row", style: { marginTop: "10px" } }, h("span", { class: "btn small", html: `${icon.edit}<span>Use in the designer</span>` }))));
}
