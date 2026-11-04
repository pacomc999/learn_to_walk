// Library: the creatures, worlds and brain designs you saved.

import { h, api, toast, confirmDialog, fmtWhen } from "../ui.js";
import { icon, animalArt, worldArt } from "../icons.js";
import { store, go } from "../app.js";

const KINDS = [
  { key: "creatures", title: "Creatures", empty: "Design a creature and press Save to library." },
  { key: "worlds", title: "Worlds", empty: "Change a world's settings and press Save world." },
  { key: "brains", title: "Brains", empty: "Save a brain design on the Brain step." },
];

export async function render(main, params) {
  const kind = KINDS.find((k) => k.key === params[0]) || KINDS[0];
  const page = h("div", { class: "page" });
  main.append(page);
  const all = Object.fromEntries(await Promise.all(KINDS.map(async (k) => [k.key, await api(k.key)])));
  page.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Library"),
    h("div", { class: "sub" }, "Everything you saved to use again. It lives in the library folder next to the studio."))));
  page.append(h("div", { class: "tabs" }, ...KINDS.map((k) => h("button", { class: k === kind ? "on" : "", onclick: () => go(`#/library/${k.key}`) },
    k.title, h("span", { class: "count" }, String(all[k.key].length))))));
  const items = all[kind.key];
  if (!items.length) {
    page.append(h("div", { class: "empty" }, h("div", { class: "t" }, `No saved ${kind.title.toLowerCase()} yet`), h("div", { class: "d" }, kind.empty),
      h("button", { class: "btn primary", onclick: () => go("#/new/creature") }, "Open the designer")));
    return null;
  }
  const grid = h("div", { class: "grid grid-auto" });
  for (const item of items) grid.append(card(kind.key, item));
  page.append(grid);
  return null;
}

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
