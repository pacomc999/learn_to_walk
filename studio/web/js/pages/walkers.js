// My walkers: every training session, newest first.

import { h, api, toast, fmtSteps, fmtNumber, fmtWhen, pillForState } from "../ui.js";
import { icon, animalArt, STYLE_COLOURS } from "../icons.js";
import { go, store } from "../app.js";
import { openWalker } from "../engine/sharing.js";

export function walkerCard(training) {
  const status = training.status;
  const style = STYLE_COLOURS[training.world.style] || STYLE_COLOURS.meadow;
  const share = status.total_steps ? Math.min(1, (status.steps || 0) / status.total_steps) : 0;
  return h("div", { class: "card walker-card", onclick: () => go(`#/walker/${training.id}`) },
    training.thumbnail
      ? h("div", { class: "top", style: { background: `center / cover no-repeat url(${training.thumbnail})` } },
        pillForState(status.state))
      : h("div", { class: "top", style: { background: `linear-gradient(180deg, ${style.sky[0]}, ${style.sky[1]} 70%, ${style.ground} 70%)` } },
        h("div", { html: animalArt(training.creature.animal, training.creature_values) }),
        pillForState(status.state)),
    h("div", { class: "bottom" },
      h("div", { class: "name" }, training.name),
      h("div", { class: "meta" }, h("span", {}, training.world.name), h("span", {}, "·"), h("span", {}, fmtWhen(training.created))),
      h("div", { class: "numbers" },
        h("div", {}, h("b", {}, status.best_quality !== null && status.best_quality !== undefined ? `${fmtNumber(status.best_quality)}%` : "-"), "best walk"),
        h("div", {}, h("b", {}, fmtSteps(status.steps || 0)), "steps"),
        h("div", { style: { textAlign: "right" } }, h("b", {}, `${fmtNumber(share * 100)}%`), "done")),
      h("div", { class: "progress" }, h("div", { style: { width: `${share * 100}%` } })), h("div", { class: "tape" })));
}

// Open a .walker file someone shared
export function openShared() {
  const input = h("input", { type: "file", accept: ".walker,.zip", style: { display: "none" } });
  input.addEventListener("change", async () => {
    const file = input.files[0];
    input.remove();
    if (!file) return;
    try {
      const id = await openWalker(file);
      toast("Walker opened", file.name, "good");
      go(`#/walker/${id}`);
    } catch (error) {
      toast("Could not open that file", error.message, "bad");
    }
  });
  document.body.append(input);
  input.click();
}

export async function render(main) {
  const page = h("div", { class: "page" });
  main.append(page);
  const trainings = await api("trainings");
  page.append(h("div", { class: "page-head" },
    h("div", {}, h("h1", {}, "My walkers"), h("div", { class: "sub" }, "Every creature you have trained, with how far along it is and how well it walks.")),
    h("div", { class: "btn-row" },
      h("button", { class: "btn", onclick: openShared, html: `${icon.open}<span>Open a shared walker</span>`, title: "Open a .walker file someone shared" }),
      h("button", { class: "btn", onclick: async () => (await import("./importer.js")).importDialog(), html: `${icon.folder}<span>From learn_to_walk</span>`, title: "Bring over the trained ducks of a learn_to_walk folder" }),
      h("button", { class: "btn primary", onclick: () => go("#/new/creature"), html: `${icon.plus}<span>New walker</span>` }))));
  if (!trainings.length) {
    page.append(h("div", { class: "empty" }, h("div", { html: animalArt("puppy") }), h("div", { class: "t" }, "Nothing trained yet"),
      h("div", { class: "d" }, "Make a walker and start its training. It shows up here."),
      h("div", { class: "btn-row", style: { justifyContent: "center" } },
        h("button", { class: "btn", onclick: async () => (await import("./importer.js")).importDialog() }, "Bring over learn_to_walk ducks"),
        h("button", { class: "btn primary", onclick: () => { store.resetDraft("duck"); go("#/new/creature"); } }, "Make a walker"))));
    return null;
  }
  const filters = ["All", "Training", "Finished", "Paused"];
  let filter = "All";
  const grid = h("div", { class: "grid grid-auto" });
  const tabs = h("div", { class: "segmented", style: { marginBottom: "16px" } });
  const paint = () => {
    tabs.replaceChildren(...filters.map((name) => h("button", { class: name === filter ? "on" : "", onclick: () => { filter = name; paint(); } }, name)));
    const keep = (t) => filter === "All" || (filter === "Training" && ["starting", "training", "examining"].includes(t.status.state))
      || (filter === "Finished" && t.status.state === "finished") || (filter === "Paused" && ["stopped", "interrupted", "error"].includes(t.status.state));
    grid.replaceChildren(...trainings.filter(keep).map(walkerCard));
  };
  paint();
  page.append(tabs, grid);
  return null;
}
