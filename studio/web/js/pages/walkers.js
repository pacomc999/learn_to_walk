// My walkers: every training session, newest first.

import { h, api, fmtSteps, fmtNumber, fmtWhen, pillForState } from "../ui.js";
import { icon, animalArt, STYLE_COLOURS } from "../icons.js";
import { go, store } from "../app.js";

export function walkerCard(training) {
  const status = training.status;
  const style = STYLE_COLOURS[training.world.style] || STYLE_COLOURS.meadow;
  const share = status.total_steps ? Math.min(1, (status.steps || 0) / status.total_steps) : 0;
  return h("div", { class: "card walker-card", onclick: () => go(`#/walker/${training.id}`) },
    h("div", { class: "top", style: { background: `linear-gradient(180deg, ${style.sky[0]}, ${style.sky[1]} 70%, ${style.ground} 70%)` } },
      h("div", { html: animalArt(training.creature.animal, training.creature_values) }),
      pillForState(status.state)),
    h("div", { class: "bottom" },
      h("div", { class: "name" }, training.name),
      h("div", { class: "meta" }, h("span", {}, training.world.name), h("span", {}, "·"), h("span", {}, fmtWhen(training.created))),
      h("div", { class: "numbers" },
        h("div", {}, h("b", {}, status.best_quality !== null && status.best_quality !== undefined ? `${fmtNumber(status.best_quality)}%` : "-"), "best walk"),
        h("div", {}, h("b", {}, fmtSteps(status.steps || 0)), "steps"),
        h("div", { style: { textAlign: "right" } }, h("b", {}, `${fmtNumber(share * 100)}%`), "done")),
      h("div", { class: "progress" }, h("div", { style: { width: `${share * 100}%` } }))));
}

export async function render(main) {
  const page = h("div", { class: "page" });
  main.append(page);
  const trainings = await api("trainings");
  page.append(h("div", { class: "page-head" },
    h("div", {}, h("h1", {}, "My walkers"), h("div", { class: "sub" }, "Every creature you have trained, with how far along it is and how well it walks.")),
    h("button", { class: "btn primary", onclick: () => go("#/new/creature"), html: `${icon.plus}<span>New walker</span>` })));
  if (!trainings.length) {
    page.append(h("div", { class: "empty" }, h("div", { html: animalArt("puppy") }), h("div", { class: "t" }, "Nothing trained yet"),
      h("div", { class: "d" }, "Make a walker and start its training. It shows up here."),
      h("button", { class: "btn primary", onclick: () => { store.resetDraft("duck"); go("#/new/creature"); } }, "Make a walker")));
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
