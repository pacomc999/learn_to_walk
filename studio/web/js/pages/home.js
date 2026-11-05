// Home: what the studio does, how to start, and the walkers made so far.

import { h, api, modal, toast } from "../ui.js";
import { icon, animalArt } from "../icons.js";
import { store, go, askForNotifications } from "../app.js";
import { walkerCard } from "./walkers.js";

// Quick start: a standard animal on flat ground, training in one click
async function quickStart() {
  let animal = "puppy";
  let length = "evening";
  const choose = (options, current, onPick) => {
    const box = h("div", { class: "segmented" });
    const paint = () => box.replaceChildren(...options.map(([value, label]) => h("button", { class: value === current ? "on" : "", onclick: () => { current = value; onPick(value); paint(); } }, label)));
    paint();
    return box;
  };
  const body = h("div", { class: "stack", style: { gap: "14px" } },
    h("div", {}, h("div", { class: "field-label", style: { marginBottom: "6px" } }, "Animal"),
      choose([["puppy", "Puppy"], ["duck", "Duck"]], animal, (value) => { animal = value; })),
    h("div", {}, h("div", { class: "field-label", style: { marginBottom: "6px" } }, "How long"),
      choose([["taste", "Quick taste"], ["evening", "An evening"], ["overnight", "Overnight"]], length, (value) => { length = value; })),
    h("div", { class: "field-help" }, "On the Flat Meadow with the standard body and brain. You can change everything later, or use Make a new walker to design your own."));
  const answer = await modal({ title: "Quick start", text: "Pick an animal and press Start. Training begins right away.", body,
    actions: [{ label: "Cancel", kind: "ghost", value: false }, { label: "Start training", kind: "primary", value: true }] });
  if (!answer) return;
  const kind = store.animal(animal);
  const preset = store.catalog.learning_presets.find((p) => p.id === length);
  try {
    askForNotifications();
    const setup = await api("trainings", { method: "POST", body: {
      name: `${kind.name} on the Flat Meadow`, creature: { animal, name: kind.name, values: JSON.parse(JSON.stringify(kind.defaults)) },
      world: store.catalog.world_templates[0], brain: store.catalog.brain_default,
      learning: { ...store.catalog.learning_default, parallel_worlds: Math.max(1, Math.min(8, store.catalog.system.cores || 4)), ...preset.values } } });
    toast("Training started", "Its first exam comes within a minute or two.", "good");
    go(`#/walker/${setup.id}`);
  } catch (error) {
    toast("Could not start", error.message, "bad");
  }
}

export async function render(main) {
  const page = h("div", { class: "page" });
  main.append(page);
  const trainings = await api("trainings");

  page.append(h("div", { class: "hero" },
    h("h1", {}, "Design a creature. Build its world. Watch it learn to walk."),
    h("p", {}, "Nobody tells it how to move its legs. A small brain tries things, earns points for walking at the speed it is asked for, and slowly finds a gait of its own."),
    h("div", { class: "btn-row" },
      h("button", { class: "btn primary big", onclick: () => go("#/new/creature"), html: `${icon.plus}<span>Make a new walker</span>` }),
      h("button", { class: "btn light big", onclick: quickStart, html: `${icon.play}<span>Quick start</span>` }),
      h("button", { class: "btn light big", onclick: () => go("#/guide"), html: `${icon.book}<span>How it works</span>` })),
    h("div", { class: "hero-art", html: `<div style="display:flex;gap:6px;align-items:flex-end;height:100%">${animalArt("puppy").replace('class="animal"', 'style="height:150px"')}${animalArt("duck").replace('class="animal"', 'style="height:170px"')}</div>` })));

  // A learn_to_walk folder nearby, and none of its ducks brought over yet: offer it
  const guess = await api("import/guess").catch(() => ({ folder: "" }));
  if (guess.folder && !trainings.some((t) => t.imported)) {
    page.append(h("div", { class: "callout", style: { marginTop: "16px", alignItems: "center" } },
      h("div", { style: { width: "36px", height: "28px", flex: "none" }, html: animalArt("duck").replace('class="animal"', 'style="width:36px;height:28px"') }),
      h("div", { style: { flex: 1 } }, h("b", {}, "Your learn_to_walk ducks are here. "),
        `Found ${guess.folder}. Bring the trained ducks over to test drive them in the new worlds, race them or teach them new tricks.`),
      h("button", { class: "btn small primary", onclick: async () => (await import("./importer.js")).importDialog() }, "Bring them over")));
  }

  const steps = [
    ["Design a creature", "A duck, a puppy or your own creation of sticks and motors. Every slider and part shows up in 3D."],
    ["Pick its world", "Flat grass, cobblestones, hills, stairs, ice or the Moon. Or make your own."],
    ["Shape its brain", "How many layers of thinking cells, what it can sense, and whether its left and right sides think alike."],
    ["Train and watch", "It practises in many worlds at once. Follow the charts and watch its walk improve exam by exam."],
  ];
  page.append(h("div", { class: "section-title" }, h("h2", {}, "Four steps")),
    h("div", { class: "steps-row" }, ...steps.map(([title, text], index) => h("div", { class: "card how" },
      h("div", { class: "n" }, String(index + 1)), h("div", { class: "t" }, title), h("div", { class: "d" }, text)))));

  page.append(h("div", { class: "section-title" }, h("h2", {}, "Your walkers"),
    trainings.length ? h("a", { href: "#/walkers", class: "btn ghost small", html: `<span>See all</span>${icon.chevronRight}` }) : null));
  if (!trainings.length) {
    page.append(h("div", { class: "empty" },
      h("div", { html: animalArt("duck").replace('class="animal"', 'class="animal" style="height:80px"') }),
      h("div", { class: "t" }, "No walkers yet"),
      h("div", { class: "d" }, "Your first one takes about two minutes to set up. The standard duck on flat ground is a great start."),
      h("button", { class: "btn primary", onclick: () => { store.resetDraft("duck"); go("#/new/creature"); }, html: `${icon.plus}<span>Make my first walker</span>` })));
  } else {
    page.append(h("div", { class: "grid grid-auto" }, ...trainings.slice(0, 6).map(walkerCard)));
  }
  return null;
}
