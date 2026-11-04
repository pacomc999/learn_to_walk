// Race: up to six trained walkers side by side in one world. Who goes furthest?

import { h, api, toast, fmtNumber, settingControl } from "../ui.js";
import { icon, animalArt, worldArt } from "../icons.js";
import { Viewer } from "../viewer.js";
import { store, go } from "../app.js";

const MEDALS = ["#f5b301", "#a8b3c2", "#cd7f32"];

export async function render(main) {
  const page = h("div", { class: "page page-wide" });
  main.append(page);
  const trainings = (await api("trainings")).filter((t) => t.has_brain);
  page.append(h("div", { class: "page-head" }, h("div", {}, h("div", { class: "eyebrow" }, "Race"), h("h1", {}, "Race your walkers"),
    h("div", { class: "sub" }, "Put up to six trained walkers side by side in any world and see who gets furthest. Each one walks on the same ground at the same asked speed."))));
  if (!trainings.length) {
    page.append(h("div", { class: "empty" }, h("div", { html: animalArt("duck") }), h("div", { class: "t" }, "No trained walkers yet"),
      h("div", { class: "d" }, "A walker can race once it has taken its first exam. Make one and start its training."),
      h("button", { class: "btn primary", onclick: () => go("#/new/creature") }, "Make a walker")));
    return null;
  }

  const catalog = store.catalog;
  const worldDefaults = Object.fromEntries(catalog.world_groups.flatMap((group) => group.settings.map((s) => [s.key, s.default])));
  const worlds = [...catalog.world_templates, ...(await api("worlds"))];
  const chosen = new Set(trainings.slice(0, 3).map((t) => t.id));
  let world = worlds[0];
  const race = { speed: 0.45, height: null, seconds: 20, seed: 1000 };

  const viewerBox = h("div", { class: "viewer-tall" });
  const panel = h("div", { class: "panel" });
  const board = h("div");
  page.append(h("div", { class: "workbench" }, panel, h("div", { class: "stage" }, viewerBox, board)));
  const viewer = new Viewer(viewerBox, { follow: true });
  viewer.showMessage("Pick the racers and a world, then press Start the race.");

  const racers = h("div", { class: "stack", style: { gap: "6px" } });
  const paintRacers = () => racers.replaceChildren(...trainings.map((t) => h("div", {
    class: `choice-item ${chosen.has(t.id) ? "on" : ""}`, onclick: () => {
      if (chosen.has(t.id)) chosen.delete(t.id);
      else if (chosen.size >= 6) toast("Six at most", "A race has room for six walkers.");
      else chosen.add(t.id);
      paintRacers();
    } },
    h("span", { class: "dot" }),
    h("div", { style: { width: "40px", height: "30px", flex: "none" }, html: animalArt(t.creature.animal, t.creature_values).replace('class="animal"', 'style="width:40px;height:30px"') }),
    h("div", { style: { minWidth: 0 } }, h("div", { style: { fontWeight: 600 } }, t.name),
      h("div", { class: "faint", style: { fontSize: "12px" } }, `trained in ${t.world.name} · best walk ${fmtNumber(t.status.best_quality || 0)}%`)))));

  const gallery = h("div", { class: "picker", style: { gridTemplateColumns: "repeat(auto-fill, minmax(112px, 1fr))" } });
  const paintGallery = () => gallery.replaceChildren(...worlds.map((w) => h("button", { class: `pick ${w === world ? "on" : ""}`, onclick: () => {
    world = w; race.height = null; paintGallery(); paintControls();
  } }, h("span", { class: "check", html: icon.check }), h("div", { class: "art", style: { height: "50px" }, html: worldArt({ ...worldDefaults, ...w.values }) }),
    h("div", { class: "pick-name", style: { fontSize: "12.5px" } }, w.name))));

  const controls = h("div");
  const paintControls = () => {
    const v = { ...worldDefaults, ...world.values };
    const tallest = v.terrain !== "flat" ? v.terrain_height : 0;
    if (race.height === null) race.height = tallest;
    controls.replaceChildren(
      settingControl({ key: "speed", label: "Asked speed", kind: "number", minimum: 0.05, maximum: 3, step: 0.05, unit: "m/s" }, race.speed, (value) => { race.speed = value; }, { defaultValue: 0.45 }),
      tallest > 0 ? settingControl({ key: "height", label: "Ground height", kind: "number", minimum: 0, maximum: Math.max(tallest * 2, 1), step: 0.5, unit: "cm" }, race.height, (value) => { race.height = value; }, { defaultValue: tallest }) : null,
      settingControl({ key: "seconds", label: "Length", kind: "number", minimum: 5, maximum: 60, step: 5, unit: "s" }, race.seconds, (value) => { race.seconds = value; }, { defaultValue: 20 }));
  };

  const start = h("button", { class: "btn primary big", style: { width: "100%" }, html: `${icon.flag}<span>Start the race</span>` });
  start.addEventListener("click", async () => {
    if (!chosen.size) { toast("No racers", "Pick at least one walker."); return; }
    start.disabled = true;
    start.innerHTML = `<span class="spinner"></span><span>Running the race...</span>`;
    viewer.setLoading(true);
    try {
      const replay = await api("race", { method: "POST", body: { ids: [...chosen], world: { ...world, values: { ...worldDefaults, ...world.values } },
        speed: race.speed, height: (race.height || 0) / 100, seconds: race.seconds, seed: race.seed } });
      viewer.showMessage("");
      viewer.load(replay);
      viewer.setChips([["World", world.name], ["Asked speed", `${fmtNumber(race.speed, 2)} m/s`], ["Racers", String(replay.results.length)]]);
      paintBoard(replay.results);
      race.seed += 1;
    } catch (error) {
      toast("The race could not run", error.message, "bad");
    } finally {
      start.disabled = false;
      start.innerHTML = `${icon.flag}<span>Start the race</span>`;
      viewer.setLoading(false);
    }
  });

  const paintBoard = (results) => board.replaceChildren(h("div", { class: "card card-pad", style: { marginTop: "14px" } },
    h("h3", { style: { marginBottom: "10px" } }, "Results"),
    h("div", { class: "stack", style: { gap: "8px" } }, ...results.map((result, index) => h("div", { class: "spread race-row" },
      h("div", { class: "row" },
        h("span", { class: "place", style: { background: MEDALS[index] || "var(--surface-3)", color: index < 3 ? "#fff" : "var(--text-2)" } }, String(index + 1)),
        h("div", {}, h("div", { style: { fontWeight: 620 } }, result.name), h("div", { class: "faint", style: { fontSize: "12px" } }, result.creature))),
      h("div", { style: { textAlign: "right" } },
        h("div", { style: { fontWeight: 650 } }, `${fmtNumber(result.distance, 2)} m`),
        h("div", { class: "faint", style: { fontSize: "12px" } }, result.fell ? `fell after ${fmtNumber(result.seconds, 1)} s` : `${fmtNumber(result.average_speed, 2)} m/s, stayed up`)))))));

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "10px", marginBottom: "12px" } }, h("h3", {}, "Racers"), racers),
    h("div", { class: "card card-pad stack", style: { gap: "10px", marginBottom: "12px" } }, h("h3", {}, "World"), gallery),
    h("div", { class: "card card-pad", style: { marginBottom: "12px" } }, h("h3", { style: { marginBottom: "6px" } }, "Rules"), controls),
    start);
  paintRacers();
  paintGallery();
  paintControls();
  return () => viewer.dispose();
}
