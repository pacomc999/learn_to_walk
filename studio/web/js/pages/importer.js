// Bring trained ducks over from a learn_to_walk folder on this computer.

import { h, api, toast, modal } from "../ui.js";
import { animalArt } from "../icons.js";
import { go } from "../app.js";

export async function importDialog() {
  const guess = await api("import/guess");
  const folderInput = h("input", { class: "input", value: guess.folder || "", placeholder: "For example C:\\Users\\you\\learn_to_walk" });
  const list = h("div", { class: "stack", style: { gap: "6px", marginTop: "12px", maxHeight: "320px", overflowY: "auto" } });
  const chosen = new Set();
  let ducks = [];
  const look = async () => {
    list.replaceChildren(h("div", { class: "faint" }, "Looking..."));
    try {
      ducks = await api("import/list", { method: "POST", body: { folder: folderInput.value.trim() } });
      chosen.clear();
      ducks.filter((d) => d.has_brain && d.kind !== "run").forEach((d) => chosen.add(d.folder));  // runs: pick them yourself
      paint();
    } catch (error) {
      list.replaceChildren(h("div", { class: "callout bad" }, error.message));
    }
  };
  const paint = () => list.replaceChildren(...ducks.flatMap((duck, index) => [
    duck.kind === "run" && (index === 0 || ducks[index - 1].kind !== "run")
      ? h("div", { class: "field-label", style: { margin: "8px 0 2px" } }, "Runs (the best brain of each training run on this computer)") : null,
    h("div", {
    class: `choice-item ${chosen.has(duck.folder) ? "on" : ""}`, style: { opacity: duck.has_brain ? 1 : 0.5 },
    onclick: () => { if (!duck.has_brain) return; if (chosen.has(duck.folder)) chosen.delete(duck.folder); else chosen.add(duck.folder); paint(); } },
    h("span", { class: "dot" }),
    h("div", { style: { width: "40px", height: "30px", flex: "none" }, html: animalArt("duck").replace('class="animal"', 'style="width:40px;height:30px"') }),
    h("div", { style: { minWidth: 0 } }, h("div", { style: { fontWeight: 600 } }, duck.name),
      h("div", { class: "faint", style: { fontSize: "12px" } }, duck.has_brain ? `${duck.world} · ${duck.body} body` : "No kept brain yet")))]));
  folderInput.addEventListener("change", look);
  const body = h("div", {},
    h("div", { class: "row" }, folderInput, h("button", { class: "btn", onclick: look }, "Look")),
    h("div", { class: "field-help" }, "The learn_to_walk project folder. It is only read, never changed."),
    list);
  if (folderInput.value) look();
  const answer = await modal({ title: "Bring over learn_to_walk ducks", text: "Each trained duck becomes a walker here, with its brain, body and world. You can then test drive it in the new worlds, race it, or teach it a new world.",
    body, actions: [{ label: "Cancel", kind: "ghost", value: false }, { label: "Bring them over", kind: "primary", value: true }] });
  if (!answer || !chosen.size) return;
  let last = null;
  for (const folder of chosen) {
    const duck = ducks.find((d) => d.folder === folder);
    toast(`Bringing over ${duck.name}`, "It takes a short exam in the studio first (about 10 seconds).");
    try {
      last = (await api("import/duck", { method: "POST", body: { folder } })).id;
    } catch (error) {
      toast(`Could not bring over ${duck.name}`, error.message, "bad");
    }
  }
  if (last) { toast("Done", "Your ducks are in My walkers.", "good"); go("#/walkers"); }
}
