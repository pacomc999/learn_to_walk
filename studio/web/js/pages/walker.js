// One walker: follow its training live, watch how it grew up, and take it for a test drive in any world.

import { h, api, toast, fmtSteps, fmtNumber, fmtDuration, fmtWhen, pillForState, confirmDialog, askName, statCard, settingControl } from "../ui.js";
import { icon, animalArt, worldArt } from "../icons.js";
import { Viewer } from "../viewer.js";
import { lineChart, legend } from "../charts.js";
import { store, go } from "../app.js";

const ACTIVE = ["starting", "training", "examining"];

// The parts of the score, as the world screen names them
const PART_LABELS = [
  ["speed", "Moving forward", "#6366f1"], ["alive", "Staying on its feet", "#10b981"], ["energy", "Energy", "#f59e0b"],
  ["straight", "Drifting", "#0ea5e9"], ["fall", "Falling", "#ef4444"], ["end_stops", "Joint limits", "#a855f7"],
  ["level", "Tilting", "#14b8a6"], ["crouch", "Crouching", "#84cc16"], ["smooth", "Jerky moves", "#ec4899"],
  ["feet", "Lifting feet", "#22c55e"], ["heading", "Turning away", "#64748b"], ["ride", "Bouncing", "#f97316"],
];

export async function render(main, params) {
  const id = params[0];
  const tab = params[1] || "training";
  const page = h("div", { class: "page" });
  main.append(page);
  let details = await api(`trainings/${id}`);
  const setup = details.setup;
  const animal = store.animal(setup.creature.animal);

  // ---------- Header ----------
  const title = h("h1", {}, setup.name);
  const statePill = h("span");
  const actions = h("div", { class: "btn-row" });
  page.append(h("div", { class: "page-head" },
    h("div", { class: "row", style: { gap: "16px", alignItems: "center" } },
      h("div", { style: { width: "64px", height: "54px", borderRadius: "14px", background: "var(--surface)", border: "1px solid var(--border)", display: "grid", placeItems: "center" },
        html: animalArt(animal.key, setup.creature.values) }),
      h("div", {},
        h("div", { class: "row", style: { gap: "10px" } }, title, statePill),
        h("div", { class: "sub" }, `${setup.creature.name || animal.name} · ${setup.world.name} · started ${fmtWhen(setup.created)}`))),
    actions));

  const tabs = h("div", { class: "tabs" },
    ...[["training", "Training"], ["drive", "Test drive"], ["recipe", "Recipe"]].map(([key, label]) =>
      h("button", { class: tab === key ? "on" : "", onclick: () => go(`#/walker/${id}/${key}`) }, label)));
  page.append(tabs);
  const body = h("div");
  page.append(body);

  const paintHeader = () => {
    const state = details.status.state;
    statePill.replaceChildren(pillForState(state));
    const running = ACTIVE.includes(state);
    actions.replaceChildren(...[
      running
        ? h("button", { class: "btn", onclick: pause, html: `${icon.pause}<span>Pause</span>` })
        : details.status.state !== "finished" || details.status.steps < details.status.total_steps
          ? h("button", { class: "btn primary", onclick: carryOn, html: `${icon.play}<span>Continue</span>` })
          : null,
      details.has_brain ? h("button", { class: "btn", onclick: trainMore, html: `${icon.sparkle}<span>Teach a new world</span>` }) : null,
      h("button", { class: "btn ghost icon-btn", title: "Rename", onclick: rename, html: icon.edit }),
      h("button", { class: "btn ghost icon-btn danger", title: "Delete", onclick: remove, html: icon.trash })].filter(Boolean));
  };

  const pause = async () => {
    await api(`trainings/${id}/stop`, { method: "POST" });
    toast("Pausing", "It finishes the current lesson, saves, and stops. You can continue later.");
  };
  const carryOn = async () => {
    await api(`trainings/${id}/continue`, { method: "POST" });
    toast("Carrying on", "Training continues from where it stopped.", "good");
    refresh();
  };
  const rename = async () => {
    const name = await askName("Rename this walker", "", setup.name);
    if (!name) return;
    await api(`trainings/${id}/rename`, { method: "POST", body: { name } });
    setup.name = name;
    title.textContent = name;
  };
  const remove = async () => {
    if (ACTIVE.includes(details.status.state)) { toast("Pause it first", "A walker that is training cannot be deleted.", "bad"); return; }
    if (!(await confirmDialog("Delete this walker?", "Its brain, replays and charts are deleted for good.", "Delete", "danger"))) return;
    await api(`trainings/${id}`, { method: "DELETE" });
    toast("Deleted", setup.name);
    go("#/walkers");
  };
  const trainMore = () => {
    store.draft = {
      name: `${setup.name} (more)`, creature: { ...setup.creature }, world: JSON.parse(JSON.stringify(setup.world)),
      brain: JSON.parse(JSON.stringify(setup.brain)), brainPreset: null, learning: { ...setup.learning }, learningPreset: null, start_from: id,
    };
    store.saveDraft();
    toast("Pick a new world", "The new training starts from this walker's best brain. Keep the creature and brain as they are.");
    go("#/new/world");
  };

  const refresh = async () => {
    try {
      details = await api(`trainings/${id}`);
      paintHeader();
      if (view && view.update) view.update(details);
    } catch { /* try again on the next tick */ }
  };

  paintHeader();
  let view = null;
  if (tab === "drive") view = await driveTab(body, id, details, setup);
  else if (tab === "recipe") view = recipeTab(body, details, setup);
  else view = trainingTab(body, id, details, setup);
  const timer = setInterval(refresh, 3000);
  return () => { clearInterval(timer); if (view && view.dispose) view.dispose(); };
}

// ---------- The training tab ----------

function trainingTab(body, id, details, setup) {
  const stats = h("div", { class: "stats" });
  const viewerBox = h("div", { class: "viewer-mid" });
  const moments = h("div", { class: "stack", style: { gap: "8px", maxHeight: "480px", overflowY: "auto" } });
  const momentsCard = h("div", { class: "card", style: { display: "flex", flexDirection: "column" } },
    h("div", { class: "card-head" }, h("div", {}, h("h3", {}, "Growing up"), h("div", { class: "desc" }, "Its walk at every exam"))),
    h("div", { class: "card-body" }, moments));
  const story = h("div");
  const charts = h("div", { class: "grid grid-2" });
  body.append(stats, h("div", { style: { height: "16px" } }), story,
    h("div", { class: "grid", style: { gridTemplateColumns: "minmax(0, 1fr) 260px", marginTop: "16px" } }, viewerBox, momentsCard),
    h("div", { class: "section-title" }, h("h2", {}, "Progress")), charts,
    h("div", { class: "section-title" }, h("h2", {}, "Training log")), h("div", { class: "log", id: "log" }));
  const viewer = new Viewer(viewerBox, { pace: true });
  viewer.showMessage("The first walk appears after the first exam, within a minute or two of starting.");
  let shown = null;
  let current = details;
  let lastBest = null;
  let lastReplayCount = -1;

  const showReplay = async (name) => {
    shown = name;
    paintMoments(current);
    viewer.setLoading(true);
    try {
      const response = await fetch(`/api/trainings/${id}/replay/${name}`);
      if (!response.ok) throw new Error("This replay is not there.");
      const replay = await response.json();
      viewer.load(replay);
      const s = replay.stats;
      viewer.setChips([
        [name === "best" ? "Best walk so far" : `At ${fmtSteps(replay.steps)} steps`, s.fell ? `fell after ${fmtNumber(s.seconds, 1)} s` : `${fmtNumber(s.seconds, 0)} s without falling`],
        ["Distance", `${fmtNumber(s.distance, 2)} m`], ["Speed", `${fmtNumber(s.average_speed, 2)} of ${fmtNumber(s.target_speed, 2)} m/s asked`]]);
    } catch (error) {
      viewer.showMessage(error.message);
    } finally {
      viewer.setLoading(false);
    }
  };

  const paintMoments = (data) => {
    const exams = data.exams;
    const byStep = Object.fromEntries(exams.map((exam) => [exam.steps, exam]));
    const items = [];
    if (data.has_best) items.push(moment("best", "Best so far", data.status.best_quality, `exam at ${fmtSteps(data.status.best_steps)}`));
    for (const name of [...data.replays].reverse()) {
      const steps = Number(name.slice(5));
      const exam = byStep[steps];
      items.push(moment(name, `${fmtSteps(steps)} steps`, exam ? exam.quality : null, exam ? `${exam.falls} of ${exam.attempts} fell` : ""));
    }
    moments.replaceChildren(...(items.length ? items : [h("div", { class: "faint", style: { fontSize: "13px" } }, "No exams yet.")]));
  };
  const moment = (name, when, quality, what) => h("div", { class: `moment ${shown === name ? "on" : ""}`, onclick: () => showReplay(name) },
    h("div", { class: "spread" }, h("span", { class: "when" }, when), quality !== null && quality !== undefined ? h("span", { class: "faint", style: { fontSize: "12px" } }, `${fmtNumber(quality)}%`) : null),
    h("div", { class: "what" }, what),
    h("div", { class: "bar" }, h("div", { style: { width: `${Math.max(0, Math.min(100, quality || 0))}%` } })));

  const update = (data) => {
    current = data;
    const status = data.status;
    const progress = data.progress;
    const exams = data.exams;
    const share = status.total_steps ? Math.min(1, (status.steps || 0) / status.total_steps) : 0;
    const speed = status.steps_per_second || 0;
    const remaining = speed ? (status.total_steps - status.steps) / speed : NaN;
    const last = exams[exams.length - 1];
    const progressCard = statCard("Progress", `${fmtSteps(status.steps || 0)}<small>/ ${fmtSteps(status.total_steps)}</small>`,
      ACTIVE.includes(status.state) && speed ? `about ${fmtDuration(remaining)} to go` : status.message || "", icon.clock);
    progressCard.append(h("div", { class: "progress" }, h("div", { style: { width: `${share * 100}%` } })));
    stats.replaceChildren(...[
      progressCard,
      statCard("Best walk", status.best_quality !== null && status.best_quality !== undefined ? `${fmtNumber(status.best_quality)}<small>%</small>` : "-",
        "of the asked distance, in its best exam", icon.trophy),
      statCard("Last exam", last ? `${last.attempts - last.falls}<small>/ ${last.attempts} stayed up</small>` : "-",
        last ? `score ${fmtNumber(last.score)} points` : "the first exam comes right at the start", icon.target),
      statCard("Practice speed", speed ? `${fmtNumber(speed)}<small>steps/s</small>` : "-", `${setup.learning.parallel_worlds} worlds at once`, icon.speed),
      setup.world.values.gradually_harder ? statCard("Ground height", `${fmtNumber((status.difficulty ?? 1) * 100)}<small>%</small>`, "of the tallest, rising as it learns", icon.stairs) : null].filter(Boolean));
    story.replaceChildren(storyCallout(status, exams, progress));

    // Charts
    const examPoints = (key) => exams.map((exam) => [exam.steps, exam[key]]);
    const qualitySeries = [{ name: "Exam: share of the asked distance", colour: "#10b981", points: examPoints("quality"), dots: true, fill: true }];
    const scoreSeries = [
      { name: "Practice score", colour: "#6366f1", points: progress.map((row) => [row.steps, row.score]) },
      { name: "Exam score", colour: "#f59e0b", points: examPoints("score"), dots: true },
    ];
    const lengthSeries = [{ name: "Seconds per attempt", colour: "#0ea5e9", points: progress.map((row) => [row.steps, row.seconds]), fill: true }];
    const fallSeries = [{ name: "Attempts that fell", colour: "#ef4444", points: progress.map((row) => [row.steps, row.fall_share === null ? null : row.fall_share * 100]), fill: true }];
    const partSeries = PART_LABELS.map(([key, label, colour]) => ({ name: label, colour,
      points: progress.map((row) => [row.steps, row[`pts_${key}`]]) }))
      .filter((s) => s.points.some((p) => p[1] !== null && p[1] !== undefined && Math.abs(p[1]) > 0.05));
    charts.replaceChildren(
      chartCard("How well it walks", "In each exam: how much of the asked distance it covered without help.", qualitySeries, { yMin: 0, yMax: 100, yFormat: (v) => `${fmtNumber(v)}%` }),
      chartCard("Points", "What it is learning to collect. Practice includes its random tries; the exam does not.", scoreSeries, {}),
      chartCard("Staying up", "How long a practice attempt lasts before it falls or time runs out.", lengthSeries, { yMin: 0 }),
      chartCard("Falls", "The share of practice attempts that end in a fall.", fallSeries, { yMin: 0, yMax: 100, yFormat: (v) => `${fmtNumber(v)}%` }),
      partSeries.length ? h("div", { style: { gridColumn: "1 / -1" } }, chartCard("Where its points come from",
        "Points per practice attempt from each part of the score. Above zero it earns them, below zero it pays them.", partSeries, {})) : null);
    const log = document.getElementById("log");
    if (log) { log.textContent = data.log || "Nothing yet."; }

    // Replays: show the best walk when it changes, and list the moments
    const replayCount = data.replays.length;
    if (data.has_best && (status.best_steps !== lastBest) && (shown === null || shown === "best")) {
      lastBest = status.best_steps;
      viewer.showMessage("");
      showReplay("best");
    } else if (replayCount !== lastReplayCount) {
      paintMoments(data);
    }
    lastReplayCount = replayCount;
  };
  update(details);
  return { update, dispose: () => viewer.dispose() };
}

function chartCard(title, description, series, options) {
  return h("div", { class: "card" },
    h("div", { class: "card-head" }, h("div", {}, h("h3", {}, title), h("div", { class: "desc" }, description))),
    h("div", { class: "card-body" }, lineChart(series, { height: 200, ...options }), h("div", { style: { marginTop: "8px" } }, legend(series))));
}

// A few plain sentences about what is going on
function storyCallout(status, exams, progress) {
  const state = status.state;
  const last = exams[exams.length - 1];
  const recent = progress.slice(-5);
  const seconds = recent.length ? recent.reduce((sum, row) => sum + (row.seconds || 0), 0) / recent.length : 0;
  let kind = "", text = "";
  if (state === "error") { kind = "bad"; text = `<b>Training stopped with a problem.</b> ${status.message || ""} The log below has the details.`; }
  else if (state === "starting") text = "<b>Getting ready.</b> Building the practice worlds and a fresh brain. This takes a few seconds.";
  else if (state === "interrupted" || state === "stopped") { kind = "warn"; text = `<b>Paused.</b> ${(status.message || "").replace(/^Paused\.\s*/, "")}`; }
  else if (state === "finished") { kind = "good"; text = `<b>Done.</b> ${status.message || ""} Its best walk covered ${fmtNumber(status.best_quality)}% of the asked distance. Try the Test drive tab to see it in other worlds.`; }
  else if (!last || status.steps < 300000) text = "<b>Flailing about.</b> At first the brain knows nothing, so it tries random moves and falls a lot. Falling quickly is normal for a while.";
  else if (seconds < 3) text = "<b>Learning to stay up.</b> Most attempts still end in a fall within a few seconds. The first real steps usually come after one to two million steps.";
  else if ((status.best_quality || 0) < 40) text = "<b>First steps.</b> It stays up longer now and is finding a way to move forward. Watch the exam replays get better.";
  else if ((status.best_quality || 0) < 80) { kind = "good"; text = "<b>Walking.</b> It covers a good part of the asked distance. Now it polishes its gait and learns to follow the asked speed."; }
  else { kind = "good"; text = "<b>Walking well.</b> It covers nearly all the asked distance. More training refines speed control and saves energy."; }
  return h("div", { class: `callout ${kind}`, html: `${icon.info}<div>${text}</div>` });
}

// ---------- The test drive tab ----------

async function driveTab(body, id, details, setup) {
  if (!details.has_brain) {
    body.append(h("div", { class: "empty" }, h("div", { class: "t" }, "No brain to drive yet"),
      h("div", { class: "d" }, "The test drive needs the walker's best brain, saved after its first exam. Come back in a minute.")));
    return null;
  }
  const catalog = store.catalog;
  const worldDefaults = Object.fromEntries(catalog.world_groups.flatMap((group) => group.settings.map((s) => [s.key, s.default])));
  const mine = await api("worlds");
  const worlds = [{ ...setup.world, name: `${setup.world.name} (its own)`, own: true }, ...catalog.world_templates.filter((w) => w.id !== setup.world.id), ...mine];
  let chosen = worlds[0];
  const values = () => ({ ...worldDefaults, ...chosen.values });
  const drive = { speed: null, height: null, seed: 1000, seconds: 20 };

  const viewerBox = h("div", { class: "viewer-tall" });
  const panel = h("div", { class: "panel" });
  body.append(h("div", { class: "workbench" }, panel, h("div", { class: "stage" }, viewerBox)));
  const viewer = new Viewer(viewerBox, { pace: true });

  const gallery = h("div", { class: "picker", style: { gridTemplateColumns: "repeat(auto-fill, minmax(118px, 1fr))" } });
  const paintGallery = () => gallery.replaceChildren(...worlds.map((world) => h("button", { class: `pick ${world === chosen ? "on" : ""}`, onclick: () => {
    chosen = world;
    drive.speed = null; drive.height = null;
    paintGallery(); paintControls(); run();
  } }, h("span", { class: "check", html: icon.check }), h("div", { class: "art", style: { height: "54px" }, html: worldArt({ ...worldDefaults, ...world.values }) }),
    h("div", { class: "pick-name", style: { fontSize: "12.5px" } }, world.name))));

  const controls = h("div");
  const paintControls = () => {
    const v = values();
    const fastest = Math.max(v.fastest_speed * 1.5, 0.5);
    if (drive.speed === null) drive.speed = Math.round((v.slowest_speed + v.fastest_speed) / 2 * 100) / 100;
    const tallest = v.terrain !== "flat" ? v.terrain_height : 0;
    if (drive.height === null) drive.height = tallest;
    controls.replaceChildren(
      settingControl({ key: "speed", label: "Asked speed", kind: "number", minimum: 0.05, maximum: Math.round(fastest * 100) / 100, step: 0.05, unit: "m/s",
        help: `It trained for ${setup.world.values.slowest_speed} to ${setup.world.values.fastest_speed} m/s. Outside that it is improvising.` },
        drive.speed, (value) => { drive.speed = value; }, { defaultValue: drive.speed }),
      tallest > 0 ? settingControl({ key: "height", label: "Ground height", kind: "number", minimum: 0, maximum: Math.max(tallest * 2, 1), step: 0.5, unit: "cm",
        help: "How tall the bumps, stones or steps are in this drive." }, drive.height, (value) => { drive.height = value; }, { defaultValue: drive.height }) : null,
      settingControl({ key: "seconds", label: "Length", kind: "number", minimum: 5, maximum: 60, step: 5, unit: "s" }, drive.seconds, (value) => { drive.seconds = value; }, { defaultValue: 20 }));
  };

  const result = h("div");
  const runButton = h("button", { class: "btn primary big", style: { width: "100%" }, html: `${icon.play}<span>Drive</span>` });
  const run = async () => {
    runButton.disabled = true;
    viewer.setLoading(true);
    try {
      const v = values();
      const replay = await api(`trainings/${id}/watch`, { method: "POST", body: {
        world: { ...chosen, values: { ...v, attempt_seconds: Math.max(v.attempt_seconds, drive.seconds) } }, speed: drive.speed,
        height: (drive.height || 0) / 100, seed: drive.seed, seconds: drive.seconds } });
      viewer.load(replay);
      const s = replay.stats;
      viewer.setChips([["World", chosen.name], ["Result", s.fell ? `fell after ${fmtNumber(s.seconds, 1)} s` : `walked all ${fmtNumber(s.seconds, 0)} s`],
        ["Distance", `${fmtNumber(s.distance, 2)} m`], ["Speed", `${fmtNumber(s.average_speed, 2)} of ${fmtNumber(s.target_speed, 2)} m/s`]]);
      result.replaceChildren(h("div", { class: `callout ${s.fell ? "warn" : "good"}`, html: `${icon.info}<div>${s.fell
        ? `It fell after ${fmtNumber(s.seconds, 1)} seconds, ${fmtNumber(s.distance, 2)} m from the start.`
        : `It stayed up for all ${fmtNumber(s.seconds, 0)} seconds and went ${fmtNumber(s.distance, 2)} m, at ${fmtNumber(s.average_speed, 2)} m/s (asked: ${fmtNumber(s.target_speed, 2)}).`}${s.rock_hits ? ` Rocks hit it ${s.rock_hits} times.` : ""} The blue ring moves at the asked speed.</div>` }));
    } catch (error) {
      toast("The drive failed", error.message, "bad");
    } finally {
      runButton.disabled = false;
      viewer.setLoading(false);
    }
  };
  runButton.addEventListener("click", run);

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "12px", marginBottom: "12px" } },
      h("div", {}, h("h3", {}, "Where"), h("div", { class: "faint", style: { fontSize: "12.5px" } }, "Any world, even ones it never practised in.")), gallery),
    h("div", { class: "card card-pad", style: { marginBottom: "12px" } }, h("h3", { style: { marginBottom: "6px" } }, "How"), controls,
      h("div", { class: "btn-row", style: { marginTop: "10px" } }, h("button", { class: "btn small", onclick: () => { drive.seed += 1; run(); }, html: `${icon.restart}<span>New ground and start</span>` }))),
    runButton, h("div", { style: { marginTop: "12px" } }, result));
  paintGallery();
  paintControls();
  run();
  return { dispose: () => viewer.dispose() };
}

// ---------- The recipe tab ----------

function recipeTab(body, details, setup) {
  const animal = store.animal(setup.creature.animal);
  const catalog = store.catalog;
  const changed = (groups, values, defaults) => {
    const rows = [];
    for (const group of groups) for (const setting of group.settings) {
      const standard = defaults[setting.key] ?? setting.default;
      const value = values[setting.key] ?? standard;
      if (value !== standard) rows.push([setting.label, describe(setting, value), describe(setting, standard)]);
    }
    return rows;
  };
  const table = (rows, empty) => rows.length
    ? h("dl", { class: "kv" }, ...rows.flatMap(([label, value, standard]) => [h("dt", {}, label), h("dd", { title: `Standard: ${standard}` }, value)]))
    : h("div", { class: "faint" }, empty);
  const worldDefaults = Object.fromEntries(catalog.world_groups.flatMap((group) => group.settings.map((s) => [s.key, s.default])));
  const brain = setup.brain;
  body.append(h("div", { class: "grid grid-2" },
    h("div", { class: "card card-pad" }, h("h3", { style: { marginBottom: "10px" } }, `Creature: ${setup.creature.name || animal.name}`),
      table(changed(animal.groups, setup.creature.values, animal.defaults), `A standard ${animal.name.toLowerCase()}.`)),
    h("div", { class: "card card-pad" }, h("h3", { style: { marginBottom: "10px" } }, `World: ${setup.world.name}`),
      table(catalog.world_groups.flatMap((group) => group.settings.map((s) => [s.label, describe(s, setup.world.values[s.key] ?? worldDefaults[s.key]), ""]))
        , "")),
    h("div", { class: "card card-pad" }, h("h3", { style: { marginBottom: "10px" } }, "Brain"),
      h("dl", { class: "kv" }, h("dt", {}, "Layers"), h("dd", {}, brain.layers.join(" x ")), h("dt", {}, "Judge layers"), h("dd", {}, brain.judge_layers.join(" x ")),
        ...catalog.brain_groups.flatMap((group) => group.settings.flatMap((s) => [h("dt", {}, s.label), h("dd", {}, describe(s, brain.values[s.key]))])))),
    h("div", { class: "card card-pad" }, h("h3", { style: { marginBottom: "10px" } }, "Training"),
      h("dl", { class: "kv" }, ...catalog.learning_groups.flatMap((group) => group.settings.flatMap((s) => [h("dt", {}, s.label), h("dd", {}, describe(s, setup.learning[s.key]))])),
        setup.start_from ? [h("dt", {}, "Started from"), h("dd", {}, setup.start_from)] : []))));
  return null;
}

function describe(setting, value) {
  if (setting.kind === "number") return `${fmtNumber(value, setting.step < 1 ? 2 : 0)}${setting.unit === "%" ? "%" : setting.unit ? ` ${setting.unit}` : ""}`;
  if (setting.kind === "bool") return value ? "On" : "Off";
  if (setting.kind === "choice") return (setting.choices.find((choice) => choice.value === value) || {}).label || String(value);
  if (setting.kind === "colour") return value;
  return String(value);
}
