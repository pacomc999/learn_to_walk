// One walker: follow its training live, watch how it grew up, and take it for a test drive in any world.

import { h, api, toast, fmtSteps, fmtNumber, fmtDuration, fmtWhen, pillForState, confirmDialog, askName, statCard, settingControl, modal } from "../ui.js";
import { icon, animalArt, worldArt } from "../icons.js";
import { Viewer } from "../viewer.js";
import { lineChart, legend } from "../charts.js";
import { GaitPanel } from "../gait.js";
import { store, go } from "../app.js";

const ACTIVE = ["starting", "training", "examining"];

// The course stage boundaries of the walker on screen, drawn on every chart
const currentMarkers = [];

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
        h("div", { class: "sub" }, `${setup.creature.name || animal.name} · ${(setup.course || []).length
          ? `course: ${[setup.world.name, ...setup.course.map((stage) => stage.world.name)].join(" → ")}` : setup.world.name} · started ${fmtWhen(setup.created)}`))),
    actions));

  const tabs = h("div", { class: "tabs" },
    ...[["training", "Training"], ["drive", "Test drive"], ["live", "Live"], ["report", "Report card"], ["recipe", "Recipe"]].map(([key, label]) =>
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
      !running && details.has_brain ? h("button", { class: "btn", onclick: trainLonger, dataset: { action: "longer" }, html: `${icon.clock}<span>Train longer</span>` }) : null,
      details.has_brain ? h("button", { class: "btn", onclick: trainMore, html: `${icon.sparkle}<span>Teach a new world</span>` }) : null,
      h("a", { class: "btn ghost icon-btn", title: "Share: save this walker as a file", href: `/api/trainings/${id}/share`, download: "", html: icon.share }),
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
  const trainLonger = async () => {
    const input = h("input", { class: "input", type: "number", min: 0.1, max: 100, step: 0.5, value: 2 });
    const answer = await modal({ title: "Train longer", text: "How many more million practice steps? It carries on from its latest brain, in the same world.", body: input,
      actions: [{ label: "Cancel", kind: "ghost", value: null }, { label: "Train longer", kind: "primary", value: "go" }] });
    if (answer !== "go") return;
    try {
      await api(`trainings/${id}/extend`, { method: "POST", body: { million: Number(input.value) || 2 } });
      toast("Training longer", `${input.value} million more steps.`, "good");
      refresh();
    } catch (error) {
      toast("Could not train longer", error.message, "bad");
    }
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
  else if (tab === "report") view = await reportTab(body, id, details);
  else if (tab === "live") view = await liveTab(body, id, details, setup);
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
  const gait = new GaitPanel(viewer);
  viewerBox.parentElement.after(gait.element);
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
      gait.show(replay);
      if (name === "best") saveThumbnail(id, viewer, replay);
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
        last ? `score ${fmtNumber(last.score)} points${last.joules_per_metre ? ` · ${fmtNumber(last.joules_per_metre, 1)} J per metre` : ""}` : "the first exam comes right at the start", icon.target),
      statCard("Practice speed", speed ? `${fmtNumber(speed)}<small>steps/s</small>` : "-", `${setup.learning.parallel_worlds} worlds at once`, icon.speed),
      (setup.course || []).length ? statCard("Stage", `${(status.stage || 0) + 1}<small>of ${setup.course.length + 1}</small>`,
        status.stage_world || setup.world.name, icon.flag) : null,
      setup.world.values.gradually_harder && !(setup.course || []).length ? statCard("Ground height", `${fmtNumber((status.difficulty ?? 1) * 100)}<small>%</small>`, "of the tallest, rising as it learns", icon.stairs) : null].filter(Boolean));
    const active = ["starting", "training", "examining"].includes(status.state);
    const tips = !active || (status.steps || 0) > 1.5e6 ? coachAdvice(data, {
      report: () => go(`#/walker/${id}/report`),
      longer: active ? null : () => document.querySelector('button[data-action="longer"]')?.click(),
      copy: (changes, what) => retrainWith(setup, changes, what),
    }) : [];
    story.replaceChildren(storyCallout(status, exams, progress), tips.length ? h("div", { class: "card card-pad", style: { marginTop: "12px" } },
      h("h3", { style: { marginBottom: "10px" } }, "Coach's notes"),
      h("div", { class: "stack", style: { gap: "8px" } }, ...tips.map(([kind, title, text, ...actions]) => {
        const note = h("div", { class: `callout ${kind === "info" ? "" : kind}`, html: `${icon.info}<div><b>${title}.</b> ${text}</div>` });
        const buttons = actions.filter(Boolean).map(([label, run]) => h("button", { class: "btn small", onclick: run }, label));
        if (buttons.length) note.lastElementChild.append(h("div", { class: "btn-row", style: { marginTop: "8px" } }, ...buttons));
        return note;
      }))) : null);

    // Charts
    const examPoints = (key) => exams.map((exam) => [exam.steps, exam[key]]);
    const qualitySeries = [{ name: "Exam: share of the asked distance", colour: "#10b981", points: examPoints("quality"), dots: true, fill: true }];
    const scoreSeries = [
      { name: "Practice score", colour: "#6366f1", points: progress.map((row) => [row.steps, row.score]) },
      { name: "Exam score", colour: "#f59e0b", points: examPoints("score"), dots: true },
    ];
    const lengthSeries = [{ name: "Seconds per attempt", colour: "#0ea5e9", points: progress.map((row) => [row.steps, row.seconds]), fill: true }];
    const fallSeries = [{ name: "Attempts that fell", colour: "#ef4444", points: progress.map((row) => [row.steps, row.fall_share === null ? null : row.fall_share * 100]), fill: true }];
    // Where each course stage begins (every chart below marks them)
    const markers = currentMarkers;
    markers.length = 0;
    let start = setup.learning.million_steps * 1e6;
    for (const stage of setup.course || []) { markers.push([start, stage.world.name]); start += stage.million_steps * 1e6; }
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

// A new walker like this one with a change, ready on the Training step of the designer
function retrainWith(setup, changes, what) {
  store.draft = {
    name: `${setup.name} (${what})`,
    creature: { ...setup.creature, values: { ...setup.creature.values, ...(changes.creature || {}) }, id: null },
    world: { ...JSON.parse(JSON.stringify(setup.world)), builtin: false, values: { ...setup.world.values, ...(changes.world || {}) } },
    brain: JSON.parse(JSON.stringify(setup.brain)), brainPreset: null, learning: { ...setup.learning }, learningPreset: null,
    start_from: null, course: JSON.parse(JSON.stringify(setup.course || [])),
  };
  store.saveDraft();
  toast("Ready to train", "Check the plan and press Start training.");
  go("#/new/plan");
}

// A picture of the best walk for the walker's card, taken once it is under way (once per best brain)
function saveThumbnail(id, viewer, replay) {
  const key = `walk-studio-thumb-${id}`;
  const stamp = `${replay.steps}`;
  try { if (localStorage.getItem(key) === stamp) return; } catch { /* fine */ }
  setTimeout(async () => {
    if (viewer.disposed || viewer.replay !== replay) return;
    const playing = viewer.playing;
    viewer.seek(Math.min(0.6, viewer.duration));  // near the start, standing, framed by the camera
    const blob = await viewer.thumbnail();
    if (playing) viewer.play();
    if (!blob) return;
    await fetch(`/api/trainings/${id}/thumbnail`, { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
    try { localStorage.setItem(key, stamp); } catch { /* fine */ }
  }, 400);
}

function chartCard(title, description, series, options) {
  return h("div", { class: "card" },
    h("div", { class: "card-head" }, h("div", {}, h("h3", {}, title), h("div", { class: "desc" }, description))),
    h("div", { class: "card-body" }, lineChart(series, { height: 200, markers: currentMarkers, ...options }), h("div", { style: { marginTop: "8px" } }, legend(series))));
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
  const drive = { speed: null, height: null, seed: 1000, seconds: 20, turn: 0 };
  const steerable = !!setup.brain.values.feel_turn;

  const viewerBox = h("div", { class: "viewer-mid" });
  const panel = h("div", { class: "panel" });
  const stage = h("div", { class: "stage" }, viewerBox);
  body.append(h("div", { class: "workbench" }, panel, stage));
  const viewer = new Viewer(viewerBox, { pace: true });
  const gait = new GaitPanel(viewer);
  stage.append(gait.element);

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
      steerable ? settingControl({ key: "turn", label: "Asked turn", kind: "number", minimum: -90, maximum: 90, step: 5, unit: "°/s",
        help: "Plus turns left, minus turns right. It learned turns up to its world's sharpest turn." }, drive.turn, (value) => { drive.turn = value; }, { defaultValue: 0 }) : null,
      settingControl({ key: "seconds", label: "Length", kind: "number", minimum: 5, maximum: 60, step: 5, unit: "s" }, drive.seconds, (value) => { drive.seconds = value; }, { defaultValue: 20 }));
  };

  const result = h("div");
  const runButton = h("button", { class: "btn primary big", style: { width: "100%" }, html: `${icon.play}<span>Drive</span>` });
  let latest = 0;
  const run = async () => {
    const request = ++latest;  // only the newest drive is shown, even if an older one answers later
    const asked = chosen;
    runButton.disabled = true;
    viewer.setLoading(true);
    try {
      const v = values();
      const replay = await api(`trainings/${id}/watch`, { method: "POST", body: {
        world: { ...chosen, values: { ...v, attempt_seconds: Math.max(v.attempt_seconds, drive.seconds) } }, speed: drive.speed,
        height: (drive.height || 0) / 100, seed: drive.seed, seconds: drive.seconds, turn: drive.turn } });
      if (request !== latest || viewer.disposed) return;
      viewer.load(replay);
      gait.show(replay);
      const s = replay.stats;
      viewer.setChips([["World", asked.name], ["Result", s.fell ? `fell after ${fmtNumber(s.seconds, 1)} s` : `walked all ${fmtNumber(s.seconds, 0)} s`],
        ["Distance", `${fmtNumber(s.distance, 2)} m`], ["Speed", `${fmtNumber(s.average_speed, 2)} of ${fmtNumber(s.target_speed, 2)} m/s`]]);
      result.replaceChildren(h("div", { class: `callout ${s.fell ? "warn" : "good"}`, html: `${icon.info}<div>${s.fell
        ? `It fell after ${fmtNumber(s.seconds, 1)} seconds, ${fmtNumber(s.distance, 2)} m from the start.`
        : `It stayed up for all ${fmtNumber(s.seconds, 0)} seconds and went ${fmtNumber(s.distance, 2)} m, at ${fmtNumber(s.average_speed, 2)} m/s (asked: ${fmtNumber(s.target_speed, 2)}).`}${s.rock_hits ? ` Rocks hit it ${s.rock_hits} times.` : ""} The blue ring moves at the asked speed.</div>` }));
    } catch (error) {
      if (request === latest) toast("The drive failed", error.message, "bad");
    } finally {
      if (request === latest) {
        runButton.disabled = false;
        viewer.setLoading(false);
      }
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
        setup.start_from ? [h("dt", {}, "Started from"), h("dd", {}, setup.start_from)] : [])),
    (setup.course || []).length ? h("div", { class: "card card-pad" }, h("h3", { style: { marginBottom: "10px" } }, "Course"),
      h("dl", { class: "kv" }, h("dt", {}, "Stage 1"), h("dd", {}, `${setup.world.name}, ${fmtNumber(setup.learning.million_steps, 1)} million`),
        ...setup.course.flatMap((stage, i) => [h("dt", {}, `Stage ${i + 2}`), h("dd", {}, `${stage.world.name}, ${fmtNumber(stage.million_steps, 1)} million`)]))) : null));
  return null;
}

function describe(setting, value) {
  if (setting.kind === "number") return `${fmtNumber(value, setting.step < 1 ? 2 : 0)}${setting.unit === "%" ? "%" : setting.unit ? ` ${setting.unit}` : ""}`;
  if (setting.kind === "bool") return value ? "On" : "Off";
  if (setting.kind === "choice") return (setting.choices.find((choice) => choice.value === value) || {}).label || String(value);
  if (setting.kind === "colour") return value;
  return String(value);
}

// ---------- The report card tab ----------

const MEDAL_LOOK = {
  gold: ["#f5b301", "Gold"], silver: ["#a8b3c2", "Silver"], bronze: ["#cd7f32", "Bronze"], none: ["var(--surface-3)", "No medal"],
};

export async function reportTab(body, id, details) {
  if (!details.has_brain) {
    body.append(h("div", { class: "empty" }, h("div", { class: "t" }, "No brain to test yet"),
      h("div", { class: "d" }, "The report card needs the walker's best brain, saved after its first exam.")));
    return null;
  }
  const viewerBox = h("div", { class: "viewer-mid" });
  const cards = h("div", { class: "grid grid-auto", style: { marginTop: "16px" } });
  const intro = h("div");
  body.append(intro, viewerBox, cards);
  const viewer = new Viewer(viewerBox, { pace: true });
  viewer.showMessage("Click a level below to watch that attempt.");

  let latestLevel = 0;
  const watchLevel = async (challenge, level) => {
    const request = ++latestLevel;
    viewer.setLoading(true);
    try {
      const result = await api(`trainings/${id}/challenges/${challenge.key}`, { method: "POST", body: { level: level.level } });
      if (request !== latestLevel || viewer.disposed) return;
      viewer.showMessage("");
      viewer.load(result.replay);
      viewer.setChips([[challenge.title, `${fmtNumber(level.level, 1)} ${challenge.unit}`],
        ["Result", result.passed ? "passed" : result.fell ? `fell after ${fmtNumber(result.seconds, 1)} s` : "too slow"],
        ["Speed", `${fmtNumber(result.average_speed, 2)} of ${fmtNumber(result.asked_speed, 2)} m/s`]]);
    } catch (error) {
      toast("Could not show that attempt", error.message, "bad");
    } finally {
      viewer.setLoading(false);
    }
  };

  const paint = (results) => {
    const challenges = results.challenges || [];
    const medals = { gold: 0, silver: 0, bronze: 0 };
    for (const challenge of challenges) if (medals[challenge.medal] !== undefined) medals[challenge.medal] += 1;
    intro.replaceChildren(h("div", { class: "spread", style: { marginBottom: "14px" } },
      h("div", {}, h("h2", {}, "Report card"), h("div", { class: "faint" },
        challenges.length ? `Each level is one ${results.seconds}-second attempt. It passes by staying up and keeping at least 60% of the asked speed. Taken ${fmtWhen(results.date)}.`
          : "Seven challenges, each a ladder of harder levels: speed, cobblestones, stairs, ice, a backpack, wind and other planets. It takes about a minute.")),
      h("div", { class: "row" },
        challenges.length ? h("div", { class: "row", style: { gap: "6px" } }, ...["gold", "silver", "bronze"].map((medal) =>
          h("span", { class: "pill", style: { background: MEDAL_LOOK[medal][0], color: "#fff", border: 0 } }, `${medals[medal]} ${MEDAL_LOOK[medal][1].toLowerCase()}`))) : null,
        runButton)));
    cards.replaceChildren(...challenges.map((challenge) => h("div", { class: "card card-pad" },
      h("div", { class: "spread" }, h("h3", {}, challenge.title),
        h("span", { class: "medal", style: { background: MEDAL_LOOK[challenge.medal][0] }, title: MEDAL_LOOK[challenge.medal][1] }, challenge.medal === "none" ? "-" : "★")),
      h("div", { class: "faint", style: { fontSize: "12.5px", margin: "2px 0 10px" } }, challenge.description),
      h("div", { class: "levels" }, ...challenge.levels.map((level) => h("button", {
        class: `level ${level.skipped ? "skipped" : level.passed ? "passed" : "failed"}`, disabled: level.skipped,
        title: level.skipped ? "Not tried: an easier level was failed" : "Watch this attempt", onclick: () => watchLevel(challenge, level) },
        `${+Number(level.level).toFixed(2)}`))),
      h("div", { class: "faint", style: { fontSize: "12px", marginTop: "6px" } }, `${challenge.unit} · passed ${challenge.passed} of ${challenge.levels.length}`))));
  };

  const runButton = h("button", { class: "btn primary", html: `${icon.trophy}<span>Take the report card</span>` });
  runButton.addEventListener("click", async () => {
    runButton.disabled = true;
    try {
      const { job } = await api(`trainings/${id}/challenges`, { method: "POST" });
      while (true) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        const status = await api(`jobs/${job}`);
        runButton.innerHTML = `<span class="spinner"></span><span>${Math.round(status.progress * 100)}% · ${status.message}</span>`;
        if (status.state === "done") { paint(status.result); break; }
        if (status.state === "error") { toast("The report card stopped", status.message, "bad"); break; }
      }
    } catch (error) {
      toast("Could not start the report card", error.message, "bad");
    } finally {
      runButton.disabled = false;
      runButton.innerHTML = `${icon.trophy}<span>Take it again</span>`;
    }
  });
  const saved = await api(`trainings/${id}/challenges`);
  if (saved.challenges) runButton.innerHTML = `${icon.trophy}<span>Take it again</span>`;
  paint(saved);
  return { dispose: () => viewer.dispose() };
}

// ---------- The live tab ----------

export async function liveTab(body, id, details, setup) {
  if (!details.has_brain) {
    body.append(h("div", { class: "empty" }, h("div", { class: "t" }, "No brain to drive yet"),
      h("div", { class: "d" }, "Live mode needs the walker's best brain, saved after its first exam.")));
    return null;
  }
  const catalog = store.catalog;
  const worldDefaults = Object.fromEntries(catalog.world_groups.flatMap((group) => group.settings.map((s) => [s.key, s.default])));
  const worlds = [{ ...setup.world, name: `${setup.world.name} (its own)` }, ...catalog.world_templates.filter((w) => w.id !== setup.world.id), ...(await api("worlds"))];
  let world = worlds[0];
  const own = setup.world.values;
  let speed = Math.round((own.slowest_speed + own.fastest_speed) / 2 * 20) / 20;
  let session = null;
  let last = -1;
  let falls = 0;
  const steerable = !!setup.brain.values.feel_turn;
  const sharpest = Math.max(30, setup.world.values.turn_rate || 30);
  let turn = 0;
  let timer = null;
  let disposed = false;

  const viewerBox = h("div", { class: "viewer-tall", tabindex: "0" });
  const panel = h("div", { class: "panel" });
  body.append(h("div", { class: "workbench" }, panel, h("div", { class: "stage" }, viewerBox)));
  const viewer = new Viewer(viewerBox, { playbar: false });
  viewer.showMessage("Press Start to let it walk in real time.");

  const speedText = h("div", { class: "live-speed" });
  const speedRange = h("input", { type: "range", min: 0, max: Math.max(2, own.fastest_speed * 1.6), step: 0.05, value: speed });
  const paintSpeed = () => {
    speedText.innerHTML = `${fmtNumber(speed, 2)}<small> m/s asked</small>`;
    speedRange.value = speed;
    speedRange.style.setProperty("--fill", `${speed / Number(speedRange.max) * 100}%`);
  };
  const setSpeed = (value) => {
    speed = Math.max(0, Math.min(Number(speedRange.max), Math.round(value * 100) / 100));
    paintSpeed();
    if (session) api(`live/${session}`, { method: "POST", body: { speed } }).catch(() => {});
  };
  speedRange.addEventListener("input", () => setSpeed(Number(speedRange.value)));
  const send = (command) => { if (session) api(`live/${session}`, { method: "POST", body: command }).catch(() => {}); };
  const turnText = h("div", { class: "live-speed" });
  const turnRange = h("input", { type: "range", min: -sharpest, max: sharpest, step: 5, value: 0 });
  const paintTurn = () => {
    turnText.innerHTML = turn === 0 ? "straight" : `${Math.abs(turn)}<small>°/s ${turn > 0 ? "left" : "right"}</small>`;
    turnRange.value = turn;
    turnRange.style.setProperty("--fill", `${(turn + sharpest) / (2 * sharpest) * 100}%`);
  };
  const setTurn = (value) => {
    turn = Math.max(-sharpest, Math.min(sharpest, Math.round(value / 5) * 5));
    paintTurn();
    send({ turn });
  };
  turnRange.addEventListener("input", () => setTurn(Number(turnRange.value)));

  const stats = h("div", { class: "stats", style: { gridTemplateColumns: "repeat(3, 1fr)" } });
  const paintStats = (state) => stats.replaceChildren(
    statCard("Going", `${fmtNumber(Math.max(0, state.speed), 2)}<small>m/s</small>`, `asked ${fmtNumber(state.asked, 2)}`),
    statCard("Distance", `${fmtNumber(Math.max(0, state.x), 1)}<small>m</small>`, state.fallen ? "down, getting up..." : "since the last start"),
    statCard("Falls", String(falls), "pushes and stumbles"));
  paintStats({ speed: 0, asked: speed, x: 0, fallen: false });

  const poll = async () => {
    if (!session || disposed) return;
    try {
      const update = await api(`live/${session}?after=${last}`);
      if (update.frames.length) {
        viewer.pushLive(update.frames);
        last = update.first + update.frames.length - 1;
      }
      falls += update.events.filter((event) => event === "fell").length;
      paintStats(update.state);
      if (!update.running) { stopLive(); viewer.showMessage("Live mode stopped. Press Start to go again."); return; }
    } catch (error) {
      stopLive();
      viewer.showMessage(error.message);
      return;
    }
    timer = setTimeout(poll, 100);
  };

  const startButton = h("button", { class: "btn primary big", style: { width: "100%" }, html: `${icon.play}<span>Start</span>` });
  const startLive = async () => {
    startButton.disabled = true;
    viewer.setLoading(true);
    try {
      const header = await api("live", { method: "POST", body: { training: id, speed, world: { ...world, values: { ...worldDefaults, ...world.values } } } });
      session = header.id;
      last = 0;
      falls = 0;
      header.marks_to = header.floor_end || 60;
      viewer.showMessage("");
      viewer.startLive(header);
      viewer.setChips([["World", world.name], ["Keys", steerable ? "↑ ↓ speed · ← → steer · A D push · R restart" : "↑ ↓ speed · ← → push · R restart"]]);
      turn = 0; paintTurn();
      startButton.innerHTML = `${icon.stop}<span>Stop</span>`;
      viewerBox.focus();
      poll();
    } catch (error) {
      toast("Live mode could not start", error.message, "bad");
    } finally {
      startButton.disabled = false;
      viewer.setLoading(false);
    }
  };
  const stopLive = () => {
    if (session) send({ stop: true });
    session = null;
    clearTimeout(timer);
    startButton.innerHTML = `${icon.play}<span>Start</span>`;
  };
  startButton.addEventListener("click", () => (session ? stopLive() : startLive()));

  const keys = (event) => {
    if (!session || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && document.activeElement !== speedRange) return;
    const actions = { ArrowUp: () => setSpeed(speed + 0.1), ArrowDown: () => setSpeed(speed - 0.1),
      ArrowLeft: () => (steerable ? setTurn(turn + 10) : send({ push: "left" })), ArrowRight: () => (steerable ? setTurn(turn - 10) : send({ push: "right" })),
      a: () => send({ push: "left" }), d: () => send({ push: "right" }), r: () => send({ restart: true }), R: () => send({ restart: true }) };
    if (actions[event.key]) { event.preventDefault(); actions[event.key](); }
  };
  window.addEventListener("keydown", keys);

  const worldSelect = h("select", { class: "input" }, ...worlds.map((w, index) => h("option", { value: String(index) }, w.name)));
  worldSelect.addEventListener("change", () => { world = worlds[Number(worldSelect.value)]; if (session) { stopLive(); startLive(); } });

  panel.append(
    h("div", { class: "card card-pad stack", style: { gap: "12px", marginBottom: "12px" } },
      h("div", {}, h("h3", {}, "Live"), h("div", { class: "faint", style: { fontSize: "12.5px" } }, "It walks in real time while you change the asked speed and push it around. Its brain decides every step, 50 times a second.")),
      h("div", {}, h("div", { class: "field-label", style: { marginBottom: "6px" } }, "World"), worldSelect),
      startButton),
    h("div", { class: "card card-pad stack", style: { gap: "10px", marginBottom: "12px" } },
      h("div", { class: "spread" }, h("h3", {}, "Asked speed"), speedText),
      speedRange,
      h("div", { class: "btn-row" },
        h("button", { class: "btn small", onclick: () => setSpeed(0) }, "Stop"),
        h("button", { class: "btn small", onclick: () => setSpeed(own.slowest_speed) }, "Slow"),
        h("button", { class: "btn small", onclick: () => setSpeed((own.slowest_speed + own.fastest_speed) / 2) }, "Medium"),
        h("button", { class: "btn small", onclick: () => setSpeed(own.fastest_speed) }, "Fast"))),
    steerable ? h("div", { class: "card card-pad stack", style: { gap: "10px", marginBottom: "12px" } },
      h("div", { class: "spread" }, h("h3", {}, "Steer"), turnText),
      turnRange,
      h("div", { class: "btn-row" },
        h("button", { class: "btn small", onclick: () => setTurn(sharpest / 2) }, "Left"),
        h("button", { class: "btn small", onclick: () => setTurn(0) }, "Straight"),
        h("button", { class: "btn small", onclick: () => setTurn(-sharpest / 2) }, "Right"))) : null,
    h("div", { class: "card card-pad stack", style: { gap: "10px", marginBottom: "12px" } },
      h("h3", {}, "Mess with it"),
      h("div", { class: "btn-row" },
        h("button", { class: "btn", onclick: () => send({ push: "left" }) }, "Push from the left"),
        h("button", { class: "btn", onclick: () => send({ push: "right" }) }, "Push from the right")),
      h("div", { class: "btn-row" },
        h("button", { class: "btn", onclick: () => send({ push: "front" }) }, "Push it back"),
        h("button", { class: "btn", onclick: () => send({ push: "back" }) }, "Push it forward"),
        h("button", { class: "btn ghost", onclick: () => send({ restart: true }), html: `${icon.restart}<span>Start again</span>` }))),
    stats);
  paintSpeed();
  paintTurn();
  return { dispose: () => { disposed = true; stopLive(); window.removeEventListener("keydown", keys); viewer.dispose(); } };
}

// ---------- The coach: plain-words advice from the results ----------

export function coachAdvice(details, act = {}) {
  const { setup, status, exams, progress } = details;
  const tips = [];
  const steps = status.steps || 0;
  const best = status.best_quality || 0;
  const last = exams[exams.length - 1];
  const world = setup.world.values;
  const creature = setup.creature.values || {};
  const direct = (creature.motor_type || "force") === "force";
  const recent = progress.slice(-10);
  const mean = (key) => (recent.length ? recent.reduce((sum, row) => sum + (row[key] || 0), 0) / recent.length : 0);
  const speedPoints = mean("pts_speed"), endStops = mean("pts_end_stops"), energy = mean("pts_energy");
  const hard = world.terrain !== "flat" && world.terrain_height > 0;

  if (best >= 85) {
    tips.push(["good", "A fine walker", "It covers nearly all of the asked distance. Try the Report card to see what else it can handle, race it, or teach it a harder world.",
      act.report && ["Take the report card", act.report]]);
  }
  if (steps >= 1.5e6 && best < 10) {
    tips.push(["warn", "It has not found a way forward yet", direct
      ? "Direct muscles take a while: learn_to_walk's Duck 4 needed about 2.5 million steps. Train longer, or try springy muscles, which hold the body up from the start and usually learn sooner."
      : "Train longer, check the drop test on the Creature step (can its muscles hold it up?), or make energy cheaper so moving costs less.",
      act.longer && ["Train longer", act.longer], direct && act.copy && ["Try springy muscles", () => act.copy({ creature: { motor_type: "position" } }, "springy")]]);
  }
  if (last && best >= 20 && last.falls >= last.attempts / 2) {
    tips.push(["warn", "It walks, but falls a lot", hard && !world.gradually_harder
      ? "Switch on Gradually harder for this world, so it learns on easier ground first, or give it the sense See the ground ahead."
      : "More practice usually helps. A narrower range of asked speeds also makes the task easier.",
      hard && !world.gradually_harder && act.copy ? ["Retrain with Gradually harder", () => act.copy({ world: { gradually_harder: true } }, "gradually harder")]
        : act.longer && ["Train longer", act.longer]]);
  }
  if (last && best >= 20 && last.speed_error > 0.3 * ((world.slowest_speed + world.fastest_speed) / 2)) {
    tips.push(["info", "It does not keep the asked speed well", "Raise the points for moving forward, or lower the energy cost, so going at the asked speed pays more."]);
  }
  const walks = best >= 20 && steps >= 500000;  // advice about style only makes sense once it walks a little
  if (walks && world.end_stops && speedPoints > 0 && -endStops > 0.3 * speedPoints) {
    tips.push(["info", "It leans on its joint limits a lot", "Resting joints on their stops holds it up for free, so it pays the joint limit penalty instead of moving them. This is the crouch learn_to_walk's Duck 4 learned. A bit more penalty may push it out of the habit; too much and it may give up."]);
  }
  if (!world.end_stops && direct && best >= 30) {
    tips.push(["info", "Watch out for crouches", "With direct muscles and no joint limit penalty, walkers often lock joints against their stops. Look at Inside the walk: muscles stuck at one end show it."]);
  }
  if (walks && speedPoints > 0 && -energy > 0.8 * speedPoints) {
    tips.push(["info", "Energy is expensive here", "Most of what it earns goes on muscle work. A lower energy cost lets it move more freely; a higher one gives calmer, more careful walking."]);
  }
  if (status.state === "finished" && (status.message || "").includes("stopped getting better") && best < 60) {
    tips.push(["info", "It stopped getting better", "Try a curiosity bonus (Training step, How it studies), a different brain size, or continue in a slightly easier world first with a course."]);
  }
  return tips;
}
