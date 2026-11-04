// Walk Studio: the app shell, the shared state and the page router.

import { h, api, toast, fmtSteps, clear } from "./ui.js";
import { icon, logoMark } from "./icons.js";

// ---------- Shared state ----------

const DRAFT_KEY = "walk-studio-draft-v1";

export const store = {
  catalog: null,
  draft: null,
  trainings: [],

  animal(key) {
    return this.catalog.animals.find((animal) => animal.key === key) || this.catalog.animals[0];
  },

  newDraft(animalKey = "duck") {
    const animal = this.animal(animalKey);
    const world = this.catalog.world_templates[0];
    return {
      name: "",
      creature: { animal: animal.key, name: animal.key === "custom" ? "My creation" : `My ${animal.name.toLowerCase()}`, values: JSON.parse(JSON.stringify(animal.defaults)), id: null },
      world: JSON.parse(JSON.stringify(world)),
      brain: JSON.parse(JSON.stringify(this.catalog.brain_default)),
      brainPreset: "standard",
      learning: { ...this.catalog.learning_default, parallel_worlds: Math.max(1, Math.min(8, (this.catalog.system.cores || 4))) },
      learningPreset: null,
      start_from: null,
    };
  },

  loadDraft() {
    try {
      const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
      if (saved && saved.creature && saved.world && saved.brain && saved.learning) {
        const animal = this.animal(saved.creature.animal);
        saved.creature.values = { ...JSON.parse(JSON.stringify(animal.defaults)), ...saved.creature.values };
        saved.learning = { ...this.catalog.learning_default, ...saved.learning };
        this.draft = saved;
        return;
      }
    } catch { /* a broken draft starts fresh */ }
    this.draft = this.newDraft();
  },

  saveDraft() {
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify(this.draft)); } catch { /* private mode */ }
  },

  resetDraft(animalKey) {
    this.draft = this.newDraft(animalKey);
    this.saveDraft();
  },
};

// ---------- Theme ----------

function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function toggleTheme() {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  const next = dark ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("walk-studio-theme", next); } catch { /* fine */ }
}

// ---------- Sidebar ----------

const NAV = [
  { href: "#/", label: "Home", icon: icon.home, match: (path) => path === "" },
  { href: "#/new/creature", label: "New walker", icon: icon.plus, match: (path) => path.startsWith("new"), cta: true },
  { href: "#/walkers", label: "My walkers", icon: icon.walkers, match: (path) => path.startsWith("walker") },
  { href: "#/library", label: "Library", icon: icon.library, match: (path) => path.startsWith("library") },
  { href: "#/guide", label: "How it works", icon: icon.book, match: (path) => path.startsWith("guide") },
];

function renderSidebar(path) {
  const sidebar = document.getElementById("sidebar");
  clear(sidebar);
  sidebar.append(
    h("a", { class: "brand", href: "#/" },
      h("div", { class: "brand-mark", html: logoMark() }),
      h("div", {}, h("div", { class: "brand-name" }, "Walk Studio"), h("div", { class: "brand-sub" }, "Teach creatures to walk"))),
    h("nav", { class: "nav" },
      ...NAV.slice(0, 2).map((item) => navLink(item, path)),
      h("div", { class: "nav-label" }, "Your work"),
      ...NAV.slice(2, 4).map((item) => navLink(item, path)),
      h("div", { class: "nav-label" }, "Help"),
      navLink(NAV[4], path)),
    h("div", { class: "sidebar-foot" },
      h("div", { id: "running" }),
      h("div", { class: "sidebar-tools" },
        h("span", {}, `${store.catalog.system.cores} processor cores`),
        h("button", { class: "icon-button-dark", title: "Light or dark", html: icon.moon, onclick: toggleTheme }))));
  paintRunning();
}

function navLink(item, path) {
  return h("a", { href: item.href, class: `${item.match(path) ? "active" : ""} ${item.cta ? "nav-cta" : ""}`, html: `${item.icon}<span>${item.label}</span>` });
}

function paintRunning() {
  const box = document.getElementById("running");
  if (!box) return;
  const running = store.trainings.filter((t) => ["starting", "training", "examining"].includes(t.status.state));
  box.replaceChildren(...running.slice(0, 2).map((t) => {
    const share = t.status.total_steps ? Math.min(1, (t.status.steps || 0) / t.status.total_steps) : 0;
    return h("div", { class: "running-card", onclick: () => { location.hash = `#/walker/${t.id}`; } },
      h("div", { class: "title" }, h("span", { class: "spinner", style: { width: "11px", height: "11px", color: "#a5b4fc" } }), t.name),
      h("div", { style: { marginTop: "4px", color: "#8c92ad" } }, `${fmtSteps(t.status.steps || 0)} of ${fmtSteps(t.status.total_steps)} steps`),
      h("div", { class: "bar" }, h("div", { style: { width: `${share * 100}%` } })));
  }));
}

async function pollTrainings() {
  try {
    store.trainings = await api("trainings");
    paintRunning();
  } catch { /* the server may be restarting */ }
}

// ---------- Router ----------

const PAGES = {
  "": () => import("./pages/home.js"),
  new: () => import("./pages/wizard.js"),
  walkers: () => import("./pages/walkers.js"),
  walker: () => import("./pages/walker.js"),
  library: () => import("./pages/library.js"),
  guide: () => import("./pages/guide.js"),
};

let cleanup = null;
let routeNumber = 0;

async function route() {
  const path = location.hash.replace(/^#\/?/, "");
  const [page, ...params] = path.split("/");
  const loader = PAGES[page] || PAGES[""];
  renderSidebar(path);
  const number = ++routeNumber;
  if (cleanup) { try { cleanup(); } catch { /* fine */ } cleanup = null; }
  const main = document.getElementById("main");
  clear(main);
  try {
    const module = await loader();
    if (number !== routeNumber) return;
    cleanup = (await module.render(main, params)) || null;
  } catch (error) {
    console.error(error);
    main.append(h("div", { class: "page" }, h("div", { class: "callout bad", html: `${icon.info}<div>${error.message}</div>` })));
  }
  window.scrollTo(0, 0);
}

export function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

async function start() {
  try { applyTheme(localStorage.getItem("walk-studio-theme")); } catch { /* fine */ }
  try {
    store.catalog = await api("catalog");
  } catch (error) {
    document.getElementById("main").append(h("div", { class: "page" }, h("h1", {}, "The studio is not answering"),
      h("p", { class: "muted" }, "Make sure the Walk Studio window is still open, then reload this page.")));
    return;
  }
  store.loadDraft();
  await pollTrainings();
  setInterval(pollTrainings, 5000);
  window.addEventListener("hashchange", route);
  route();
}

window.addEventListener("unhandledrejection", (event) => {
  toast("Something went wrong", event.reason ? event.reason.message : "", "bad");
});

start();
