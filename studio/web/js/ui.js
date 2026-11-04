// Small building blocks for the pages: making elements, controls for settings, toasts, dialogs and the API.

import { icon, groupIcon } from "./icons.js";

// ---------- Elements ----------

export function h(tag, attrs = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") element.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(element.style, value);
    else if (key === "html") element.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") element.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "dataset") Object.assign(element.dataset, value);
    else if (value === true) element.setAttribute(key, "");
    else element.setAttribute(key, value);
  }
  append(element, children);
  return element;
}

function append(element, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export const svg = (markup, cls = "") => h("span", { class: `svg-wrap ${cls}`, style: { display: "contents" }, html: markup });

export function clear(element) {
  while (element.firstChild) element.firstChild.remove();
  return element;
}

// ---------- Formatting ----------

export function fmtSteps(steps) {
  if (steps === null || steps === undefined) return "-";
  if (steps >= 1e6) return `${(steps / 1e6).toFixed(steps >= 1e7 ? 0 : 1)}M`;
  if (steps >= 1e3) return `${Math.round(steps / 1e3)}k`;
  return String(Math.round(steps));
}

export function fmtDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "-";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} h ${rest} min` : `${hours} h`;
}

export function fmtNumber(value, digits = 0) {
  if (value === null || value === undefined || !isFinite(value)) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function fmtWhen(text) {
  if (!text) return "";
  const date = new Date(text);
  const seconds = (Date.now() - date.getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function settingText(setting, value) {
  if (setting.kind === "number") {
    const digits = setting.step < 0.01 ? 3 : setting.step < 0.1 ? 2 : setting.step < 1 ? 1 : 0;
    const unit = setting.unit === "%" ? "%" : setting.unit ? ` ${setting.unit}` : "";
    const sign = setting.key === "leg_position" && value > 0 ? "+" : "";
    return `${sign}${fmtNumber(value, digits)}${unit}`;
  }
  if (setting.kind === "bool") return value ? "On" : "Off";
  if (setting.kind === "choice") return (setting.choices.find((choice) => choice.value === value) || {}).label || value;
  return value;
}

export function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

// ---------- API ----------

export async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(`/api/${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await response.json(); } catch { data = null; }
  if (!response.ok) throw new Error((data && data.error) || `The studio answered ${response.status}.`);
  return data;
}

// ---------- Toasts and dialogs ----------

export function toast(title, detail = "", kind = "") {
  let box = document.querySelector(".toasts");
  if (!box) { box = h("div", { class: "toasts" }); document.body.append(box); }
  const item = h("div", { class: `toast ${kind}` }, h("div", { class: "t" }, title), detail ? h("div", { class: "d" }, detail) : null);
  box.append(item);
  setTimeout(() => { item.style.opacity = "0"; item.style.transition = "opacity .3s"; }, 4200);
  setTimeout(() => item.remove(), 4600);
}

export function modal({ title, text = "", body = null, actions = [] }) {
  return new Promise((resolve) => {
    const close = (value) => { back.remove(); resolve(value); };
    const back = h("div", { class: "modal-back", onclick: (event) => { if (event.target === back) close(null); } },
      h("div", { class: "modal" },
        h("div", { class: "modal-head" }, h("h2", {}, title)),
        h("div", { class: "modal-body" }, text ? h("p", { class: "muted" }, text) : null, body),
        h("div", { class: "modal-foot" },
          ...actions.map((action) => h("button", {
            class: `btn ${action.kind || ""}`,
            onclick: () => close(action.value !== undefined ? action.value : action.label),
          }, action.label)))));
    document.body.append(back);
    const first = back.querySelector("input");
    if (first) first.focus();
  });
}

export async function confirmDialog(title, text, yes = "Yes", kind = "primary") {
  const answer = await modal({ title, text, actions: [{ label: "Cancel", kind: "ghost", value: false }, { label: yes, kind, value: true }] });
  return answer === true;
}

export async function askName(title, text, value) {
  const input = h("input", { class: "input", value: value || "" });
  const answer = await modal({ title, text, body: input, actions: [{ label: "Cancel", kind: "ghost", value: null }, { label: "Save", kind: "primary", value: "save" }] });
  return answer === "save" ? input.value.trim() : null;
}

// ---------- Setting controls ----------

export function tipIcon(text) {
  if (!text) return null;
  return h("span", { class: "tip-icon", title: text, html: icon.info });
}

// One control for one setting. onChange(value) is called while it changes
export function settingControl(setting, value, onChange, { defaultValue = setting.default, onHover } = {}) {
  const field = h("div", { class: "field", dataset: { key: setting.key } });
  if (onHover) {
    field.addEventListener("mouseenter", () => onHover(setting.key));
    field.addEventListener("mouseleave", () => onHover(null));
  }
  const resetButton = h("button", { class: "reset-link", title: "Back to the standard value", html: icon.reset });
  const showReset = (current) => { resetButton.style.visibility = current === defaultValue ? "hidden" : "visible"; };

  if (setting.kind === "number") {
    const range = h("input", { type: "range", min: setting.minimum, max: setting.maximum, step: setting.step, value });
    const box = h("input", { class: "field-value", value: settingText(setting, value) });
    const paint = (current) => {
      const share = (current - setting.minimum) / (setting.maximum - setting.minimum || 1);
      range.style.setProperty("--fill", `${Math.max(0, Math.min(1, share)) * 100}%`);
      box.value = settingText(setting, current);
      showReset(current);
    };
    const set = (current, notify = true) => {
      current = Math.min(setting.maximum, Math.max(setting.minimum, current));
      range.value = current;
      paint(current);
      if (notify) onChange(current);
    };
    range.addEventListener("input", () => set(Number(range.value)));
    box.addEventListener("focus", () => { box.value = String(Number(range.value)); box.select(); });
    box.addEventListener("change", () => { const typed = parseFloat(box.value.replace(",", ".")); if (isFinite(typed)) set(typed); else paint(Number(range.value)); });
    box.addEventListener("blur", () => paint(Number(range.value)));
    box.addEventListener("keydown", (event) => { if (event.key === "Enter") box.blur(); });
    resetButton.addEventListener("click", () => set(defaultValue));
    paint(value);
    field.append(
      h("div", { class: "field-top" }, h("span", { class: "field-label" }, setting.label, tipIcon(setting.help)), h("span", { class: "row", style: { gap: "4px" } }, resetButton, box)),
      range);
    field.setValue = (current) => set(current, false);
    return field;
  }

  if (setting.kind === "bool") {
    const input = h("input", { type: "checkbox" });
    input.checked = !!value;
    input.addEventListener("change", () => { showReset(input.checked); onChange(input.checked); });
    resetButton.addEventListener("click", () => { input.checked = !!defaultValue; showReset(input.checked); onChange(input.checked); });
    showReset(!!value);
    field.append(h("div", { class: "field-top", style: { marginBottom: 0 } },
      h("span", { class: "field-label" }, setting.label), h("span", { class: "row", style: { gap: "6px" } }, resetButton, h("label", { class: "switch" }, input, h("span")))));
    if (setting.help) field.append(h("div", { class: "field-help" }, setting.help));
    field.setValue = (current) => { input.checked = !!current; showReset(!!current); };
    return field;
  }

  if (setting.kind === "choice") {
    const short = setting.choices.length <= 4 && setting.choices.every((choice) => String(choice.label).length <= 14);
    let current = value;
    const container = h("div", { class: short ? "segmented" : "choice-list" });
    const paint = () => {
      [...container.children].forEach((child, index) => child.classList.toggle("on", setting.choices[index].value === current));
      showReset(current);
    };
    const pick = (choiceValue) => { current = choiceValue; paint(); onChange(choiceValue); };
    for (const choice of setting.choices) {
      container.append(short
        ? h("button", { type: "button", onclick: () => pick(choice.value) }, choice.label)
        : h("div", { class: "choice-item", onclick: () => pick(choice.value) }, h("span", { class: "dot" }), choice.label));
    }
    resetButton.addEventListener("click", () => pick(defaultValue));
    paint();
    field.append(h("div", { class: "field-top" }, h("span", { class: "field-label" }, setting.label), resetButton), container);
    if (setting.help) field.append(h("div", { class: "field-help" }, setting.help));
    field.setValue = (choiceValue) => { current = choiceValue; paint(); };
    return field;
  }

  if (setting.kind === "colour") {
    const input = h("input", { type: "color", value });
    input.addEventListener("input", () => { showReset(input.value); onChange(input.value); });
    resetButton.addEventListener("click", () => { input.value = defaultValue; showReset(input.value); onChange(input.value); });
    showReset(value);
    field.append(h("div", { class: "colour-field" }, input, h("span", { class: "field-label" }, setting.label), resetButton));
    field.style.borderTop = "0";
    field.setValue = (colourValue) => { input.value = colourValue; showReset(colourValue); };
    return field;
  }
  return field;
}

// All the setting groups as collapsible cards. values is changed in place; onChange(key, value) follows
export function settingGroups(groups, values, onChange, { open = [], onHover, defaults = {}, only = null } = {}) {
  const wrap = h("div", { class: "groups" });
  const controls = {};
  for (const group of groups) {
    if (only && !only.includes(group.key)) continue;
    const isColours = group.settings.every((setting) => setting.kind === "colour");
    const body = h("div", { class: isColours ? "group-body colours-grid" : "group-body" });
    const dot = h("span", { class: "changed-dot", hidden: true });
    const markChanged = () => {
      dot.hidden = !group.settings.some((setting) => values[setting.key] !== (setting.key in defaults ? defaults[setting.key] : setting.default));
    };
    for (const setting of group.settings) {
      const standard = setting.key in defaults ? defaults[setting.key] : setting.default;
      const control = settingControl(setting, values[setting.key] ?? standard, (value) => {
        values[setting.key] = value;
        markChanged();
        onChange(setting.key, value);
      }, { defaultValue: standard, onHover });
      controls[setting.key] = control;
      body.append(control);
    }
    const card = h("div", { class: `group ${open.includes(group.key) ? "open" : ""}` },
      h("div", { class: "group-head", onclick: () => card.classList.toggle("open") },
        h("div", { class: "group-icon", html: groupIcon[group.key] || icon.sparkle }),
        h("div", {}, h("div", { class: "group-title" }, group.title, dot), h("div", { class: "group-desc" }, group.description)),
        h("span", { class: "group-chevron", html: icon.chevron, style: { width: "18px", height: "18px", display: "grid" } })),
      body);
    markChanged();
    wrap.append(card);
  }
  wrap.controls = controls;
  return wrap;
}

export function statCard(label, value, note = "", iconMarkup = null) {
  return h("div", { class: "stat" },
    h("div", { class: "label", html: (iconMarkup || "") + `<span>${label}</span>` }),
    h("div", { class: "value", html: value }),
    note ? h("div", { class: "note" }, note) : null);
}

export function pillForState(state) {
  const map = {
    starting: ["info live", "Starting"], training: ["accent live", "Training"], examining: ["accent live", "Taking an exam"],
    finished: ["good", "Finished"], stopped: ["warn", "Paused"], interrupted: ["warn", "Interrupted"], error: ["bad", "Problem"],
  };
  const [cls, text] = map[state] || ["", state || "Unknown"];
  return h("span", { class: `pill ${cls}` }, h("span", { class: "dot" }), text);
}
