// A look inside a walk: which feet touch the ground over time (a gait diagram) and what every muscle is
// being told right now. It follows the viewer's playback.

import { h } from "./ui.js";

const FOOT_COLOURS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#0ea5e9", "#a855f7", "#14b8a6", "#ec4899"];

function pretty(name) {
  return name.replace(/_/g, " ");
}

export class GaitPanel {
  constructor(viewer) {
    this.viewer = viewer;
    this.element = h("div", { class: "card card-pad gait", hidden: true });
    this.canvas = h("canvas", { class: "gait-canvas" });
    this.bars = h("div", { class: "muscles" });
    this.element.append(
      h("div", { class: "spread" }, h("div", {}, h("h3", {}, "Inside the walk"),
        h("div", { class: "faint", style: { fontSize: "12.5px" } }, "Top: when each foot touches the ground (its rhythm, or gait). Bottom: what the brain tells every muscle at this moment."))),
      this.canvas, this.bars);
    viewer.onTime = (time) => this.paint(time);
    this.lastIndex = -1;
  }

  show(replay) {
    this.replay = replay;
    const ok = replay && replay.contacts && replay.contacts.length > 1;
    this.element.hidden = !ok;
    if (!ok) return;
    this.barElements = [];
    this.bars.replaceChildren(...replay.motors.map((name) => {
      const fill = h("div", { class: "muscle-fill" });
      this.barElements.push(fill);
      return h("div", { class: "muscle" }, h("span", { class: "muscle-name", title: pretty(name) }, pretty(name)),
        h("div", { class: "muscle-track" }, fill));
    }));
    this.lastIndex = -1;
    this.drawBase();
  }

  // The footfall pattern of the whole replay, drawn once
  drawBase() {
    const replay = this.replay;
    const feet = replay.feet;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(300, this.canvas.clientWidth || 600);
    const rowHeight = 16, top = 4, left = 92;
    const height = top + feet.length * (rowHeight + 4);
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    this.canvas.style.height = `${height}px`;
    const context = this.canvas.getContext("2d");
    context.scale(ratio, ratio);
    this.geometry = { width, height, rowHeight, top, left };
    const frames = replay.contacts.length;
    const step = (width - left - 4) / frames;
    const styles = getComputedStyle(document.documentElement);
    context.font = "12px Inter, system-ui, sans-serif";
    context.textBaseline = "middle";
    feet.forEach((name, row) => {
      const y = top + row * (rowHeight + 4);
      context.fillStyle = styles.getPropertyValue("--text-2") || "#555";
      context.fillText(pretty(name), 0, y + rowHeight / 2);
      context.fillStyle = styles.getPropertyValue("--surface-2") || "#eee";
      context.fillRect(left, y, width - left - 4, rowHeight);
      context.fillStyle = FOOT_COLOURS[row % FOOT_COLOURS.length];
      replay.contacts.forEach((touching, frame) => {
        if (touching[row]) context.fillRect(left + frame * step, y, Math.ceil(step), rowHeight);
      });
    });
    this.baseImage = context.getImageData(0, 0, this.canvas.width, this.canvas.height);
  }

  paint(time) {
    if (!this.replay || this.element.hidden || !this.baseImage) return;
    const replay = this.replay;
    const index = Math.min(replay.contacts.length - 1, Math.round(time * replay.fps));
    if (index === this.lastIndex) return;
    this.lastIndex = index;
    const context = this.canvas.getContext("2d");
    context.putImageData(this.baseImage, 0, 0);
    const { width, height, left } = this.geometry;
    const x = left + (index / replay.contacts.length) * (width - left - 4);
    context.fillStyle = "#111827";
    context.fillRect(x - 1, 0, 2, height);
    const commands = replay.commands[index] || [];
    commands.forEach((value, i) => {
      const fill = this.barElements[i];
      if (!fill) return;
      fill.style.left = value >= 0 ? "50%" : `${50 + value * 50}%`;
      fill.style.width = `${Math.abs(value) * 50}%`;
      fill.classList.toggle("negative", value < 0);
    });
  }
}
