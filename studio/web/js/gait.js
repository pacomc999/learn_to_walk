// A look inside a walk: which feet touch the ground over time (a gait diagram) and what every muscle is
// being told right now. It follows the viewer's playback.

import { h } from "./ui.js";

const FOOT_COLOURS = ["#2f7d4f", "#c28a2c", "#3f7fb5", "#b5533c", "#6b7280", "#2a8f86", "#a8642a", "#b5577a"];

function pretty(name) {
  return name.replace(/_/g, " ");
}

export class GaitPanel {
  constructor(viewer) {
    this.viewer = viewer;
    this.element = h("div", { class: "card card-pad gait", hidden: true });
    this.canvas = h("canvas", { class: "gait-canvas" });
    this.bars = h("div", { class: "muscles" });
    this.brainCanvas = h("canvas", { class: "brain-canvas" });
    this.gaitFacts = h("div", { class: "gait-facts" });
    this.brainBox = h("div", { class: "brain-box" }, h("div", { class: "field-label", style: { fontSize: "12.5px", marginBottom: "4px" } }, "Its brain at work"),
      h("div", { class: "faint", style: { fontSize: "11.5px", marginBottom: "6px" } }, "A sample of the thinking cells of each layer: orange is firing one way, blue the other."),
      this.brainCanvas);
    this.element.append(
      h("div", { class: "spread" }, h("div", {}, h("h3", {}, "Inside the walk"),
        h("div", { class: "faint", style: { fontSize: "12.5px" } }, "When each foot touches the ground (its rhythm, or gait), what its brain is doing, and what it tells every muscle at this moment."))),
      this.canvas, this.gaitFacts, h("div", { class: "inside-grid" }, this.brainBox, h("div", {}, h("div", { class: "field-label", style: { fontSize: "12.5px", marginBottom: "8px" } }, "Muscle commands"), this.bars)));
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
    this.brainBox.hidden = !(replay.cells && replay.cells.length);
    this.paintGaitFacts();
    this.drawBase();
  }

  // The brain: columns of cells (each hidden layer's sample), then the muscles; colour shows each cell's value
  drawBrain(index) {
    const replay = this.replay;
    if (!replay.cells || !replay.cells.length) return;
    const layers = [...(replay.cells[index] || replay.cells[0]), replay.commands[index] || []];
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(220, this.brainCanvas.clientWidth || 320);
    const rows = Math.max(...layers.map((layer) => layer.length));
    const height = Math.min(260, rows * 10 + 16);
    if (this.brainCanvas.width !== width * ratio) {
      this.brainCanvas.width = width * ratio;
      this.brainCanvas.height = height * ratio;
      this.brainCanvas.style.height = `${height}px`;
    }
    const context = this.brainCanvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const xs = layers.map((_, i) => 14 + (width - 28) * i / Math.max(1, layers.length - 1));
    const ys = layers.map((layer) => layer.map((_, j) => 8 + (height - 16) * (layer.length === 1 ? 0.5 : j / (layer.length - 1))));
    context.strokeStyle = "rgba(115, 115, 115, 0.08)";
    for (let c = 0; c < layers.length - 1; c++) {
      for (const y1 of ys[c]) for (const y2 of ys[c + 1]) { context.beginPath(); context.moveTo(xs[c], y1); context.lineTo(xs[c + 1], y2); context.stroke(); }
    }
    layers.forEach((layer, c) => layer.forEach((value, j) => {
      const v = Math.max(-1, Math.min(1, value));
      context.fillStyle = v >= 0 ? `rgba(245, 158, 11, ${0.15 + 0.85 * v})` : `rgba(59, 130, 246, ${0.15 - 0.85 * v})`;
      context.beginPath();
      context.arc(xs[c], ys[c][j], c === layers.length - 1 ? 4.5 : 3.6, 0, Math.PI * 2);
      context.fill();
    }));
  }

  // Gait numbers, as learn_to_walk's catalogue cards give them: steps per second and feet on the ground
  paintGaitFacts() {
    const replay = this.replay;
    const end = replay.stats && replay.stats.fell ? Math.min(replay.contacts.length, Math.round(replay.stats.seconds * replay.fps)) : replay.contacts.length;
    const frames = replay.contacts.slice(0, Math.max(end, 1));
    const feet = replay.feet.length;
    const counts = new Array(feet + 1).fill(0);
    let landings = 0;
    frames.forEach((touching, index) => {
      counts[touching.reduce((sum, value) => sum + value, 0)] += 1;
      if (index) touching.forEach((value, foot) => { if (value && !frames[index - 1][foot]) landings += 1; });
    });
    const seconds = frames.length / replay.fps;
    const share = (n) => `${Math.round((100 * n) / frames.length)}%`;
    const items = [["Steps per second", seconds > 0.5 ? (landings / seconds).toFixed(1) : "-"], ["All feet in the air", share(counts[0])]];
    if (feet >= 2) items.push(["One foot down", share(counts[1])]);
    if (feet >= 2) items.push([feet === 2 ? "Both feet down" : "Two or more down", share(counts.slice(2).reduce((a, b) => a + b, 0))]);
    this.gaitFacts.replaceChildren(...items.map(([label, value]) => h("div", { class: "gait-fact" }, h("span", {}, label), h("b", {}, value))));
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
    this.drawBrain(index);
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
