// Line charts drawn as SVG, with a hover readout. series: [{name, colour, points: [[x, y], ...], dots}]

import { h, fmtSteps, fmtNumber } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

// Round numbers for the axis ticks
function niceTicks(low, high, count = 5) {
  if (low === high) { low -= 1; high += 1; }
  const span = high - low;
  const step0 = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= step0) || step0;
  const ticks = [];
  for (let v = Math.ceil(low / step) * step; v <= high + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

// The chart is drawn at its real width (so text never stretches) and redrawn when that width changes
export function lineChart(series, options = {}) {
  const wrap = h("div", { class: "chart" });
  const all = series.flatMap((s) => s.points).filter((p) => p[1] !== null && p[1] !== undefined && isFinite(p[1]));
  if (all.length < 1) {
    wrap.append(h("div", { class: "chart-empty" }, "Waiting for the first numbers..."));
    return wrap;
  }
  let drawnWidth = 0;
  const draw = () => {
    const width = Math.max(240, Math.round(wrap.clientWidth || 640));
    if (Math.abs(width - drawnWidth) < 2) return;
    drawnWidth = width;
    wrap.replaceChildren(...drawChart(series, all, width, options));
  };
  const observer = new ResizeObserver(() => { if (!wrap.isConnected) { observer.disconnect(); return; } draw(); });
  observer.observe(wrap);
  requestAnimationFrame(draw);
  return wrap;
}

function drawChart(series, all, width, { height = 220, yFormat = (v) => fmtNumber(v, Math.abs(v) < 10 ? 1 : 0), xFormat = fmtSteps,
  yMin = null, yMax = null, xLabel = "practice steps", band = null, markers = [] } = {}) {
  const pad = { left: 48, right: 14, top: 12, bottom: 28 };
  let x0 = Math.min(...all.map((p) => p[0])), x1 = Math.max(...all.map((p) => p[0]));
  const singleX = x0 === x1;
  if (singleX) x1 = x0 + 1;
  let y0 = yMin ?? Math.min(...all.map((p) => p[1])), y1 = yMax ?? Math.max(...all.map((p) => p[1]));
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const yTicks = niceTicks(y0, y1, 4);
  y0 = Math.min(y0, yTicks[0]); y1 = Math.max(y1, yTicks[yTicks.length - 1]);
  const X = (x) => pad.left + (x - x0) / (x1 - x0) * (width - pad.left - pad.right);
  const Y = (y) => pad.top + (1 - (y - y0) / (y1 - y0)) * (height - pad.top - pad.bottom);

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, style: `height:${height}px` });
  const grid = el("g", { class: "grid" });
  const axis = el("g", { class: "axis" });
  for (const tick of yTicks) {
    grid.append(el("line", { x1: pad.left, x2: width - pad.right, y1: Y(tick), y2: Y(tick) }));
    const label = el("text", { x: pad.left - 8, y: Y(tick) + 4, "text-anchor": "end" });
    label.textContent = yFormat(tick);
    axis.append(label);
  }
  for (const tick of singleX ? [x0] : niceTicks(x0, x1, Math.max(2, Math.round(width / 130)))) {
    if (tick < x0 || tick > x1) continue;
    const label = el("text", { x: X(tick), y: height - 8, "text-anchor": "middle" });
    label.textContent = xFormat(tick);
    axis.append(label);
  }
  svg.append(grid, axis);
  if (band) {
    svg.append(el("rect", { x: pad.left, width: width - pad.left - pad.right, y: Y(Math.min(band[1], y1)), height: Math.max(0, Y(Math.max(band[0], y0)) - Y(Math.min(band[1], y1))), fill: "var(--good)", opacity: 0.07 }));
  }

  // Marked moments, such as where a course moves on to its next world
  for (const [x, text] of markers) {
    if (x <= x0 || x >= x1) continue;
    svg.append(el("line", { x1: X(x), x2: X(x), y1: pad.top, y2: height - pad.bottom, stroke: "var(--accent)", "stroke-dasharray": "4 4", opacity: 0.6 }));
    const label = el("text", { x: X(x) + 4, y: pad.top + 10, style: "font-size:10.5px;fill:var(--accent)" });
    label.textContent = text;
    svg.append(label);
  }

  for (const s of series) {
    const points = s.points.filter((p) => p[1] !== null && isFinite(p[1]));
    if (!points.length) continue;
    const path = points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join(" ");
    if (s.fill) {
      const area = `${path} L${X(points[points.length - 1][0]).toFixed(1)} ${Y(y0)} L${X(points[0][0]).toFixed(1)} ${Y(y0)} Z`;
      svg.append(el("path", { d: area, fill: s.colour, opacity: 0.08 }));
    }
    svg.append(el("path", { d: path, fill: "none", stroke: s.colour, "stroke-width": s.width || 2, "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round", "stroke-dasharray": s.dash || "" }));
    if (s.dots) for (const p of points) svg.append(el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3.2, fill: "var(--surface)", stroke: s.colour, "stroke-width": 2, "vector-effect": "non-scaling-stroke" }));
  }

  const cross = el("line", { y1: pad.top, y2: height - pad.bottom, stroke: "var(--text-3)", "stroke-dasharray": "3 3", visibility: "hidden" });
  svg.append(cross);
  const tip = h("div", { class: "tip", hidden: true });
  const parts = [svg, tip];
  svg.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    if (!box.width) return;
    const x = x0 + ((event.clientX - box.left) / box.width * width - pad.left) / (width - pad.left - pad.right) * (x1 - x0);
    const lines = [];
    let nearestX = null;
    for (const s of series) {
      const points = s.points.filter((p) => p[1] !== null && isFinite(p[1]));
      if (!points.length) continue;
      const nearest = points.reduce((best, p) => (Math.abs(p[0] - x) < Math.abs(best[0] - x) ? p : best), points[0]);
      nearestX = nearestX ?? nearest[0];
      lines.push(`<div><span style="color:${s.colour}">●</span> ${s.name}: <b>${yFormat(nearest[1])}</b></div>`);
    }
    if (nearestX === null) return;
    cross.setAttribute("x1", X(nearestX)); cross.setAttribute("x2", X(nearestX));
    cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<div class="faint">${xFormat(nearestX)} ${xLabel}</div>${lines.join("")}`;
    tip.hidden = false;
    const left = (X(nearestX) / width) * box.width;
    tip.style.left = `${Math.min(left + 12, box.width - 180)}px`;
    tip.style.top = "8px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  return parts;
}

export function legend(series) {
  return h("div", { class: "legend" }, ...series.map((s) => h("span", {}, h("i", { style: { background: s.colour } }), s.name)));
}
