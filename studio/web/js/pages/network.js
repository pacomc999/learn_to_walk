// The brain drawn as a network: senses on the left, layers of thinking cells, motors on the right.

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text = null) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== null) node.textContent = text;
  return node;
}

// The senses in groups, as the environment orders them
function senseGroups(info, design) {
  const motors = info.outputs;
  const groups = [
    ["Height and drift", 2], ["Balance (tilt)", 4], ["Body motion", 6], ["Joint angles", motors], ["Joint speeds", motors],
  ];
  const feet = info.inputs - 13 - 2 * motors - (design.values.see_ground ? 7 : 0);
  if (design.values.feel_feet && feet > 0) groups.push(["Feet touching", feet]);
  if (design.values.see_ground) groups.push(["Ground ahead", 7]);
  groups.push(["Asked speed", 1]);
  return groups;
}

function prettyMotor(name) {
  return name.replace(/_/g, " ");
}

export function networkDiagram(info, design) {
  const wrap = document.createElement("div");
  wrap.className = "network";
  const inputs = senseGroups(info, design);
  const layers = design.layers;
  const outputs = info.motor_names;
  const height = Math.max(340, Math.max(inputs.length, outputs.length) * 24 + 70);
  const columns = 2 + layers.length;
  const width = Math.max(640, 170 + columns * 110 + 120);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", style: `max-width:${width}px` });
  const left = 150, right = width - 140;
  const xs = Array.from({ length: columns }, (_, i) => left + (right - left) * i / (columns - 1));
  const top = 46, bottom = height - 20;
  const spread = (count) => Array.from({ length: count }, (_, i) => (count === 1 ? (top + bottom) / 2 : top + (bottom - top) * i / (count - 1)));

  // How many circles to draw for each column (big layers show a sample and a count)
  const inputYs = spread(inputs.length);
  const layerYs = layers.map((size) => spread(Math.min(size, 12)));
  const outputYs = spread(outputs.length);
  const columnsYs = [inputYs, ...layerYs, outputYs];

  const edges = el("g");
  for (let c = 0; c < columnsYs.length - 1; c++) {
    for (const y1 of columnsYs[c]) for (const y2 of columnsYs[c + 1]) {
      edges.append(el("line", { class: "edge", x1: xs[c], y1, x2: xs[c + 1], y2 }));
    }
  }
  svg.append(edges);

  // Titles
  const titles = ["Senses", ...layers.map((_, i) => `Layer ${i + 1}`), "Motors"];
  titles.forEach((title, c) => svg.append(el("text", { class: "layer-title", x: xs[c], y: 20, "text-anchor": "middle" }, title)));
  layers.forEach((size, i) => svg.append(el("text", { x: xs[i + 1], y: 36, "text-anchor": "middle" }, `${size} cells`)));

  inputs.forEach(([name, count], i) => {
    svg.append(el("circle", { class: "node input", cx: xs[0], cy: inputYs[i], r: 6 }));
    svg.append(el("text", { x: xs[0] - 12, y: inputYs[i] + 4, "text-anchor": "end" }, `${name} (${count})`));
  });
  layerYs.forEach((ys, l) => {
    ys.forEach((y, i) => {
      const gap = layers[l] > 12 && i === 5;
      if (gap) svg.append(el("text", { x: xs[l + 1], y: y + 4, "text-anchor": "middle" }, "⋮"));
      else svg.append(el("circle", { class: "node", cx: xs[l + 1], cy: y, r: 5 }));
    });
  });
  outputs.forEach((name, i) => {
    svg.append(el("circle", { class: "node output", cx: xs[columns - 1], cy: outputYs[i], r: 5.5 }));
    svg.append(el("text", { x: xs[columns - 1] + 12, y: outputYs[i] + 4 }, prettyMotor(name)));
  });
  if (design.values.symmetric) {
    const badge = el("g", { transform: `translate(${(xs[0] + xs[1]) / 2 - 46}, ${height - 18})` });
    badge.append(el("rect", { width: 92, height: 18, rx: 9, fill: "var(--good-soft)" }));
    badge.append(el("text", { x: 46, y: 13, "text-anchor": "middle", fill: "var(--good)", style: "font-weight:600;font-size:11px" }, "left/right mirrored"));
    svg.append(badge);
  }
  wrap.append(svg);
  return wrap;
}
