// Scoring at a glance: what standing still, walking, rushing and falling are worth under a world's rules,
// with warnings about the traps learn_to_walk ran into (standing still, falling on purpose).

import { h, fmtNumber } from "../ui.js";
import { icon } from "../icons.js";
import { lineChart } from "../charts.js";

const STEPS_PER_SECOND = 50;

// Rough muscle work per moment, in joules, from learn_to_walk's Duck 4 (0.135 to 0.27 J per step while walking)
const WALK_ENERGY = 0.18;
const HOLD_ENERGY = 0.03;

// The share of the speed points earned at a speed, as the simulation scores it (studio/env.py)
export function speedScore(values, speed, target) {
  if (values.speed_shape === "faster") return speed;
  if (values.speed_shape === "bell") {
    const width = Math.max(0.25 * target, 0.05);
    return target * Math.exp(-(((speed - target) / width) ** 2));
  }
  return target - Math.abs(speed - target);
}

// Points per second in a few situations
export function scenarios(values) {
  const target = (values.slowest_speed + values.fastest_speed) / 2;
  const perStep = (speed, energy, extra = 0) =>
    values.speed_points * speedScore(values, speed, target) + values.alive_points - values.energy_cost * energy + extra;
  const crouch = values.no_crouch ? -values.crouch_cost * 0.3 : 0;
  const stops = values.end_stops ? -values.end_stop_cost * 2 : 0;
  return {
    target,
    standing: perStep(0, HOLD_ENERGY) * STEPS_PER_SECOND,
    walking: perStep(target, WALK_ENERGY) * STEPS_PER_SECOND,
    crouched: perStep(target * 0.8, WALK_ENERGY * 0.7, crouch + stops) * STEPS_PER_SECOND,
    rushing: perStep(target * 2, WALK_ENERGY * 2.5) * STEPS_PER_SECOND,
    fall: -values.fall_cost,
  };
}

function warnings(values, s) {
  const notes = [];
  if (values.speed_shape === "faster") notes.push(["info", "The asked speed is ignored: it is scored only on how fast it goes."]);
  if (s.walking <= 0) notes.push(["bad", "Even good walking loses points every second. It will probably learn to fall as fast as it can (the falling trap). Lower the costs or raise the forward or staying-up points."]);
  else if (s.standing >= s.walking * 0.8) notes.push(["bad", "Standing still pays almost as well as walking. It may decide walking is not worth the risk. Lower \"Staying on its feet\" or raise \"Moving forward\"."]);
  if (s.standing < 0 && s.walking > 0) notes.push(["warn", "Standing still loses points. That pushes it to move, but early on, before it can walk, ending the attempt by falling may look attractive."]);
  if (values.end_stops && values.end_stop_cost >= 0.15) notes.push(["warn", "A strong joint limit penalty: learn_to_walk's duck preferred falling on purpose at 0.2."]);
  if (values.energy_cost >= 3) notes.push(["warn", "Energy is expensive: at 3 points per joule learn_to_walk's duck dove forward and gave up walking."]);
  if (s.rushing > s.walking && values.speed_shape !== "faster") notes.push(["warn", "Rushing at twice the asked speed scores better than walking at it. Check the speed points against the energy cost."]);
  if (!notes.length) notes.push(["good", "Balanced: walking at the asked speed clearly pays best."]);
  return notes;
}

export function scoringGlance(values) {
  const s = scenarios(values);
  const rows = [
    ["Standing still", s.standing], ["Walking at the asked speed", s.walking], ["Crouched, slower walking", s.crouched],
    ["Rushing at twice the speed", s.rushing],
  ];
  const most = Math.max(1, ...rows.map(([, value]) => Math.abs(value)));
  const bars = h("div", { class: "score-bars" }, ...rows.map(([label, value]) => h("div", { class: "score-row" },
    h("span", { class: "score-label" }, label),
    h("div", { class: "score-track" }, h("div", { class: `score-fill ${value < 0 ? "negative" : ""}`,
      style: { width: `${Math.abs(value) / most * 50}%`, [value < 0 ? "right" : "left"]: "50%" } })),
    h("span", { class: `score-value ${value < 0 ? "negative" : ""}` }, `${value > 0 ? "+" : ""}${fmtNumber(value)}`))));
  const speeds = Array.from({ length: 41 }, (_, i) => (i / 40) * s.target * 2.5);
  const curve = lineChart([{ name: "Forward points per second", colour: "var(--accent)", fill: true,
    points: speeds.map((v) => [v, values.speed_points * speedScore(values, v, s.target) * STEPS_PER_SECOND]) }],
  { height: 140, xFormat: (v) => `${fmtNumber(v, 2)}`, xLabel: "m/s", yFormat: (v) => fmtNumber(v) });
  const notes = warnings(values, s).map(([kind, text]) => h("div", { class: `callout ${kind === "info" ? "" : kind}`, html: `${icon.info}<div>${text}</div>` }));
  return h("div", { class: "card card-pad stack", style: { gap: "12px", marginTop: "14px" } },
    h("div", { class: "spread" }, h("div", {}, h("h3", {}, "Scoring at a glance"),
      h("div", { class: "faint", style: { fontSize: "12.5px" } }, `Points per second at the middle asked speed (${fmtNumber(s.target, 2)} m/s), with typical muscle work. A fall ends the attempt${s.fall ? ` and costs ${fmtNumber(-s.fall)} points` : ""}.`))),
    bars,
    h("div", {}, h("div", { class: "field-label", style: { marginBottom: "4px", fontSize: "13px" } }, "Forward points by speed"), curve),
    ...notes);
}
