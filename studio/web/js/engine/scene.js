// The scene: one built animal in one world, as a MuJoCo model. A port of studio/scene.py.
// It puts the body on the floor (a flat plane, or a height field for any other terrain), sets gravity and grip,
// and adds the world's extras: a backpack on the walker's back and a rock waiting to be thrown.

import { num, splitLines, vec } from "./kit.js";
import { idOf, modelFromXml, setStandPose, verticalReach } from "./physics.js";
import { capacity, floorLayout } from "./terrain.js";
import { plainWorld } from "./worlds.js";

export const TIMESTEP = 0.005;  // 200 physics steps per second, as in learn_to_walk
const ROCK_DENSITY = 2500.0;
const ROCK_PARKED = [-1.5, 0.95];
// The floor friction is tiny: MuJoCo takes the larger friction of two touching shapes, so the walker's own
// shapes carry the grip (their grip times the world's grip)
const FLOOR_FRICTION = "0.001 0.005 0.0001";
const START_GAP = 0.001;  // how far the feet start above the ground, in metres

export function rockRadius(mass) {
  return ((3 * mass) / (4 * Math.PI * ROCK_DENSITY)) ** (1 / 3);
}

// Multiply the sliding friction (first number) of every shape with its own friction by the grip
function scaleFriction(xml, grip) {
  return xml.replace(/friction="([0-9.eE+-]+)((?: [0-9.eE+-]+)*)"/g, (_, first, rest) => `friction="${num(parseFloat(first) * grip)}${rest}"`);
}

function assets(build, layout, most) {
  const lines = Object.entries(build.materials).map(([name, colour]) => `<material name="${name}" rgba="${colour}"/>`);
  lines.push('<material name="ground" rgba="0.55 0.72 0.5 1"/>');
  if (layout) {
    const halfX = (layout.xMax - layout.xMin) / 2;
    lines.push(`<hfield name="ground" nrow="${layout.rows}" ncol="${layout.columns}" size="${vec(halfX, layout.yHalf, Math.max(most, 1e-4), 0.05)}"/>`);
  }
  return lines.join("\n    ");
}

function floor(layout) {
  if (!layout) return `<geom name="floor" type="plane" size="0 0 0.05" material="ground" contype="1" conaffinity="1" friction="${FLOOR_FRICTION}"/>`;
  const centreX = (layout.xMax + layout.xMin) / 2;
  return `<geom name="floor" type="hfield" hfield="ground" pos="${vec(centreX, 0, 0)}" material="ground" contype="1" conaffinity="1" friction="${FLOOR_FRICTION}"/>`;
}

function rock(world, grip, most) {
  const mass = world.values.rock_weight / 1000.0;
  const radius = rockRadius(mass);
  const [x, y] = ROCK_PARKED;
  const z = most + radius + 0.01;
  return `<body name="rock" pos="${vec(x, y, z)}"><freejoint name="rock"/><geom name="rock" type="sphere" size="${num(radius)}" mass="${num(mass)}" rgba="0.45 0.45 0.45 1" contype="1" conaffinity="1" friction="${num(grip)} 0.05 0.01"/></body>`;
}

// The backpack: a box fixed on top of the torso, scaled with the animal
function backpack(world, build, torsoTop) {
  const s = build.size;
  const half = [0.04 * s, 0.035 * s, 0.02 * s];
  const mass = world.values.backpack_weight / 1000.0;
  return `<geom name="backpack" type="box" size="${vec(...half)}" pos="${vec(-0.03 * s, 0, torsoTop + half[2])}" mass="${num(mass)}" rgba="0.2 0.4 0.75 1"/>`;
}

// The whole MuJoCo file
export function sceneXml(build, world, layout, most, torsoTop = 0.0) {
  const grip = world.grip;
  let body = scaleFriction(build.bodyXml, grip);
  if (world.values.backpack) body = body.replace('<freejoint name="root"/>', '<freejoint name="root"/>\n  ' + backpack(world, build, torsoTop));
  const rockXml = world.values.rocks ? rock(world, grip, most) : "";
  const bodyLines = splitLines(body).map((line) => "    " + line).join("\n");
  const actuators = splitLines(build.actuatorXml).map((line) => "    " + line).join("\n");
  return `<mujoco model="${build.animal}">
  <compiler angle="radian"/>
  <option timestep="${TIMESTEP}" integrator="implicitfast" gravity="0 0 ${num(-world.values.gravity)}"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    ${assets(build, layout, most)}
  </asset>
  <default>
    <joint damping="${num(build.jointDamping)}" armature="${num(build.jointArmature)}"/>
    <geom contype="1" conaffinity="0" friction="${num(grip)} 0.05 0.01"/>
  </default>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1" directional="true"/>
    ${floor(layout)}
${bodyLines}
    ${rockXml}
  </worldbody>
  <actuator>
${actuators}
  </actuator>
</mujoco>
`;
}

// Put the animal in its standing pose with the torso at height 0, and work out where its parts reach
function measure(mujoco, build) {
  const model = modelFromXml(mujoco, sceneXml(build, plainWorld("measure"), null, 0.0));
  const data = new mujoco.MjData(model);
  try {
    setStandPose(mujoco, model, data, build, 0.0);
    mujoco.mj_forward(model, data);
    const torso = idOf(mujoco, model, "body", "torso");
    let lowest = Infinity, top = -Infinity;
    const bodyOf = model.geom_bodyid, contype = model.geom_contype;
    for (let geomId = 0; geomId < model.ngeom; geomId++) {
      if (bodyOf[geomId] === 0) continue;
      const [low, high] = verticalReach(model, data, geomId);
      if (contype[geomId]) lowest = Math.min(lowest, low);
      if (bodyOf[geomId] === torso) top = Math.max(top, high);
    }
    return [-lowest + START_GAP, top];
  } finally {
    data.delete();
    model.delete();
  }
}

// The scene of an animal in a world. The caller owns scene.model and must call disposeScene() when done
export function buildScene(mujoco, build, world) {
  const [standHeight, torsoTop] = measure(mujoco, build);
  const terrain = world.terrain;
  let layout = null, most = 0.0;
  if (terrain !== "flat") {
    layout = floorLayout(terrain, world.speedRange[1], world.values.attempt_seconds, build.size);
    most = capacity(terrain, world.tallest);
  }
  const xml = sceneXml(build, world, layout, most, torsoTop);
  const model = modelFromXml(mujoco, xml);
  const totalMass = model.body_subtreemass[idOf(mujoco, model, "body", "torso")];
  return { model, build, world, layout, floorCapacity: most, standHeight, torsoTop, totalMass, xml };
}

export function disposeScene(scene) {
  if (scene && scene.model) { scene.model.delete(); scene.model = null; }
}
