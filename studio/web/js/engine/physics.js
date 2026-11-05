// MuJoCo in the browser: loading the WebAssembly engine once, and small helpers around its arrays.
// The engine is the official @mujoco/mujoco build (vendor/mujoco), the same MuJoCo version as the Python studio.
// Model and data arrays (qpos, xpos, geom_size, ...) are live views into the engine's memory.

let loading = null;

// The MuJoCo module, loaded the first time it is needed
export function loadPhysics() {
  if (!loading) {
    loading = import("../../vendor/mujoco/mujoco.js").then((module) => module.default());
  }
  return loading;
}

// ---------- Names and numbers ----------

export const OBJ = { body: 1, joint: 3, geom: 5, actuator: 19 };  // mjtObj values used by the studio
export const GEOM = { plane: 0, hfield: 1, sphere: 2, capsule: 3, ellipsoid: 4, cylinder: 5, box: 6 };
export const HINGE = 3;
export const FREE = 0;
export const BIAS_NONE = 0;

// The number of a named body, joint, geom or actuator (-1 if there is none)
export function idOf(mujoco, model, kind, name) {
  return mujoco.mj_name2id(model, OBJ[kind], name);
}

// The name of a numbered body, joint, geom or actuator ("" if it has none)
export function nameOf(mujoco, model, kind, id) {
  return mujoco.mj_id2name(model, OBJ[kind], id) || "";
}

// A model from XML text, with a plain message if MuJoCo refuses it
export function modelFromXml(mujoco, xml) {
  try {
    return mujoco.MjModel.from_xml_string(xml);
  } catch (error) {
    const detail = typeof error === "number" && mujoco.getExceptionMessage ? mujoco.getExceptionMessage(error) : String(error?.message || error);
    throw new Error(`MuJoCo could not build this creature: ${detail}`);
  }
}

// ---------- The standing pose ----------

// The torso upright at `height`, every joint at its standing angle; springy motors aim for that angle
export function setStandPose(mujoco, model, data, build, height) {
  mujoco.mj_resetData(model, data);
  const qpos = data.qpos;
  qpos.set([0, 0, height, 1, 0, 0, 0], 0);
  const qposAdr = model.jnt_qposadr;
  for (const [name, angle] of Object.entries(build.standPose)) qpos[qposAdr[idOf(mujoco, model, "joint", name)]] = angle;
  const trnid = model.actuator_trnid, biastype = model.actuator_biastype, ctrl = data.ctrl;
  for (let motor = 0; motor < model.nu; motor++) {
    if (biastype[motor] !== BIAS_NONE) ctrl[motor] = qpos[qposAdr[trnid[motor * 2]]];  // position motors hold the pose
  }
}

// The lowest and highest point of one shape, from its centre, size and turn (a port of scene._vertical_reach)
export function verticalReach(model, data, geomId) {
  const centreZ = data.geom_xpos[geomId * 3 + 2];
  const mat = data.geom_xmat;
  const up = [mat[geomId * 9 + 6], mat[geomId * 9 + 7], mat[geomId * 9 + 8]];  // how much each own axis points up
  const size = model.geom_size;
  const s = [size[geomId * 3], size[geomId * 3 + 1], size[geomId * 3 + 2]];
  const kind = model.geom_type[geomId];
  let reach;
  if (kind === GEOM.sphere) reach = s[0];
  else if (kind === GEOM.capsule) reach = Math.abs(up[2]) * s[1] + s[0];
  else if (kind === GEOM.cylinder) reach = Math.abs(up[2]) * s[1] + s[0] * Math.sqrt(Math.max(0, 1 - up[2] ** 2));
  else if (kind === GEOM.ellipsoid) reach = Math.sqrt((up[0] * s[0]) ** 2 + (up[1] * s[1]) ** 2 + (up[2] * s[2]) ** 2);
  else reach = Math.abs(up[0]) * s[0] + Math.abs(up[1]) * s[1] + Math.abs(up[2]) * s[2];  // box
  return [centreZ - reach, centreZ + reach];
}
