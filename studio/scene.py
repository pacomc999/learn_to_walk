"""The scene: one built animal in one world, as a MuJoCo model.

It puts the animal's body on the floor (a flat plane, or a height field for any other
terrain), sets gravity and grip, and adds the world's extras: a backpack on the walker's
back and a rock waiting to be thrown.
"""
import re
from dataclasses import dataclass

import mujoco
import numpy as np

from studio.creatures.kit import Build, num, vec
from studio.terrain import FloorLayout, capacity, floor_layout
from studio.worlds import DEFAULTS, World

# Simulation: 200 physics steps per second, as in learn_to_walk
TIMESTEP = 0.005

# The rock: a stone ball, parked off to the side between throws
ROCK_DENSITY = 2500.0
ROCK_PARKED = (-1.5, 0.95)

# The floor friction is tiny: MuJoCo takes the larger friction of two touching shapes, so the
# walker's own shapes carry the grip (their grip times the world's grip)
FLOOR_FRICTION = "0.001 0.005 0.0001"

# How far the feet start above the ground, in metres
START_GAP = 0.001


# A scene ready to simulate, with the facts the environment needs
@dataclass
class Scene:
    model: mujoco.MjModel
    build: Build
    world: World
    layout: FloorLayout | None          # None on the flat plane
    floor_capacity: float               # the tallest ground the floor model can hold
    stand_height: float                 # torso height above the ground when standing at the start
    torso_top: float                    # height of the torso's top above the torso centre
    total_mass: float


# The radius of a stone ball of this mass, in metres
def rock_radius(mass: float) -> float:
    return (3 * mass / (4 * np.pi * ROCK_DENSITY)) ** (1 / 3)


# Multiply the sliding friction (first number) of every shape with its own friction by the grip
def _scale_friction(xml: str, grip: float) -> str:
    def scaled(match):
        first, rest = match.group(1), match.group(2)
        return f'friction="{num(float(first) * grip)}{rest}"'
    return re.sub(r'friction="([0-9.eE+-]+)((?: [0-9.eE+-]+)*)"', scaled, xml)


# The materials of the scene: the animal's colours and the floor
def _assets(build: Build, layout: FloorLayout | None, most: float) -> str:
    lines = [f'<material name="{name}" rgba="{colour}"/>' for name, colour in build.materials.items()]
    lines.append('<material name="ground" rgba="0.55 0.72 0.5 1"/>')
    if layout is not None:
        half_x = (layout.x_max - layout.x_min) / 2
        lines.append(f'<hfield name="ground" nrow="{layout.rows}" ncol="{layout.columns}" '
                     f'size="{vec(half_x, layout.y_half, max(most, 1e-4), 0.05)}"/>')
    return "\n    ".join(lines)


def _floor(layout: FloorLayout | None) -> str:
    if layout is None:
        return (f'<geom name="floor" type="plane" size="0 0 0.05" material="ground" contype="1" conaffinity="1" '
                f'friction="{FLOOR_FRICTION}"/>')
    centre_x = (layout.x_max + layout.x_min) / 2
    return (f'<geom name="floor" type="hfield" hfield="ground" pos="{vec(centre_x, 0, 0)}" material="ground" '
            f'contype="1" conaffinity="1" friction="{FLOOR_FRICTION}"/>')


def _rock(world: World, grip: float, most: float) -> str:
    mass = world.values["rock_weight"] / 1000.0
    radius = rock_radius(mass)
    x, y = ROCK_PARKED
    z = most + radius + 0.01
    return (f'<body name="rock" pos="{vec(x, y, z)}"><freejoint name="rock"/>'
            f'<geom name="rock" type="sphere" size="{num(radius)}" mass="{num(mass)}" rgba="0.45 0.45 0.45 1" '
            f'contype="1" conaffinity="1" friction="{num(grip)} 0.05 0.01"/></body>')


# The backpack: a box fixed on top of the torso, scaled with the animal
def _backpack(world: World, build: Build, torso_top: float) -> str:
    s = build.size
    half = (0.04 * s, 0.035 * s, 0.02 * s)
    mass = world.values["backpack_weight"] / 1000.0
    return (f'<geom name="backpack" type="box" size="{vec(*half)}" pos="{vec(-0.03 * s, 0, torso_top + half[2])}" '
            f'mass="{num(mass)}" rgba="0.2 0.4 0.75 1"/>')


# The whole MuJoCo file
def scene_xml(build: Build, world: World, layout: FloorLayout | None, most: float, torso_top: float = 0.0) -> str:
    grip = world.grip
    body = _scale_friction(build.body_xml, grip)
    if world.values["backpack"]:
        body = body.replace('<freejoint name="root"/>',
                            '<freejoint name="root"/>\n  ' + _backpack(world, build, torso_top), 1)
    rock = _rock(world, grip, most) if world.values["rocks"] else ""
    body_lines = "\n".join("    " + line for line in body.splitlines())
    actuators = "\n".join("    " + line for line in build.actuator_xml.splitlines())
    return f"""<mujoco model="{build.animal}">
  <compiler angle="radian"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" gravity="0 0 {num(-world.values['gravity'])}"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    {_assets(build, layout, most)}
  </asset>
  <default>
    <joint damping="{num(build.joint_damping)}" armature="{num(build.joint_armature)}"/>
    <geom contype="1" conaffinity="0" friction="{num(grip)} 0.05 0.01"/>
  </default>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1" directional="true"/>
    {_floor(layout)}
{body_lines}
    {rock}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


# Put the animal in its standing pose with the torso at height 0, and work out where its parts reach
def _measure(build: Build) -> tuple[float, float]:
    plain = World("measure", "measure", "", dict(DEFAULTS))
    model = mujoco.MjModel.from_xml_string(scene_xml(build, plain, None, 0.0))
    data = mujoco.MjData(model)
    set_stand_pose(model, data, build, 0.0)
    mujoco.mj_forward(model, data)
    torso = model.body("torso").id
    lowest, top = np.inf, -np.inf
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == 0:
            continue
        low, high = _vertical_reach(model, data, geom_id)
        if model.geom_contype[geom_id]:
            lowest = min(lowest, low)
        if model.geom_bodyid[geom_id] == torso:
            top = max(top, high)
    return -lowest + START_GAP, top


# The lowest and highest point of one shape, from its centre, size and turn
def _vertical_reach(model, data, geom_id: int) -> tuple[float, float]:
    centre = data.geom_xpos[geom_id]
    turn = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    kind = int(model.geom_type[geom_id])
    up = turn[2]  # how much each of the shape's own axes points up
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        reach = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        reach = abs(up[2]) * size[1] + size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        reach = abs(up[2]) * size[1] + size[0] * np.sqrt(max(0.0, 1 - up[2] ** 2))
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        reach = float(np.sqrt(np.sum((up * size) ** 2)))
    else:  # box
        reach = float(np.sum(np.abs(up) * size))
    return centre[2] - reach, centre[2] + reach


# The standing pose: torso upright at `height`, every joint at its standing angle
def set_stand_pose(model, data, build: Build, height: float) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:7] = [0, 0, height, 1, 0, 0, 0]
    for name, angle in build.stand_pose.items():
        data.qpos[model.joint(name).qposadr[0]] = angle
    for motor_id in range(model.nu):
        joint = model.actuator_trnid[motor_id][0]
        if model.actuator_biastype[motor_id] != mujoco.mjtBias.mjBIAS_NONE:  # position motors hold the pose
            data.ctrl[motor_id] = data.qpos[model.jnt_qposadr[joint]]


# The scene of an animal in a world
def build_scene(build: Build, world: World) -> Scene:
    stand_height, torso_top = _measure(build)
    terrain = world.terrain
    layout = None
    most = 0.0
    if terrain != "flat":
        layout = floor_layout(terrain, world.speed_range[1], world.values["attempt_seconds"], build.size)
        most = capacity(terrain, world.tallest)
    model = mujoco.MjModel.from_xml_string(scene_xml(build, world, layout, most, torso_top))
    total_mass = float(model.body_subtreemass[model.body("torso").id])
    return Scene(model, build, world, layout, most, stand_height, torso_top, total_mass)
