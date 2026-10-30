"""The scene: the MuJoCo model of one body in one world.

It starts from the body's XML, swaps the flat plane for the cobblestone height field when
the world has stones, then adds the world's extras: a backpack on the duck's back and a
rock waiting to be thrown.
"""
import re

import mujoco

from duck.bodies import Body, body_xml
from duck.floor import MAX_ROUGHNESS, floor_xml
from duck.rock_thrower import add_rock
from duck.worlds import World

# The backpack: an 8 x 7 x 4 cm box on the duck's back (half sizes, in metres), a little behind the middle
BACKPACK_SIZE = "0.04 0.035 0.02"
BACKPACK_POSITION = "-0.04 0 0.1"


# The torso's free joint, written either way (a changed body is written back with a space before "/>")
TORSO_JOINT = re.compile(r'<freejoint name="root"\s*/>')


# The duck XML with a backpack box fixed to the torso (no joint, so it moves with the body)
def add_backpack(xml: str, mass: float) -> str:
    backpack = (f'<freejoint name="root"/>\n      <geom name="backpack" type="box" size="{BACKPACK_SIZE}" '
                f'pos="{BACKPACK_POSITION}" mass="{mass}" rgba="0.2 0.4 0.75 1"/>')
    return TORSO_JOINT.sub(backpack, xml, count=1)


# The XML of a body in a world: on the flat plane or on a height field floor up to `most` metres
def scene_xml(world: World, body: Body, flat: bool, most: float = MAX_ROUGHNESS) -> str:
    xml = body_xml(body)
    if not flat:
        xml = floor_xml(xml, most)
    if world.backpack:
        xml = add_backpack(xml, world.backpack.mass)
    if world.rocks:
        # On the stone floor the waiting rock floats just above the tallest possible stone
        xml = add_rock(xml, world.rocks.mass, floor_height=0.0 if flat else most)
    return xml


# The MuJoCo model of a body in a world
def load_scene(world: World, body: Body, flat: bool, most: float = MAX_ROUGHNESS) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(scene_xml(world, body, flat, most))
