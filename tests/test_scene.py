from dataclasses import replace

import mujoco
import numpy as np
import pytest

from duck.bodies import body_from_dict
from duck.duck_env import DuckWalkEnv
from duck.model import DUCK_XML_PATH
from duck.scene import load_scene, scene_xml
from duck.worlds import Backpack, Rocks
from tests.worlds_for_tests import BODY, FLAT, STONES


# The names of the shapes touching each other right now, as pairs
def contact_geom_names(env: DuckWalkEnv) -> list[tuple[str, str]]:
    return [(env.model.geom(contact.geom1).name, env.model.geom(contact.geom2).name)
            for contact in env.data.contact[:env.data.ncon]]


# The standard body on flat ground with no extras is exactly today's model
def test_standard_flat_scene_is_todays_model():
    assert scene_xml(FLAT, BODY, flat=True) == DUCK_XML_PATH.read_text()


# Every combination loads: a changed body on stones with both extras
def test_changed_body_on_stones_with_extras_loads():
    body = body_from_dict({"name": "strong_legs", "description": "x", "motor_force_limits": {"hips": 8}})
    world = replace(STONES, rocks=Rocks(0.1), backpack=Backpack(0.3))
    model = load_scene(world, body, flat=False)
    assert model.nhfield == 1 and model.body("rock") is not None
    assert list(model.actuator("left_hip_swing").forcerange) == [-8, 8]
    assert model.geom("backpack").size[0] == pytest.approx(0.04)


# On the stone floor the waiting rock starts above the tallest stones (then settles on them, as it settles
# on the flat plane) and never touches the duck before the first throw
def test_rock_parks_clear_of_the_tallest_stones():
    world = replace(STONES, rocks=Rocks(0.4))
    env = DuckWalkEnv(world=world, body=BODY, roughness=0.04)
    env.reset(seed=0)
    assert env.model.body("rock").pos[2] > 0.04  # starts above the tallest stone
    for _ in range(50):  # 1 s, before the first throw at 2 s
        env.step(np.zeros(12))
        assert not any("rock" in pair and "floor" not in pair for pair in contact_geom_names(env))
    assert env.rock_thrower.hits == 0
