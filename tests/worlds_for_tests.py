# Worlds and the body most tests use: flat ground, stones up to 4 cm, and the standard duck
from duck.bodies import STANDARD_BODY
from duck.worlds import load_world

FLAT = load_world("flat_walking")
STONES = load_world("stones_up_to_4cm")
BODY = STANDARD_BODY
