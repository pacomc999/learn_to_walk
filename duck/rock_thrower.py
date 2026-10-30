"""A rock thrown at the duck: the stone ball in the model and the rules for throwing it.

Used by the environment when a world has rocks (world extras) and by the rocks challenge.
A rock is a real MuJoCo body (a stone ball with its own free joint) added after the duck,
so it pushes the duck by actually hitting it. The duck senses only its own joints, so the
rock never shows up in its 39 senses.
"""
import numpy as np

from duck.worlds import Rocks

# The rocks: stone balls (about granite) thrown from 0.8 m to the side
ROCK_DENSITY = 2500.0      # kg per cubic metre
THROW_DISTANCE = 0.8       # metres
GRAVITY = 9.81             # metres per second squared, for the arc of the throw

# The first rock flies at 2 s; each is taken away again after 1 s,
# so a rock never lies on the floor as an obstacle
FIRST_THROW = 2.0
ROCK_LIFETIME = 1.0

# Where the rock waits between throws: off the walking line, behind the start, resting on the
# flat plane. On the stone floor it waits higher, clear of the tallest stone
PARKED_POSITION = (-1.5, 0.95, 0.05)
STONE_FLOOR_GAP = 0.01  # metres between the tallest stone and the bottom of the waiting rock


# The radius of a stone ball of this mass, in metres
def rock_radius(mass: float) -> float:
    return (3 * mass / (4 * np.pi * ROCK_DENSITY)) ** (1 / 3)


# Where the rock waits above a floor whose stones reach floor_height (0 on the flat plane)
def parked_position(mass: float, floor_height: float = 0.0) -> tuple[float, float, float]:
    x, y, z = PARKED_POSITION
    if floor_height > 0:
        z = floor_height + rock_radius(mass) + STONE_FLOOR_GAP
    return x, y, z


# The duck XML with a grey stone ball after the duck, waiting in the standing keyframe
def add_rock(xml: str, mass: float, floor_height: float = 0.0) -> str:
    x, y, z = parked_position(mass, floor_height)
    rock = (f'    <body name="rock" pos="{x} {y} {z}"><freejoint name="rock"/>'
            f'<geom name="rock" type="sphere" size="{rock_radius(mass):.4f}" mass="{mass}" '
            f'rgba="0.45 0.45 0.45 1" contype="1" conaffinity="1"/></body>\n  </worldbody>')
    xml = xml.replace("  </worldbody>", rock, 1)
    key_start = xml.index('<key name="stand" qpos="') + len('<key name="stand" qpos="')
    key_end = xml.index('"', key_start)
    return xml[:key_end] + f"  {x} {y} {z} 1 0 0 0" + xml[key_end:]


# Throws the rock at the duck on a schedule, counts hits and takes the rock away after each throw
class RockThrower:
    def __init__(self, model, rocks: Rocks, floor_height: float = 0.0):
        self.rocks = rocks
        self.parked = parked_position(rocks.mass, floor_height)
        self.qpos_start = model.joint("rock").qposadr[0]
        self.qvel_start = model.joint("rock").dofadr[0]
        self.rock_geom = model.geom("rock").id
        self.floor_geom = model.geom("floor").id
        self.rng = np.random.default_rng(0)
        self.next_throw = FIRST_THROW
        self.thrown_at = None
        self.hit_this_throw = False
        self.throws = 0
        self.hits = 0

    # A new attempt: no rocks thrown yet; rng picks the side of every throw
    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.next_throw = FIRST_THROW
        self.thrown_at = None
        self.hit_this_throw = False
        self.throws = 0
        self.hits = 0

    # Throw the rock at the body from one side, aiming where the body will be and arcing for gravity
    def _throw(self, data) -> None:
        side = 1.0 if self.rng.random() < 0.5 else -1.0
        flight_seconds = THROW_DISTANCE / self.rocks.speed
        target = data.body("torso").subtree_com + data.qvel[0:3] * flight_seconds
        start = target + np.array([0.0, side * THROW_DISTANCE, 0.0])
        velocity = (target - start) / flight_seconds
        velocity[2] += 0.5 * GRAVITY * flight_seconds
        data.qpos[self.qpos_start:self.qpos_start + 7] = [*start, 1, 0, 0, 0]
        data.qvel[self.qvel_start:self.qvel_start + 6] = [*velocity, 0, 0, 0]
        self.thrown_at = data.time
        self.hit_this_throw = False
        self.throws += 1

    # Put the rock back where it waits
    def _park(self, data) -> None:
        data.qpos[self.qpos_start:self.qpos_start + 7] = [*self.parked, 1, 0, 0, 0]
        data.qvel[self.qvel_start:self.qvel_start + 6] = 0.0
        self.thrown_at = None

    # Whether the rock touches any part of the duck right now (anything but the floor)
    def _rock_touches_duck(self, data) -> bool:
        for contact in data.contact[:data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if self.rock_geom in pair and self.floor_geom not in pair:
                return True
        return False

    # Before a control step: throw when it is time
    def before_step(self, data) -> None:
        if data.time >= self.next_throw:
            self._throw(data)
            self.next_throw += self.rocks.every_seconds

    # After a control step: count a hit once per throw, and take the rock away when its time is up
    def after_step(self, data) -> None:
        if self.thrown_at is None:
            return
        if not self.hit_this_throw and self._rock_touches_duck(data):
            self.hit_this_throw = True
            self.hits += 1
        if data.time - self.thrown_at >= ROCK_LIFETIME:
            self._park(data)
