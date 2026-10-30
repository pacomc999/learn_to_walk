"""Worlds: everything around the duck and everything asked of him.

A world is a small JSON file in worlds/: the floor (how tall the stones get), the task
(the speeds asked for and the reward weights) and extras that happen to the duck
(thrown rocks, a backpack to carry). A duck keeps a frozen copy of its world in its
duck.json, so editing a world file later never changes an old duck.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from duck.floor import MAX_ROUGHNESS
from duck.rewards import RewardWeights

# Where the world files live (the worlds folder at the project root)
WORLDS_DIR = Path(__file__).parent.parent / "worlds"

# The keys a world file may have, and the extras it may switch on
WORLD_KEYS = {"name", "description", "floor", "target_speed_range", "reward", "extras", "notes"}
EXTRA_NAMES = {"rocks", "backpack"}
REWARD_KEYS = ["forward", "alive", "energy", "sideways"]


# A stone ball thrown at the duck: its mass in kg, how often (seconds) and how fast (m/s)
@dataclass(frozen=True)
class Rocks:
    mass: float
    every_seconds: float = 2.0
    speed: float = 4.0


# A box fixed on the duck's back: its mass in kg
@dataclass(frozen=True)
class Backpack:
    mass: float


# One world, read from its file
@dataclass(frozen=True)
class World:
    name: str
    description: str
    tallest_stones: float  # metres; 0 is the plain flat plane
    target_speed_range: tuple[float, float]
    reward: RewardWeights
    rocks: Rocks | None = None
    backpack: Backpack | None = None
    notes: str = ""


# ---------- Checking a file ----------

# Stop with a plain message that names the file and the key
def _refuse(source: str, key: str, problem: str) -> None:
    raise ValueError(f"{source}: {key} {problem}")


# A number above 0, or a plain refusal
def _positive(value, source: str, key: str) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        _refuse(source, key, f"must be a number above 0, not {value!r}")
    return float(value)


# The floor section: the tallest stones, from 0 up to what the floor model holds
def _tallest_stones(floor: dict, source: str) -> float:
    unknown = set(floor) - {"tallest_stones"}
    if unknown:
        _refuse(source, f"floor.{sorted(unknown)[0]}", "is not something a floor can have")
    tallest = floor.get("tallest_stones", 0.0)
    if not isinstance(tallest, (int, float)) or not 0 <= tallest <= MAX_ROUGHNESS:
        _refuse(source, "floor.tallest_stones", f"must be from 0 to {MAX_ROUGHNESS} metres, not {tallest!r}")
    return float(tallest)


# The speed range: two numbers above 0, the slowest first
def _speed_range(value, source: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        _refuse(source, "target_speed_range", "must be two speeds, for example [0.2, 0.7]")
    low = _positive(value[0], source, "target_speed_range")
    high = _positive(value[1], source, "target_speed_range")
    if low > high:
        _refuse(source, "target_speed_range", f"must have the slowest speed first, not {value!r}")
    return low, high


# The four reward weights, all present
def _reward(value: dict, source: str) -> RewardWeights:
    missing = [key for key in REWARD_KEYS if key not in value]
    if missing or set(value) - set(REWARD_KEYS):
        _refuse(source, "reward", f"must have exactly the weights {', '.join(REWARD_KEYS)}")
    return RewardWeights(**{key: float(value[key]) for key in REWARD_KEYS})


# The extras section: rocks and a backpack, each optional
def _extras(extras: dict, source: str) -> tuple[Rocks | None, Backpack | None]:
    unknown = set(extras) - EXTRA_NAMES
    if unknown:
        _refuse(source, f"extras.{sorted(unknown)[0]}", f"is not a known extra ({', '.join(sorted(EXTRA_NAMES))})")
    rocks = None
    if "rocks" in extras:
        settings = extras["rocks"]
        rocks = Rocks(mass=_positive(settings.get("mass"), source, "extras.rocks.mass"),
                      every_seconds=_positive(settings.get("every_seconds", 2.0), source, "extras.rocks.every_seconds"),
                      speed=_positive(settings.get("speed", 4.0), source, "extras.rocks.speed"))
    backpack = None
    if "backpack" in extras:
        backpack = Backpack(mass=_positive(extras["backpack"].get("mass"), source, "extras.backpack.mass"))
    return rocks, backpack


# ---------- Reading and writing ----------

# A world from the contents of its file, checked; source names the file in messages
def world_from_dict(data: dict, source: str = "world") -> World:
    unknown = set(data) - WORLD_KEYS
    if unknown:
        _refuse(source, sorted(unknown)[0], "is not something a world can have")
    for key in ("name", "description", "target_speed_range", "reward"):
        if key not in data:
            _refuse(source, key, "is missing")
    rocks, backpack = _extras(data.get("extras", {}), source)
    return World(
        name=data["name"],
        description=data["description"],
        tallest_stones=_tallest_stones(data.get("floor", {}), source),
        target_speed_range=_speed_range(data["target_speed_range"], source),
        reward=_reward(data["reward"], source),
        rocks=rocks,
        backpack=backpack,
        notes=data.get("notes", ""),
    )


# The file layout of a world: floor and extras as sections
def world_to_dict(world: World) -> dict:
    extras = {}
    if world.rocks:
        extras["rocks"] = {"mass": world.rocks.mass, "every_seconds": world.rocks.every_seconds,
                           "speed": world.rocks.speed}
    if world.backpack:
        extras["backpack"] = {"mass": world.backpack.mass}
    return {
        "name": world.name,
        "description": world.description,
        "floor": {"tallest_stones": world.tallest_stones},
        "target_speed_range": list(world.target_speed_range),
        "reward": {key: getattr(world.reward, key) for key in REWARD_KEYS},
        "extras": extras,
        "notes": world.notes,
    }


# Read the world file with this name from the worlds folder
def load_world(name: str, folder: Path = WORLDS_DIR) -> World:
    path = folder / f"{name}.json"
    if not path.exists():
        raise ValueError(f"There is no world called {name} (looked for {path.as_posix()}).")
    return world_from_dict(json.loads(path.read_text()), source=path.as_posix())
