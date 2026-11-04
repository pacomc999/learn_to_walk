"""Worlds: the ground, the physics and the task a walker trains in.

A world is a set of values for the settings below, saved as a small JSON file. The
built-in worlds live in studio/world_templates/ and are read only; worlds made on the
World screen are saved in the library folder. Every world uses the same settings, so the
web page can draw the World screen from SETTING_GROUPS alone.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from studio.creatures.kit import Setting, SettingGroup, clean_values, default_values, percent, switch

TEMPLATES_DIR = Path(__file__).parent / "world_templates"

TERRAINS = (
    ("flat", "Flat ground"),
    ("cobblestones", "Cobblestones"),
    ("hills", "Rolling hills"),
    ("stairs", "Stairs"),
    ("craters", "Craters"),
    ("blocks", "Rocky blocks"),
)

STYLES = (
    ("meadow", "Green meadow"),
    ("street", "Stone street"),
    ("desert", "Sandy desert"),
    ("snow", "Snow and ice"),
    ("moon", "Moon dust"),
    ("night", "Night"),
)


def number(key, label, help, default, minimum, maximum, step, unit) -> Setting:
    return Setting(key, label, "number", default, help, minimum, maximum, step, unit)


SETTING_GROUPS = (
    SettingGroup("ground", "Ground", "What the walker stands on.", (
        Setting("terrain", "Kind of ground", "choice", "flat", "The shape of the ground.", choices=TERRAINS),
        number("terrain_height", "Tallest bumps", "How tall the bumps, stones, hills or steps get. 0 is flat.",
               3.0, 0.0, 40.0, 0.5, "cm"),
        switch("mixed_heights", "Mix of heights", "Each practice attempt picks a height from flat up to the "
               "tallest, so the walker meets easy and hard ground. Off: always the tallest."),
        switch("gradually_harder", "Gradually harder", "Start on flat ground and raise the bumps step by step "
               "as training goes on. Helps with hard ground.", default=False),
        percent("ground_grip", "Ground grip", "How slippery the ground is. Ice is about 15%.", minimum=5,
                maximum=200),
    )),
    SettingGroup("physics", "Physics", "The laws of nature in this world.", (
        number("gravity", "Gravity", "Earth 9.8, Mars 3.7, Moon 1.6. Low gravity makes every step a jump.",
               9.81, 1.0, 25.0, 0.1, "m/s²"),
    )),
    SettingGroup("task", "The task", "What the walker is asked to do in every attempt.", (
        number("slowest_speed", "Slowest asked speed", "Each attempt asks for a speed between the slowest and "
               "the fastest. The walker learns to follow the speed it is given.", 0.2, 0.05, 3.0, 0.05, "m/s"),
        number("fastest_speed", "Fastest asked speed", "", 0.7, 0.05, 4.0, 0.05, "m/s"),
        number("attempt_seconds", "Attempt length", "How long one practice attempt lasts if the walker does "
               "not fall.", 20, 5, 60, 1, "s"),
    )),
    SettingGroup("reward", "What it cares about", "Points the walker earns or loses every moment. Learning is "
                 "the search for more points, so these shape how it walks.", (
        number("speed_points", "Walking at the asked speed", "Points for moving forward at the asked speed. The "
               "main goal.", 3.0, 0.0, 10.0, 0.1, "points"),
        number("alive_points", "Staying on its feet", "Points for every moment without falling. Too many and it "
               "learns to just stand still.", 0.2, 0.0, 2.0, 0.05, "points"),
        number("energy_cost", "Saving energy", "Points lost per joule the muscles use. Higher gives calmer, more "
               "efficient walking; too high and it barely moves.", 1.5, 0.0, 10.0, 0.1, "points"),
        number("straight_cost", "Walking straight", "Points lost per metre of drifting off the straight line.",
               0.5, 0.0, 5.0, 0.1, "points"),
        switch("end_stops", "Penalty for joints at their limits", "Lose points for resting joints against their "
               "end stops (the last 5% of their range). With direct muscles, leaning on the stops holds the body up "
               "for free, so walkers learn odd crouches without this. Learn_to_walk's Duck 4 used it."),
        number("end_stop_cost", "Joint limit penalty strength", "Points lost per joint fully at its limit, every "
               "moment. 0.05 is gentle; at 0.2 the duck preferred falling on purpose.", 0.05, 0.0, 0.5, 0.01,
               "points"),
    )),
    SettingGroup("extras", "Surprises", "Things that happen to the walker while it practises.", (
        switch("rocks", "Thrown rocks", "Stones are thrown at the walker from the side.", default=False),
        number("rock_weight", "Rock weight", "", 100, 10, 2000, 10, "g"),
        number("rock_every", "Time between rocks", "", 2.0, 0.5, 10.0, 0.5, "s"),
        switch("wind", "Gusts of wind", "Sudden pushes from a random direction.", default=False),
        number("wind_strength", "Gust strength", "How hard a gust pushes, as a share of the walker's weight.",
               30, 5, 150, 5, "%"),
        number("wind_every", "Time between gusts", "", 3.0, 0.5, 10.0, 0.5, "s"),
        switch("backpack", "Backpack", "A box to carry on its back.", default=False),
        number("backpack_weight", "Backpack weight", "", 200, 10, 3000, 10, "g"),
    )),
    SettingGroup("looks", "Looks", "Only for the 3D view.", (
        Setting("style", "Scenery", "choice", "meadow", "", choices=STYLES),
    )),
)

DEFAULTS = default_values(SETTING_GROUPS)


# One world with its values cleaned and checked
@dataclass(frozen=True)
class World:
    id: str
    name: str
    description: str
    values: dict
    builtin: bool = False

    # Shortcuts to the values the simulation reads often
    @property
    def terrain(self) -> str:
        return self.values["terrain"] if self.tallest > 0 else "flat"

    @property
    def tallest(self) -> float:  # metres
        return self.values["terrain_height"] / 100.0 if self.values["terrain"] != "flat" else 0.0

    @property
    def speed_range(self) -> tuple[float, float]:
        low, high = self.values["slowest_speed"], self.values["fastest_speed"]
        return (min(low, high), max(low, high))

    @property
    def grip(self) -> float:
        return self.values["ground_grip"] / 100.0

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description, "builtin": self.builtin,
                "values": dict(self.values)}


# A world from saved data, every missing value filled with its default
def world_from_dict(data: dict) -> World:
    values = clean_values(SETTING_GROUPS, data.get("values", {}))
    return World(id=str(data.get("id", "world")), name=str(data.get("name", "World")),
                 description=str(data.get("description", "")), values=values, builtin=bool(data.get("builtin")))


# The built-in worlds, in the order the page shows them
def builtin_worlds() -> list[World]:
    worlds = []
    for path in sorted(TEMPLATES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["builtin"] = True
        worlds.append(world_from_dict(data))
    return sorted(worlds, key=lambda world: _ORDER.get(world.id, 99))


# The order of the built-in worlds on the page
_ORDER = {"flat_meadow": 0, "cobblestone_street": 1, "running_track": 2, "rolling_hills": 3, "grand_staircase": 4,
          "frozen_lake": 5, "moon_base": 6, "stormy_crossing": 7, "rocky_canyon": 8}
