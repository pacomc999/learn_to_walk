"""Run settings: which duck a run belongs to, and the world and body it trains in.

A run keeps frozen copies of its world and body, so it always continues in its own world,
even if the duck's files change later.
"""
from dataclasses import dataclass
from pathlib import Path

from duck.bodies import Body
from duck.worlds import World


# The world and body of one run (or one duck), and the duck folder it belongs to (None for a scratch run)
@dataclass(frozen=True)
class RunSettings:
    duck_dir: Path | None
    world: World
    body: Body
