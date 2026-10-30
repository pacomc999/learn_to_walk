"""Challenges: situations a kept duck never trained for, to see how its brain copes.

Each challenge walks the duck's kept brain through a fixed set of attempts and saves one table,
<duck folder>/challenges/<challenge>/results.csv. The same command always gives the same numbers,
so ducks can be compared fairly. challenges/README.md is rewritten from every saved table after
each run; remarks written by hand go in challenges/notes.md and are copied into it.

Run from the project root, for example:
    python -m duck.challenges duck_catalogue/duck_03_1 speed backpack
    python -m duck.challenges duck_catalogue/duck_03_1 rocks --film 150
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv
from duck.floor import MAX_ROUGHNESS, load_floor_model
from duck.model import DUCK_XML_PATH
from duck.policy import choose_snapshot_or_exit, load_policy
from duck.rewards import RewardWeights
from duck.rocks import RockWorld, film_round, rock_radius

CHALLENGES_FOLDER = "challenges"
RESULTS_NAME = "results.csv"
NOTES_NAME = "notes.md"
README_NAME = "README.md"

# The backpack: an 8 x 7 x 4 cm box fixed on the back, slightly behind the middle (a body part, no joint)
BACKPACK_SIZE = "0.04 0.035 0.02"
BACKPACK_POSITION = "-0.04 0 0.1"


# ---------- Worlds the challenges need ----------

# The duck XML with a backpack of this mass (kg) fixed to its body
def add_backpack(xml: str, mass: float) -> str:
    joint = '<freejoint name="root"/>'
    backpack = (f'{joint}\n      <geom name="backpack" type="box" size="{BACKPACK_SIZE}" pos="{BACKPACK_POSITION}" '
                f'mass="{mass}" rgba="0.2 0.4 0.75 1"/>')
    return xml.replace(joint, backpack, 1)


# The duck on the flat floor carrying a backpack
class BackpackWorld(DuckWalkEnv):
    def __init__(self, mass: float, **kwargs):
        self.backpack_mass = mass  # needed by _load_model, which the parent calls
        super().__init__(target_speed=EVALUATION_TARGET_SPEED, roughness=0.0, **kwargs)

    def _load_model(self, roughness: float | None) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(add_backpack(DUCK_XML_PATH.read_text(), self.backpack_mass))


# The duck on cobblestones of one height, which may be taller than training ever used
class TallStoneWorld(DuckWalkEnv):
    def __init__(self, height: float, **kwargs):
        super().__init__(target_speed=EVALUATION_TARGET_SPEED, roughness=height, **kwargs)

    def _load_model(self, roughness: float | None) -> mujoco.MjModel:
        return load_floor_model(most=max(MAX_ROUGHNESS, roughness))


# ---------- The challenges ----------

# One challenge: what it asks, the values it tries, and how to build the world for one value
@dataclass(frozen=True)
class Challenge:
    title: str
    description: str
    column: str                              # what the tried value means, for the table
    values: tuple
    world: Callable[[float, int], DuckWalkEnv]
    attempts: int = 10


CHALLENGES = {
    "speed": Challenge(
        title="Speeds above training",
        description="Asked for one speed for the whole attempt, on the flat floor. Training asked for 0.2 to 0.7 m/s.",
        column="asked m/s",
        values=(0.4, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5),
        world=lambda speed, seed: DuckWalkEnv(target_speed=speed, roughness=0.0),
    ),
    "backpack": Challenge(
        title="A backpack",
        description=f"An 8 x 7 x 4 cm box fixed on the back, on the flat floor at "
                    f"{EVALUATION_TARGET_SPEED} m/s. The duck weighs 1.53 kg and has no sense for the extra weight.",
        column="backpack g",
        values=(0, 100, 200, 300, 500, 800),
        world=lambda grams, seed: (BackpackWorld(grams / 1000) if grams
                                   else DuckWalkEnv(target_speed=EVALUATION_TARGET_SPEED, roughness=0.0)),
    ),
    "stones": Challenge(
        title="Taller stones",
        description=f"Cobblestones of one height for the whole attempt, at {EVALUATION_TARGET_SPEED} m/s, a new layout "
                    "each attempt. 30 attempts per height, because on tall stones tiny differences decide falls.",
        column="stones cm",
        values=(1, 2, 3, 4, 5, 6, 8),
        world=lambda cm, seed: TallStoneWorld(cm / 100),
        attempts=30,
    ),
    "rocks": Challenge(
        title="Thrown rocks",
        description=f"A stone ball thrown at the body every 2 s from a random side at 4 m/s, on the flat floor at "
                    f"{EVALUATION_TARGET_SPEED} m/s (see duck/rocks.py). The rock is taken away 1 s after each throw.",
        column="rock g",
        values=(25, 50, 100, 125, 150, 175, 200, 400),
        world=lambda grams, seed: RockWorld(grams / 1000, throw_seed=seed),
    ),
}


# ---------- Running them ----------

# How one attempt went
@dataclass
class Attempt:
    seconds: float
    fell: bool
    speed: float             # average forward speed, m/s
    joules_per_metre: float
    hits: int | None = None  # rocks only


# One attempt: walk until a fall or the 20 s end, measuring speed and motor energy
def run_attempt(env: DuckWalkEnv, policy, seed: int) -> Attempt:
    observation, info = env.reset(seed=seed)
    start_x, energy_points, fell, finished = info["x_position"], 0.0, False, False
    while not finished:
        observation, _, fell, timed_out, info = env.step(policy(observation))
        energy_points += info["reward_parts"].get("energy", 0.0)
        finished = fell or timed_out
    seconds, distance = env.data.time, info["x_position"] - start_x
    joules = -energy_points / RewardWeights().energy
    return Attempt(seconds, fell, distance / seconds, joules / max(distance, 1e-6), getattr(env, "hits", None))


# One row of a challenge's table: every attempt with one value, summed up
def challenge_row(challenge: Challenge, value, policy, attempts: int) -> dict:
    results = []
    for seed in range(attempts):
        env = challenge.world(value, seed)
        results.append(run_attempt(env, policy, seed))
        env.close()
    row = {
        challenge.column: value,
        "walked": sum(not result.fell for result in results),
        "attempts": attempts,
        "average seconds up": round(float(np.mean([r.seconds for r in results])), 1),
        "actual m/s": round(float(np.mean([r.speed for r in results])), 2),
        "J per metre": round(float(np.mean([r.joules_per_metre for r in results])), 1),
    }
    if results[0].hits is not None:
        row["rock diameter cm"] = round(rock_radius(value / 1000) * 200, 1)
        row["hits"] = sum(r.hits for r in results)
    return row


# Run one challenge and save its table
def run_challenge(name: str, policy, duck_dir: Path, attempts: int | None = None) -> list[dict]:
    challenge = CHALLENGES[name]
    rows = []
    for value in challenge.values:
        rows.append(challenge_row(challenge, value, policy, attempts or challenge.attempts))
        print(f"{name} {value}: {rows[-1]['walked']} of {rows[-1]['attempts']} walked", flush=True)
    write_table(rows, duck_dir / CHALLENGES_FOLDER / name / RESULTS_NAME)
    return rows


# Save rows as a CSV file
def write_table(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------- The README ----------

# A saved table as Markdown
def _markdown_table(path: Path) -> list[str]:
    with path.open(newline="") as table_file:
        rows = list(csv.reader(table_file))
    header, body = rows[0], rows[1:]
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(row) + " |" for row in body])


# Rewrite challenges/README.md from every saved table, plus the notes written by hand
def write_readme(duck_dir: Path, name: str) -> Path:
    folder = duck_dir / CHALLENGES_FOLDER
    lines = [f"# {name}: challenges", "",
             "Situations the duck never trained for, met by its kept brain (no wobble). Each attempt lasts up to 20 s; "
             "walked means it did not fall. Written by `python -m duck.challenges`; remarks go in `notes.md`.", ""]
    for key, challenge in CHALLENGES.items():
        table = folder / key / RESULTS_NAME
        if table.exists():
            lines += [f"## {challenge.title} (`{key}`)", "", challenge.description, "", *_markdown_table(table), ""]
    notes = folder / NOTES_NAME
    if notes.exists():
        lines += ["## Notes", "", notes.read_text(encoding="utf-8").strip(), ""]
    readme = folder / README_NAME
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


# The duck's name from its story, or the folder name
def duck_name(duck_dir: Path) -> str:
    story = duck_dir / "duck.json"
    return json.loads(story.read_text(encoding="utf-8"))["name"] if story.exists() else duck_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Put a duck's kept brain through challenges it never trained for.")
    parser.add_argument("duck_dir", type=Path, help="duck folder, for example duck_catalogue/duck_03_1")
    parser.add_argument("challenges", nargs="+", choices=list(CHALLENGES), help="which challenges to run")
    parser.add_argument("--attempts", type=int, default=None, help="attempts per value (default: the challenge's own)")
    parser.add_argument("--film", type=int, default=None, help="rocks only: film every attempt with rocks of this many grams")
    arguments = parser.parse_args()

    policy = load_policy(choose_snapshot_or_exit(arguments.duck_dir / "brain", None))
    for name in arguments.challenges:
        run_challenge(name, policy, arguments.duck_dir, arguments.attempts)
    if arguments.film is not None and "rocks" in arguments.challenges:
        film_round(policy, duck_name(arguments.duck_dir), arguments.film,
                   arguments.duck_dir / CHALLENGES_FOLDER / "rocks" / "films")
    print(f"Saved in {write_readme(arguments.duck_dir, duck_name(arguments.duck_dir))}")


if __name__ == "__main__":
    main()
