"""The rock test: throw stone balls at a walking duck and count how often it falls.

A rock is a real MuJoCo body (a stone ball with its own free joint) added after the duck,
so it pushes the duck by actually hitting it. The duck senses only its own joints, so the
rock never shows up in its 39 senses. The duck walks on the flat floor at 0.4 m/s, so the
rocks are the only difficulty. Every 2 s a rock flies at its body from a random side.

Results go to <duck folder>/rocks/results.csv. With --film, the attempts with one rock size
are filmed to <duck folder>/rocks/films/ with a page that shows them all (films are not kept
in git; the same command makes them again).

Run from the project root, for example:
    python -m duck.rocks ducks/duck_03_1
    python -m duck.rocks ducks/duck_03_1 --film 150
"""
import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imageio
import mujoco
import numpy as np

from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv
from duck.model import DUCK_XML_PATH
from duck.policy import choose_snapshot_or_exit, load_policy
from duck.video import add_caption

# The rocks: stone balls (about granite) thrown at 4 m/s from 0.8 m to the side
ROCK_DENSITY = 2500.0      # kg per cubic metre
THROW_SPEED = 4.0          # metres per second
THROW_DISTANCE = 0.8       # metres
GRAVITY = 9.81             # metres per second squared, for the arc of the throw

# When rocks fly: the first at 2 s, then every 2 s; each is taken away again after 1 s,
# so a rock never lies on the floor as an obstacle
FIRST_THROW = 2.0
THROW_EVERY = 2.0
ROCK_LIFETIME = 1.0

# Where the rock waits between throws: off the walking line, behind the start
PARKED_POSITION = (-1.5, 0.95, 0.05)

# The default test: these rock sizes, 10 attempts of 20 s each
ROCK_GRAMS = (25, 50, 100, 125, 150, 175, 200, 400)
ATTEMPTS = 10

# Films: 30 pictures a second, kept running 1.5 s after a fall
FILM_FPS = 30
SECONDS_AFTER_FALL = 1.5

RESULTS_NAME = "results.csv"
FILMS_FOLDER = "films"


# ---------- The world with a rock ----------

# The radius of a stone ball of this mass, in metres
def rock_radius(mass: float) -> float:
    return (3 * mass / (4 * np.pi * ROCK_DENSITY)) ** (1 / 3)


# The duck XML with a grey stone ball after the duck, parked in the standing keyframe
def add_rock(xml: str, mass: float) -> str:
    x, y, z = PARKED_POSITION
    rock = (f'    <body name="rock" pos="{x} {y} {z}"><freejoint name="rock"/>'
            f'<geom name="rock" type="sphere" size="{rock_radius(mass):.4f}" mass="{mass}" '
            f'rgba="0.45 0.45 0.45 1" contype="1" conaffinity="1"/></body>\n  </worldbody>')
    xml = xml.replace("  </worldbody>", rock, 1)
    key_start = xml.index('<key name="stand" qpos="') + len('<key name="stand" qpos="')
    key_end = xml.index('"', key_start)
    return xml[:key_end] + f"  {x} {y} {z} 1 0 0 0" + xml[key_end:]


# The duck walking on the flat floor while a rock is thrown at it every THROW_EVERY seconds
class RockWorld(DuckWalkEnv):
    def __init__(self, rock_mass: float, throw_seed: int, **kwargs):
        self.rock_mass = rock_mass  # needed by _load_model, which the parent calls
        super().__init__(target_speed=EVALUATION_TARGET_SPEED, roughness=0.0, **kwargs)
        self.throw_rng = np.random.default_rng(throw_seed)
        self.rock_qpos = self.model.joint("rock").qposadr[0]
        self.rock_qvel = self.model.joint("rock").dofadr[0]
        self.rock_geom = self.model.geom("rock").id
        self.floor_geom = self.model.geom("floor").id

    # The flat duck world with the rock added
    def _load_model(self, roughness: float | None) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(add_rock(DUCK_XML_PATH.read_text(), self.rock_mass))

    # A normal start (the keyframe parks the rock), and no rocks thrown yet
    def reset(self, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.next_throw = FIRST_THROW
        self.thrown_at = None
        self.hit_this_throw = False
        self.throws = 0
        self.hits = 0
        return result

    # Throw the rock at the body from one side, aiming where the body will be and arcing for gravity
    def _throw(self) -> None:
        side = 1.0 if self.throw_rng.random() < 0.5 else -1.0
        flight_seconds = THROW_DISTANCE / THROW_SPEED
        target = self.data.body("torso").subtree_com + self.data.qvel[0:3] * flight_seconds
        start = target + np.array([0.0, side * THROW_DISTANCE, 0.0])
        velocity = (target - start) / flight_seconds
        velocity[2] += 0.5 * GRAVITY * flight_seconds
        self.data.qpos[self.rock_qpos:self.rock_qpos + 7] = [*start, 1, 0, 0, 0]
        self.data.qvel[self.rock_qvel:self.rock_qvel + 6] = [*velocity, 0, 0, 0]
        self.thrown_at = self.data.time
        self.hit_this_throw = False
        self.throws += 1

    # Put the rock back where it waits
    def _park(self) -> None:
        self.data.qpos[self.rock_qpos:self.rock_qpos + 7] = [*PARKED_POSITION, 1, 0, 0, 0]
        self.data.qvel[self.rock_qvel:self.rock_qvel + 6] = 0.0
        self.thrown_at = None

    # Whether the rock touches any part of the duck right now (anything but the floor)
    def _rock_touches_duck(self) -> bool:
        for contact in self.data.contact[:self.data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if self.rock_geom in pair and self.floor_geom not in pair:
                return True
        return False

    # Count a hit once per throw, and take the rock away when its time is up
    def _follow_rock(self) -> None:
        if not self.hit_this_throw and self._rock_touches_duck():
            self.hit_this_throw = True
            self.hits += 1
        if self.data.time - self.thrown_at >= ROCK_LIFETIME:
            self._park()

    # One control step: throw when it is time, step the duck, then follow the rock
    def step(self, action):
        if self.data.time >= self.next_throw:
            self._throw()
            self.next_throw += THROW_EVERY
        result = super().step(action)
        if self.thrown_at is not None:
            self._follow_rock()
        return result


# ---------- Testing a brain ----------

# How one attempt went
@dataclass
class RockAttempt:
    seconds: float
    fell: bool
    throws: int
    hits: int


# How one rock size went over all attempts
@dataclass
class RockRound:
    grams: int
    radius_cm: float
    walked: int
    attempts: int
    average_seconds: float
    throws: int
    hits: int


# One 20 s attempt with rocks of this size; seed picks the duck's start and the sides of the throws
def run_attempt(policy, grams: int, seed: int) -> RockAttempt:
    env = RockWorld(grams / 1000, throw_seed=seed)
    observation, _ = env.reset(seed=seed)
    fell = finished = False
    while not finished:
        observation, _, fell, timed_out, _ = env.step(policy(observation))
        finished = fell or timed_out
    attempt = RockAttempt(round(env.data.time, 2), fell, env.throws, env.hits)
    env.close()
    return attempt


# All attempts with one rock size, summed up
def rock_round(policy, grams: int, attempts: int = ATTEMPTS) -> RockRound:
    results = [run_attempt(policy, grams, seed) for seed in range(attempts)]
    return RockRound(
        grams=grams,
        radius_cm=round(rock_radius(grams / 1000) * 100, 1),
        walked=sum(not result.fell for result in results),
        attempts=attempts,
        average_seconds=round(float(np.mean([result.seconds for result in results])), 1),
        throws=sum(result.throws for result in results),
        hits=sum(result.hits for result in results),
    )


# Save the rounds as a CSV file, one row per rock size
def write_results(rounds: list[RockRound], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=list(asdict(rounds[0])))
        writer.writeheader()
        writer.writerows(asdict(one_round) for one_round in rounds)


# ---------- Films ----------

# The pictures of one attempt, and how it went. Once the duck falls it lies still (no commands)
def film_attempt(policy, grams: int, seed: int) -> tuple[list[np.ndarray], RockAttempt]:
    env = RockWorld(grams / 1000, throw_seed=seed, render_mode="rgb_array")
    observation, _ = env.reset(seed=seed)
    frames, fell_at = [env.render()], None
    while env.data.time < 20.0 and (fell_at is None or env.data.time < fell_at + SECONDS_AFTER_FALL):
        action = np.zeros(env.motor_count) if fell_at is not None else policy(observation)
        observation, _, fell, _, _ = env.step(action)
        if fell and fell_at is None:
            fell_at = env.data.time
        if env.data.time >= len(frames) / FILM_FPS - 1e-9:  # simulated time decides the pictures
            frames.append(env.render())
    attempt = RockAttempt(round(fell_at or env.data.time, 2), fell_at is not None, env.throws, env.hits)
    env.close()
    return frames, attempt


# How an attempt ended, in words
def outcome_words(attempt: RockAttempt) -> str:
    return f"falls after {attempt.seconds:.1f} s" if attempt.fell else "walks all 20 s"


# Film every attempt with one rock size and save a page that plays them all
def film_round(policy, name: str, grams: int, folder: Path, attempts: int = ATTEMPTS) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for seed in range(attempts):
        frames, attempt = film_attempt(policy, grams, seed)
        caption = f"{name}, {grams} g rocks at {THROW_SPEED:g} m/s: {outcome_words(attempt)}"
        imageio.mimsave(folder / f"attempt_{seed + 1}.mp4", [add_caption(frame, caption) for frame in frames],
                        fps=FILM_FPS, macro_block_size=1)
        outcomes.append(attempt)
        print(f"Attempt {seed + 1}: {outcome_words(attempt)}", flush=True)
    page = folder / "index.html"
    page.write_text(film_page(name, grams, outcomes), encoding="utf-8")
    return page


# One film on the page, with its attempt number and how it ended
def _film_card(number: int, attempt: RockAttempt) -> str:
    kind = "fall" if attempt.fell else "walk"
    return (f'    <figure class="card {kind}">\n'
            f'      <video src="attempt_{number}.mp4" muted autoplay loop playsinline controls></video>\n'
            f'      <figcaption><b>Attempt {number}</b><span>{outcome_words(attempt)}</span></figcaption>\n'
            f'    </figure>')


# A page that plays every film of a round side by side, light and dark
def film_page(name: str, grams: int, outcomes: list[RockAttempt]) -> str:
    walked = sum(not attempt.fell for attempt in outcomes)
    cards = "\n".join(_film_card(number, attempt) for number, attempt in enumerate(outcomes, start=1))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} Rocks</title>
<style>
  /* Colours, with a dark version */
  :root {{ --bg: #f4f3ef; --card: #fcfcfb; --ink: #0b0b0b; --soft: #52514e; --walk: #0c7a0c; --fall: #b5452f; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #121211; --card: #1a1a19; --ink: #f0efec; --soft: #c3c2b7; --walk: #3fc53f; --fall: #e07a63; }}
  }}
  /* Page and the grid of films */
  body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.45 system-ui, "Segoe UI", sans-serif; }}
  main {{ max-width: 1400px; margin: 0 auto; padding: 24px 16px 40px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  .sub {{ color: var(--soft); margin: 0 0 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
  .card {{ margin: 0; background: var(--card); border-radius: 12px; padding: 10px; }}
  .card video {{ width: 100%; border-radius: 8px; display: block; aspect-ratio: 4 / 3; background: #000; }}
  figcaption {{ display: flex; justify-content: space-between; padding: 8px 4px 2px; }}
  .walk span {{ color: var(--walk); font-weight: 600; }}
  .fall span {{ color: var(--fall); font-weight: 600; }}
</style>
</head>
<body>
<main>
  <h1>{name} against {grams} g rocks</h1>
  <p class="sub">Flat floor, asked to walk at {EVALUATION_TARGET_SPEED} m/s. A {grams} g stone ball
  ({rock_radius(grams / 1000) * 100:.1f} cm) is thrown at its body every {THROW_EVERY:g} s from a random side
  at {THROW_SPEED:g} m/s. {walked} of {len(outcomes)} attempts walk the full 20 s.
  Falling films stop {SECONDS_AFTER_FALL:g} s after the fall.</p>
  <section class="grid">
{cards}
  </section>
</main>
</body>
</html>
"""


# ---------- Command line ----------

# The duck's name from its story, or the folder name
def _duck_name(duck_dir: Path) -> str:
    story_path = duck_dir / "duck.json"
    if story_path.exists():
        return json.loads(story_path.read_text(encoding="utf-8"))["name"]
    return duck_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Throw rocks at a duck's kept brain and count the falls.")
    parser.add_argument("duck_dir", type=Path, help="duck folder, for example ducks/duck_03_1")
    parser.add_argument("--grams", type=int, nargs="+", default=list(ROCK_GRAMS), help="rock sizes to test")
    parser.add_argument("--film", type=int, default=None, help="also film every attempt with rocks of this size")
    arguments = parser.parse_args()

    policy = load_policy(choose_snapshot_or_exit(arguments.duck_dir / "brain", None))
    rounds = []
    for grams in arguments.grams:
        one_round = rock_round(policy, grams)
        rounds.append(one_round)
        print(f"{grams} g: {one_round.walked} of {one_round.attempts} walked, "
              f"{one_round.hits} hits of {one_round.throws} throws", flush=True)
    write_results(rounds, arguments.duck_dir / "rocks" / RESULTS_NAME)
    if arguments.film is not None:
        page = film_round(policy, _duck_name(arguments.duck_dir), arguments.film,
                          arguments.duck_dir / "rocks" / FILMS_FOLDER)
        print(f"Films and page saved in {page.parent}")


if __name__ == "__main__":
    main()
