"""The rock test: throw stone balls at a walking duck and count how often it falls.

The rock itself and the throwing rules live in duck/rock_thrower.py, so any world can have
rocks (world extras). This file holds the rocks challenge's world: the duck on the flat
floor at 0.4 m/s with rocks as the only difficulty, the throws decided by their own seed
(as when the challenge was first run), and the films of one rock size.

The test runs as the "rocks" challenge of duck/challenges.py, for example:
    python -m duck.challenges duck_catalogue/duck_03_1 rocks --film 150
"""
from dataclasses import dataclass, replace
from pathlib import Path

import imageio
import numpy as np

from duck.bodies import Body
from duck.duck_env import EVALUATION_TARGET_SPEED, DuckWalkEnv
from duck.rock_thrower import FIRST_THROW, PARKED_POSITION, ROCK_LIFETIME, add_rock, rock_radius  # noqa: F401
from duck.video import FilmCard, add_caption
from duck.video import film_page as video_film_page
from duck.worlds import Rocks, World

# The challenge throws at 4 m/s every 2 s (the Rocks defaults)
THROW_SPEED = Rocks(mass=1.0).speed
THROW_EVERY = Rocks(mass=1.0).every_seconds

# Films: one per attempt, 10 attempts
ATTEMPTS = 10

# Films: 30 pictures a second, kept running 1.5 s after a fall
FILM_FPS = 30
SECONDS_AFTER_FALL = 1.5


# ---------- The world with a rock ----------

# The duck walking on the flat floor of its world while a rock is thrown at it every 2 s.
# The sides of the throws come from throw_seed, not the attempt's seed, as in the first runs
class RockWorld(DuckWalkEnv):
    def __init__(self, rock_mass: float, throw_seed: int, world: World, body: Body, **kwargs):
        super().__init__(world=replace(world, rocks=Rocks(rock_mass)), body=body,
                         target_speed=EVALUATION_TARGET_SPEED, roughness=0.0, **kwargs)
        self.throw_rng = np.random.default_rng(throw_seed)

    # Throws and hits so far, and where the rock's position starts in qpos
    @property
    def throws(self) -> int:
        return self.rock_thrower.throws

    @property
    def hits(self) -> int:
        return self.rock_thrower.hits

    @property
    def rock_qpos(self) -> int:
        return self.rock_thrower.qpos_start


# ---------- Films ----------

# How one filmed attempt went
@dataclass
class RockAttempt:
    seconds: float
    fell: bool
    throws: int
    hits: int


# The pictures of one attempt, and how it went. Once the duck falls it lies still (no commands)
def film_attempt(policy, grams: int, seed: int, world: World, body: Body) -> tuple[list[np.ndarray], RockAttempt]:
    env = RockWorld(grams / 1000, throw_seed=seed, world=world, body=body, render_mode="rgb_array")
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
def film_round(policy, name: str, grams: int, folder: Path, world: World, body: Body, attempts: int = ATTEMPTS) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for seed in range(attempts):
        frames, attempt = film_attempt(policy, grams, seed, world, body)
        caption = f"{name}, {grams} g rocks at {THROW_SPEED:g} m/s: {outcome_words(attempt)}"
        imageio.mimsave(folder / f"attempt_{seed + 1}.mp4", [add_caption(frame, caption) for frame in frames],
                        fps=FILM_FPS, macro_block_size=1)
        outcomes.append(attempt)
        print(f"Attempt {seed + 1}: {outcome_words(attempt)}", flush=True)
    page = folder / "index.html"
    page.write_text(film_page(name, grams, outcomes), encoding="utf-8")
    return page


# A page that plays every film of a round side by side
def film_page(name: str, grams: int, outcomes: list[RockAttempt]) -> str:
    walked = sum(not attempt.fell for attempt in outcomes)
    intro = (f"Flat floor, asked to walk at {EVALUATION_TARGET_SPEED} m/s. A {grams} g stone ball "
             f"({rock_radius(grams / 1000) * 200:.1f} cm across) is thrown at its body every {THROW_EVERY:g} s from a "
             f"random side at {THROW_SPEED:g} m/s. {walked} of {len(outcomes)} attempts walk the full 20 s. "
             f"Falling films stop {SECONDS_AFTER_FALL:g} s after the fall.")
    cards = [FilmCard(f"attempt_{number}.mp4", f"Attempt {number}", outcome_words(attempt), attempt.fell)
             for number, attempt in enumerate(outcomes, start=1)]
    return video_film_page(f"{name} against {grams} g rocks", intro, [(None, None, cards)])
