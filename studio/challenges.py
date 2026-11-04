"""The report card: a set of challenges that measure what a trained walker can do.

Each challenge is a ladder of levels that get harder (faster asked speeds, taller stones, more
slippery ice, a heavier backpack...). The walker tries each level once, for CHALLENGE_SECONDS,
and the ladder stops at the first level it fails. A level counts as passed when it stays up the
whole time and keeps at least MIN_SPEED_SHARE of the asked speed. Results go to
<training>/challenges.json, so the page can show them as a report card with medals.
"""
import threading
import traceback
import uuid

import numpy as np

from studio import library
from studio.brain import clean_design, design_senses
from studio.creatures import build_creature
from studio.env import WalkEnv
from studio.replay import record
from studio.worlds import world_from_dict

CHALLENGE_SECONDS = 15.0
MIN_SPEED_SHARE = 0.6
SEED = 2024

# name, title, what the levels change, the levels, and how to show a level
CHALLENGES = [
    {"key": "speed", "title": "Speed", "description": "Walk or run at faster and faster asked speeds on flat ground.",
     "levels": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5], "unit": "m/s"},
    {"key": "stones", "title": "Cobblestones", "description": "Cobblestones that get taller.",
     "levels": [1, 2, 3, 4, 5, 6], "unit": "cm"},
    {"key": "stairs", "title": "Stairs", "description": "Flights of steps that get taller.",
     "levels": [1, 2, 3, 4, 5, 6], "unit": "cm"},
    {"key": "ice", "title": "Ice", "description": "Ground that gets more and more slippery.",
     "levels": [60, 40, 25, 15, 10], "unit": "% grip"},
    {"key": "backpack", "title": "Backpack", "description": "A heavier and heavier box on its back.",
     "levels": [10, 25, 50, 100, 200], "unit": "% of its weight"},
    {"key": "wind", "title": "Wind", "description": "Stronger and stronger gusts from random directions.",
     "levels": [10, 20, 35, 50, 75], "unit": "% of its weight"},
    {"key": "gravity", "title": "Other planets", "description": "Walking where gravity is different.",
     "levels": [3.7, 1.6, 20.0, 25.0], "unit": "m/s²", "ladder": False},
]

_jobs: dict[str, dict] = {}


# The world of one level, built on top of the walker's own world (its speeds and scoring)
def level_world(base: dict, challenge: str, level: float, mass: float) -> dict:
    values = dict(base.get("values") or {})
    values.update(terrain="flat", terrain_height=0.0, rocks=False, wind=False, backpack=False, gradually_harder=False,
                  mixed_heights=False, attempt_seconds=CHALLENGE_SECONDS)
    if challenge == "stones":
        values.update(terrain="cobblestones", terrain_height=level, style="street")
    elif challenge == "stairs":
        values.update(terrain="stairs", terrain_height=level, style="street")
    elif challenge == "ice":
        values.update(ground_grip=level, style="snow")
    elif challenge == "backpack":
        values.update(backpack=True, backpack_weight=max(10.0, level / 100 * mass * 1000))
    elif challenge == "wind":
        values.update(wind=True, wind_strength=level, wind_every=2.0, style="night")
    elif challenge == "gravity":
        values.update(gravity=level, style="moon" if level < 5 else "desert")
    elif challenge == "speed":
        values.update(slowest_speed=min(level, values.get("slowest_speed", 0.2)), fastest_speed=max(level, 0.1))
    return {"id": f"challenge_{challenge}", "name": f"{challenge.capitalize()} challenge", "values": values}


def _speed_for(challenge: str, level: float, world) -> float:
    if challenge == "speed":
        return float(level)
    return float(np.mean(world.speed_range))


def run_level(training_id: str, challenge: str, level: float, keep_replay: bool = False) -> dict:
    from studio.previews import _brain  # the cached brain loader
    setup = library.read_json(library.training_folder(training_id) / "setup.json")
    build = build_creature(setup["creature"]["animal"], setup["creature"]["values"])
    senses = design_senses(clean_design(setup.get("brain")))
    from studio.scene import build_scene
    from studio.worlds import World
    plain = build_scene(build, World("plain", "plain", "", dict(world_from_dict({}).values)))
    world = world_from_dict(level_world(setup["world"], challenge, level, plain.total_mass))
    speed = _speed_for(challenge, level, world_from_dict(setup["world"]))
    env = WalkEnv(build, world, senses, target_speed=speed, terrain_height=world.tallest)
    replay = record(env, _brain(training_id).act, seed=SEED, seconds=CHALLENGE_SECONDS)
    stats = replay["stats"]
    passed = (not stats["fell"]) and stats["average_speed"] >= MIN_SPEED_SHARE * speed
    result = {"level": level, "passed": bool(passed), "fell": stats["fell"], "seconds": stats["seconds"],
              "distance": stats["distance"], "average_speed": stats["average_speed"], "asked_speed": speed}
    if keep_replay:
        result["replay"] = replay
    return result


def _medal(passed: int, total: int) -> str:
    share = passed / total
    if share >= 0.99:
        return "gold"
    if share >= 0.6:
        return "silver"
    if share >= 0.3:
        return "bronze"
    return "none"


def _run_all(job_id: str, training_id: str) -> None:
    job = _jobs[job_id]
    results = {"challenges": [], "seconds": CHALLENGE_SECONDS, "date": library.now()}
    total = sum(len(challenge["levels"]) for challenge in CHALLENGES)
    done = 0
    try:
        for challenge in CHALLENGES:
            levels, failed = [], False
            for level in challenge["levels"]:
                job["message"] = f"{challenge['title']}: {level:g} {challenge['unit']}"
                if failed:
                    levels.append({"level": level, "passed": False, "skipped": True})
                else:
                    outcome = run_level(training_id, challenge["key"], level)
                    levels.append(outcome)
                    failed = challenge.get("ladder", True) and not outcome["passed"]
                done += 1
                job["progress"] = done / total
            passed = sum(1 for level in levels if level["passed"])
            results["challenges"].append({**{k: challenge[k] for k in ("key", "title", "description", "unit")},
                                          "levels": levels, "passed": passed,
                                          "medal": _medal(passed, len(levels))})
        library.write_json(library.training_folder(training_id) / "challenges.json", results)
        job.update(state="done", progress=1.0, message="Done", result=results)
    except Exception as error:
        traceback.print_exc()
        job.update(state="error", message=f"Something went wrong: {error}")


# Start the report card in the background; the page follows it with job_status
def start(training_id: str) -> str:
    folder = library.training_folder(training_id)
    if not (folder / "brain" / "best.zip").exists():
        raise ValueError("This walker has no trained brain yet.")
    for job in _jobs.values():
        if job["training"] == training_id and job["state"] == "running":
            return job["id"]
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"id": job_id, "training": training_id, "state": "running", "progress": 0.0,
                     "message": "Starting"}
    threading.Thread(target=_run_all, args=(job_id, training_id), daemon=True).start()
    return job_id


def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise ValueError("That job is not known (was the studio restarted?).")
    return job


def saved_results(training_id: str) -> dict | None:
    return library.read_json(library.training_folder(training_id) / "challenges.json")
