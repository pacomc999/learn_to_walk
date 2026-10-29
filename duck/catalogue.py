"""The duck catalogue: one folder per training recipe, kept in git.

A duck is a recipe (brain type, rules and settings), not a single brain. A new duck
starts only when the recipe changes. Each folder in ducks/ holds:
- duck.json: the story (what changed and why, where it starts, its runs) and the
  recipe, frozen from the code when the duck was created
- brain/snapshots/: the recipe's best brain so far (one snapshot)
- results.json and <folder>.gif: the best brain's measurements and a captioned GIF
- card.md: the readable card, written from the files above

Run from the project root, for example:
    python -m duck.catalogue new ducks/duck_02 --name "Duck 2"
    python -m duck.catalogue keep ducks/duck_02 runs/duck_02/main --snapshot 2007040
    python -m duck.catalogue cards
"""
import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from duck.duck_env import (EVALUATION_TARGET_SPEED, FRAME_SKIP, MAX_EPISODE_SECONDS, START_NOISE,
                           TARGET_SPEED_RANGE, DuckWalkEnv)
from duck.evaluate import evaluate_snapshot
from duck.exam import take_exam
from duck.policy import choose_snapshot_or_exit, load_policy
from duck.rewards import RewardWeights
from duck.snapshots import Snapshot, list_snapshots
from duck.train import DEFAULT_PPO_SETTINGS, NET_ARCH, NEW_RUN_POLICY, START_LOG_STD
from duck.video import add_caption, record_clip

CATALOGUE_DIR = Path("ducks")
STORY_NAME = "duck.json"
RESULTS_NAME = "results.json"
CARD_NAME = "card.md"
INDEX_NAME = "README.md"

# Gait measurements: three attempts at the judging speed, seeds 0, 1 and 2
GAIT_SEEDS = (0, 1, 2)

# The GIF: the first 16 seconds of attempt 0, small enough to send by phone
GIF_SECONDS = 16.0
GIF_FPS = 15
GIF_SIZE = (320, 240)


# ---------- The recipe, frozen from the code ----------

# The current code version, or "unknown" outside git
def _code_version() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Everything that defines how a new run learns, read from the code right now
def current_recipe() -> dict:
    return {
        "brain": NEW_RUN_POLICY.__name__,
        "layers": NET_ARCH,
        "start_log_std": START_LOG_STD,
        "ppo": DEFAULT_PPO_SETTINGS,
        "reward": asdict(RewardWeights()),
        "target_speed_range": list(TARGET_SPEED_RANGE),
        "attempt_seconds": MAX_EPISODE_SECONDS,
        "physics_steps_per_action": FRAME_SKIP,
        "start_noise": START_NOISE,
        "code_version": _code_version(),
    }


# ---------- Reading and writing a duck folder ----------

# Read a JSON file from a duck folder, or an empty dict if it is not there yet
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# Write a JSON file in a readable layout
def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# Start a new duck folder with the current recipe and an empty story to fill in
def new_duck(duck_dir: Path, name: str) -> Path:
    if (duck_dir / STORY_NAME).exists():
        raise FileExistsError(f"{duck_dir} already holds a duck")
    duck_dir.mkdir(parents=True)
    story = {
        "name": name,
        "started": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "summary": "",
        "what_changed": "",
        "why": "",
        "starts_from": "zero",
        "runs": [],
        "notes": "",
        "recipe": current_recipe(),
    }
    _write_json(duck_dir / STORY_NAME, story)
    write_card(duck_dir)
    return duck_dir


# ---------- Measuring a brain ----------

# The noise-free exam on a snapshot, with the sense scaling saved alongside it
def _exam_snapshot(snapshot: Snapshot):
    venv = DummyVecEnv([DuckWalkEnv])
    vec_normalize = VecNormalize.load(str(snapshot.norm_path), venv)
    vec_normalize.training = False
    model = PPO.load(str(snapshot.model_path), device="cpu")
    result = take_exam(model, vec_normalize)
    venv.close()
    return result


# One attempt at the judging speed: time spent with 0, 1 or 2 feet down, energy and distance
def _gait_attempt(policy, seed: int) -> dict:
    env = DuckWalkEnv(target_speed=EVALUATION_TARGET_SPEED)
    observation, _ = env.reset(seed=seed)
    feet_counts, energy, steps = [0, 0, 0], 0.0, 0
    while True:
        observation, _, fell, timed_out, info = env.step(policy(observation))
        steps += 1
        energy += -info["reward_parts"].get("energy", 0.0) / env.reward_weights.energy
        feet_counts[int(observation[-3] > 0.5) + int(observation[-2] > 0.5)] += 1
        if fell or timed_out:
            break
    env.close()
    return {"feet": np.array(feet_counts) / steps, "energy": energy,
            "distance": info["forward_distance"], "seconds": steps * env.model.opt.timestep * FRAME_SKIP, "fell": fell}


# Average foot use, energy per metre and speed over the gait attempts
def _gait_summary(snapshot: Snapshot) -> dict:
    policy = load_policy(snapshot)
    attempts = [_gait_attempt(policy, seed) for seed in GAIT_SEEDS]
    feet = np.mean([a["feet"] for a in attempts], axis=0)
    distance = sum(a["distance"] for a in attempts)
    return {
        "feet_in_air": round(float(feet[0]), 3),
        "one_foot_down": round(float(feet[1]), 3),
        "both_feet_down": round(float(feet[2]), 3),
        "joules_per_metre": round(sum(a["energy"] for a in attempts) / max(distance, 1e-6), 1),
        "speed": round(distance / sum(a["seconds"] for a in attempts), 2),
        "falls": sum(a["fell"] for a in attempts),
    }


# All measurements of one brain, plus where it came from
def measure_brain(snapshot: Snapshot, run_dir: Path) -> dict:
    exam = _exam_snapshot(snapshot)
    walking = evaluate_snapshot(snapshot)
    return {
        "from_run": run_dir.as_posix(),
        "steps": snapshot.steps,
        "measured": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exam_score": round(exam.score, 1),
        "exam_falls": exam.falls,
        "walking_test": f"{walking.successes} of {walking.attempts}",
        "gait_at_judging_speed": _gait_summary(snapshot),
    }


# ---------- Keeping the best brain ----------

# Copy a snapshot into the duck's brain folder, replacing the brain kept before
def _copy_brain(duck_dir: Path, snapshot: Snapshot) -> None:
    brain_dir = duck_dir / "brain" / "snapshots"
    if brain_dir.exists():
        shutil.rmtree(brain_dir)
    brain_dir.mkdir(parents=True)
    shutil.copy2(snapshot.model_path, brain_dir / snapshot.model_path.name)
    shutil.copy2(snapshot.norm_path, brain_dir / snapshot.norm_path.name)


# A GIF of the kept brain with the duck's name and exam score written on every frame
def _make_gif(duck_dir: Path, snapshot: Snapshot, caption: str) -> Path:
    frames = record_clip(load_policy(snapshot), seconds=GIF_SECONDS, fps=GIF_FPS,
                         width=GIF_SIZE[0], height=GIF_SIZE[1], seed=0)
    gif_path = duck_dir / f"{duck_dir.name}.gif"
    imageio.mimsave(gif_path, [add_caption(frame, caption) for frame in frames], duration=1000 / GIF_FPS, loop=0)
    return gif_path


# Measure a snapshot and keep it as the duck's brain if its exam beats the kept one (or if forced)
def keep_brain(duck_dir: Path, run_dir: Path, snapshot: Snapshot, force: bool = False) -> bool:
    story = _read_json(duck_dir / STORY_NAME)
    kept = _read_json(duck_dir / RESULTS_NAME)
    results = measure_brain(snapshot, run_dir)
    if kept and results["exam_score"] <= kept["exam_score"] and not force:
        print(f"Not kept: exam {results['exam_score']:.0f} does not beat the kept brain's {kept['exam_score']:.0f}.")
        return False
    _copy_brain(duck_dir, snapshot)
    _write_json(duck_dir / RESULTS_NAME, results)
    _make_gif(duck_dir, snapshot, f"{story['name']}, exam score {results['exam_score']:.0f}")
    write_card(duck_dir)
    print(f"Kept as {story['name']}'s best brain: exam {results['exam_score']:.0f}, "
          f"{results['exam_falls']} falls of 12.")
    return True


# ---------- The card and the list page ----------

# The card's recipe lines
def _recipe_lines(recipe: dict) -> list[str]:
    reward, ppo = recipe["reward"], recipe["ppo"]
    low, high = recipe["target_speed_range"]
    return [
        f"- Brain: {recipe['brain']}, actor layers {recipe['layers']['pi']}, judge layers {recipe['layers']['vf']}, "
        f"starting wobble log_std {recipe['start_log_std']}",
        f"- Reward per step: forward {reward['forward']} per m/s up to the target speed, alive {reward['alive']}, "
        f"energy {reward['energy']} per joule, sideways {reward['sideways']} per metre",
        f"- Target speed: random from {low} to {high} m/s per attempt; attempts last {recipe['attempt_seconds']:.0f} s",
        f"- PPO: " + ", ".join(f"{key} {value}" for key, value in ppo.items()),
        f"- Code version: {recipe['code_version']}",
    ]


# The card's lines about the kept brain
def _results_lines(results: dict, gif_name: str) -> list[str]:
    if not results:
        return ["No brain kept yet."]
    gait = results["gait_at_judging_speed"]
    return [
        f"![{gif_name}]({gif_name})",
        "",
        f"- From `{results['from_run']}` at {results['steps']:,} steps (measured {results['measured']})",
        f"- Exam (12 fixed attempts, no wobble): score {results['exam_score']:.0f}, falls {results['exam_falls']} of 12",
        f"- Walking test: {results['walking_test']} attempts walked",
        f"- At {EVALUATION_TARGET_SPEED} m/s: speed {gait['speed']} m/s, {gait['joules_per_metre']} J per metre, "
        f"feet in the air {gait['feet_in_air']:.0%}, one foot down {gait['one_foot_down']:.0%}, "
        f"both down {gait['both_feet_down']:.0%}",
    ]


# The card's training history, one line per run
def _run_lines(runs: list[dict]) -> list[str]:
    if not runs:
        return ["No runs yet."]
    return [f"- `{run['folder']}` ({run['dates']}, {run['steps']}): {run['outcome']}" for run in runs]


# Write card.md from the story, the recipe and the kept brain's results
def write_card(duck_dir: Path) -> Path:
    story = _read_json(duck_dir / STORY_NAME)
    results = _read_json(duck_dir / RESULTS_NAME)
    lines = [
        f"# {story['name']}", "", f"**{story['summary']}**" if story["summary"] else "", "",
        f"Started {story['started']}. Starts from: {story['starts_from']}.", "",
        "## What changed and why", "", story["what_changed"] or "(to write)", "", story["why"] or "", "",
        "## Recipe", "", *_recipe_lines(story["recipe"]), "",
        "## Training history", "", *_run_lines(story["runs"]), "",
        "## Best brain", "", *_results_lines(results, f"{duck_dir.name}.gif"), "",
        "## Notes", "", story["notes"] or "(none)", "",
    ]
    card_path = duck_dir / CARD_NAME
    card_path.write_text("\n".join(lines), encoding="utf-8")
    return card_path


# One line of the list page
def _index_line(duck_dir: Path) -> str:
    story = _read_json(duck_dir / STORY_NAME)
    results = _read_json(duck_dir / RESULTS_NAME)
    score = f"{results['exam_score']:.0f} ({results['exam_falls']} falls of 12)" if results else "none yet"
    return f"| [{story['name']}]({duck_dir.name}/{CARD_NAME}) | {story['started'][:10]} | {story['summary']} | {score} |"


# Write ducks/README.md with one line per duck, oldest first
def write_index(catalogue_dir: Path = CATALOGUE_DIR) -> Path:
    duck_dirs = sorted(path.parent for path in catalogue_dir.glob(f"*/{STORY_NAME}"))
    lines = ["# The ducks", "",
             "Each duck is a training recipe. Its card tells what changed, how it trained and how its best brain does.", "",
             "| Duck | Started | What changed | Best exam score |", "|---|---|---|---|",
             *[_index_line(duck_dir) for duck_dir in duck_dirs], ""]
    index_path = catalogue_dir / INDEX_NAME
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


# ---------- Command line ----------

# Rewrite every card and the list page (after editing a duck.json by hand)
def rewrite_all_cards(catalogue_dir: Path = CATALOGUE_DIR) -> None:
    for story_path in catalogue_dir.glob(f"*/{STORY_NAME}"):
        write_card(story_path.parent)
    write_index(catalogue_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep track of the ducks.")
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="start a duck with the current recipe")
    new.add_argument("duck_dir", type=Path)
    new.add_argument("--name", required=True)
    keep = commands.add_parser("keep", help="measure a snapshot and keep it if it is the duck's best")
    keep.add_argument("duck_dir", type=Path)
    keep.add_argument("run_dir", type=Path)
    keep.add_argument("--snapshot", type=int, default=None, help="step count (default: latest)")
    keep.add_argument("--force", action="store_true", help="keep it even if the exam is not better")
    commands.add_parser("cards", help="rewrite all cards and the list page")
    args = parser.parse_args()

    if args.command == "new":
        new_duck(args.duck_dir, args.name)
    elif args.command == "keep":
        snapshot = choose_snapshot_or_exit(args.run_dir, args.snapshot)
        keep_brain(args.duck_dir, args.run_dir, snapshot, force=args.force)
    rewrite_all_cards(args.duck_dir.parent if hasattr(args, "duck_dir") else CATALOGUE_DIR)


if __name__ == "__main__":
    main()
