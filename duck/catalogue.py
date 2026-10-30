"""The duck catalogue: one folder per training recipe, kept in git.

A duck is a recipe (brain type, rules and settings), not a single brain. A new duck
starts only when the recipe changes. Each folder in duck_catalogue/ holds:
- duck.json: the story (what changed and why, where it starts, its runs) and the
  recipe, frozen from the code when the duck was created
- brain/snapshots/: the recipe's best brain so far (one snapshot)
- results.json: the best brain's measurements
- <folder>.mp4: a captioned film of the best brain, not kept in git (the brain and the
  code are, so the film can always be made again with the film command)
- card.md: the readable card, written from the files above

Run from the project root, for example:
    python -m duck.catalogue new duck_catalogue/duck_02 --name "Duck 2"
    python -m duck.catalogue keep duck_catalogue/duck_02 runs/duck_02/main --snapshot 2007040
    python -m duck.catalogue film duck_catalogue/duck_02
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
                           START_STRIDE, TARGET_SPEED_RANGE, DuckWalkEnv)
from duck.evaluate import evaluate_snapshot
from duck.exam import take_full_exam
from duck.floor import FLOOR_SHAPE, TRAINING_ROUGHNESS
from duck.policy import choose_snapshot_or_exit, load_policy
from duck.rewards import RewardWeights
from duck.snapshots import Snapshot, list_snapshots, slim_brain_file
from duck.train import DEFAULT_PPO_SETTINGS, NET_ARCH, NEW_RUN_POLICY, START_LOG_STD, STEP_SIZE_RULE
from duck.video import add_caption, record_clip

CATALOGUE_DIR = Path("duck_catalogue")
STORY_NAME = "duck.json"
RESULTS_NAME = "results.json"
CARD_NAME = "card.md"
INDEX_NAME = "README.md"

# Gait measurements: three attempts at the judging speed, seeds 0, 1 and 2
GAIT_SEEDS = (0, 1, 2)

# The film: the first 16 seconds of attempt 0 at the full film size (about 5 MB, not kept in git)
FILM_SECONDS = 16.0
FILM_FPS = 15


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
        "step_size_rule": STEP_SIZE_RULE,
        "reward": asdict(RewardWeights()),
        "forward_reward": f"{RewardWeights().forward} x (target - |speed - target|)",
        "floor": {"max_roughness": TRAINING_ROUGHNESS, **FLOOR_SHAPE,
                  "rough_exam": "spread" if TRAINING_ROUGHNESS else "none"},
        "target_speed_range": list(TARGET_SPEED_RANGE),
        "attempt_seconds": MAX_EPISODE_SECONDS,
        "physics_steps_per_action": FRAME_SKIP,
        "start_noise": START_NOISE,
        "start_stride": START_STRIDE,
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

# The two part noise-free exam on a snapshot, with the sense scaling saved alongside it
def _exam_snapshot(snapshot: Snapshot):
    venv = DummyVecEnv([lambda: DuckWalkEnv(roughness=0.0)])
    vec_normalize = VecNormalize.load(str(snapshot.norm_path), venv)
    vec_normalize.training = False
    model = PPO.load(str(snapshot.model_path), device="cpu")
    result = take_full_exam(model, vec_normalize)
    venv.close()
    return result


# One attempt at the judging speed: time spent with 0, 1 or 2 feet down, energy and distance
def _gait_attempt(policy, seed: int, roughness: float) -> dict:
    env = DuckWalkEnv(target_speed=EVALUATION_TARGET_SPEED, roughness=roughness)
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


# Average foot use, energy per metre and speed over the gait attempts on a floor of this roughness
def _gait_summary(snapshot: Snapshot, roughness: float = 0.0) -> dict:
    policy = load_policy(snapshot)
    attempts = [_gait_attempt(policy, seed, roughness) for seed in GAIT_SEEDS]
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
    walking = evaluate_snapshot(snapshot, roughness=0.0)
    results = {
        "from_run": run_dir.as_posix(),
        "steps": snapshot.steps,
        "measured": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exam_score": round(exam.score, 1),
        "exam_falls": exam.falls,
        "walking_test": f"{walking.successes} of {walking.attempts}",
        "gait_at_judging_speed": _gait_summary(snapshot, 0.0),
    }
    if TRAINING_ROUGHNESS > 0:  # training on stones: the exam's halves, walking test and gait on stones too
        walking_rough = evaluate_snapshot(snapshot, roughness=TRAINING_ROUGHNESS)
        results.update({
            "exam_attempts": exam.attempts,
            "exam_lower_score": round(exam.parts["lower"].score, 1),
            "exam_taller_score": round(exam.parts["taller"].score, 1),
            "walking_test_rough": f"{walking_rough.successes} of {walking_rough.attempts}",
            "gait_on_rough_floor": _gait_summary(snapshot, TRAINING_ROUGHNESS),
            "rough_floor_height": TRAINING_ROUGHNESS,
        })
    return results


# ---------- Keeping the best brain ----------

# Copy a snapshot into the duck's brain folder, replacing the brain kept before
def _copy_brain(duck_dir: Path, snapshot: Snapshot) -> None:
    brain_dir = duck_dir / "brain" / "snapshots"
    if brain_dir.exists():
        shutil.rmtree(brain_dir)
    brain_dir.mkdir(parents=True)
    shutil.copy2(snapshot.model_path, brain_dir / snapshot.model_path.name)
    shutil.copy2(snapshot.norm_path, brain_dir / snapshot.norm_path.name)
    slim_brain_file(brain_dir / snapshot.model_path.name)  # a kept brain needs no optimizer memory


# A film of the kept brain with the duck's name and exam score written on every frame
def _make_film(duck_dir: Path, snapshot: Snapshot, caption: str) -> Path:
    frames = record_clip(load_policy(snapshot), seconds=FILM_SECONDS, fps=FILM_FPS, seed=0)
    film_path = duck_dir / f"{duck_dir.name}.mp4"
    imageio.mimsave(film_path, [add_caption(frame, caption) for frame in frames], fps=FILM_FPS, macro_block_size=1)
    return film_path


# The caption on a duck's film
def _film_caption(story: dict, results: dict) -> str:
    return f"{story['name']}, exam score {results['exam_score']:.0f}"


# Make the film of a duck's kept brain again, for example after a fresh clone of the repository
def film_duck(duck_dir: Path) -> Path | None:
    kept = list_snapshots(duck_dir / "brain")
    if not kept:
        print(f"{duck_dir} has no kept brain to film.")
        return None
    story = _read_json(duck_dir / STORY_NAME)
    film_path = _make_film(duck_dir, kept[-1], _film_caption(story, _read_json(duck_dir / RESULTS_NAME)))
    print(f"Film saved as {film_path}")
    return film_path


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
    _make_film(duck_dir, snapshot, _film_caption(story, results))
    write_card(duck_dir)
    print(f"Kept as {story['name']}'s best brain: exam {results['exam_score']:.0f}, "
          f"{results['exam_falls']} falls of {_exam_attempts(results)}.")
    return True


# ---------- The card and the list page ----------

# How many exam attempts a stored result covers: as stored, else 24 with both old parts, 12 for older results
def _exam_attempts(results: dict) -> int:
    if "exam_attempts" in results:
        return results["exam_attempts"]
    return 24 if "exam_flat_score" in results else 12


# The extra recipe line for a duck started from another brain
STARTED_FROM_BRAIN_LINE = ("- Started from another brain: it keeps that brain's wobble, step size and PPO settings, "
                           "so the starting values above apply only to runs from zero")


# True when the story says the duck started from a brain (old ducks say "zero" or start with "zero")
def _started_from_brain(story: dict) -> bool:
    return not story["starts_from"].lower().startswith("zero")


# The card's recipe lines
def _recipe_lines(recipe: dict, started_from_brain: bool = False) -> list[str]:
    reward, ppo = recipe["reward"], recipe["ppo"]
    low, high = recipe["target_speed_range"]
    # Runs started from another brain keep that brain's settings, so say so on the card
    started_line = [STARTED_FROM_BRAIN_LINE] if started_from_brain else []
    return [
        f"- Brain: {recipe['brain']}, actor layers {recipe['layers']['pi']}, judge layers {recipe['layers']['vf']}, "
        f"starting wobble log_std {recipe['start_log_std']}",
        *started_line,
        f"- Reward per step: {_forward_wording(recipe)}, alive {reward['alive']}, "
        f"energy {reward['energy']} per joule, sideways {reward['sideways']} per metre",
        f"- Target speed: random from {low} to {high} m/s per attempt; attempts last {recipe['attempt_seconds']:.0f} s",
        _floor_line(recipe),
        _start_line(recipe),
        f"- PPO: " + ", ".join(f"{key} {value}" for key, value in ppo.items()),
        _step_size_line(recipe),
        f"- Code version: {recipe['code_version']}",
    ]


# The forward reward: the overshoot rule if the recipe has one, else the older cap rule
def _forward_wording(recipe: dict) -> str:
    rule = recipe.get("forward_reward")
    if rule is None:
        return f"forward {recipe['reward']['forward']} per m/s up to the target speed"
    return f"forward {rule} (speed over the target loses points too)"


# The floor: flat for older recipes, otherwise every number of the cobblestone shape
def _floor_line(recipe: dict) -> str:
    floor = recipe.get("floor")
    if floor is None:
        return "- Floor: flat"
    if floor["max_roughness"] == 0:
        return "- Floor: flat in training (training roughness 0; the cobblestone floor is kept for tests). The exam is flat only"
    low, high = floor["wavelength_range"]
    stone_low, stone_high = floor["stone_height_range"]
    return (f"- Floor: random cobblestones up to {floor['max_roughness']} m high, drawn at every reset with a random "
            f"roughness in training: {floor['waves']} waves of {low} to {high} m wavelength, stones {floor['stone_size']} m wide "
            f"with tops {stone_low} to {stone_high} of the tallest stone, gaps {floor['gap_share']} of the stone size on each side, "
            f"bump weight {floor['bump_share']} (the stones get the rest)" + _rough_exam_words(floor))


# How the exam's rough part chose its floor heights. Recipes from before Duck 3.1 used the full height
def _rough_exam_words(floor: dict) -> str:
    if floor.get("rough_exam") == "spread":
        return (". The exam matches training: 24 attempts with floor heights spread evenly from flat up to the full "
                "height, scored in a lower and a taller half")
    if floor.get("rough_exam") == "climbing":
        return ". The exam's rough part climbs: its 12 attempts are spread evenly up to the full height"
    return ". The exam's rough part uses the full height for all 12 attempts"


# How each attempt starts: the random nudge, and the stride stance if the recipe has one
def _start_line(recipe: dict) -> str:
    nudge = f"- Start: standing pose, each joint nudged randomly by up to {recipe['start_noise']} rad"
    stride = recipe.get("start_stride")
    if stride is None:
        return nudge + ", no stride stance"
    return nudge + f", then a stride stance: a coin flip picks the leading leg, one hip moves {stride} rad forward and the other {stride} rad back"


# How the learning rate behaves: fixed (older ducks) or adapted after every update
def _step_size_line(recipe: dict) -> str:
    rule = recipe.get("step_size_rule")
    if rule is None:
        return "- Step size: fixed at the PPO learning rate for the whole run"
    return (f"- Step size: adaptive, aiming for a change of {rule['target_change']} per update "
            f"(divided or multiplied by {rule['factor']}, kept between {rule['min_learning_rate']} "
            f"and {rule['max_learning_rate']}); the PPO learning rate is only the starting value")


# One gait line: speed, energy per metre and foot use
def _gait_line(label: str, gait: dict) -> str:
    return (f"- {label}: speed {gait['speed']} m/s, {gait['joules_per_metre']} J per metre, "
            f"feet in the air {gait['feet_in_air']:.0%}, one foot down {gait['one_foot_down']:.0%}, "
            f"both down {gait['both_feet_down']:.0%}")


# The exam line: old results show one 12 attempt score, newer ones the combined score and both parts
# (flat and rough before Duck 3.1's second run, lower and taller stones from it)
def _exam_line(results: dict) -> str:
    if "exam_lower_score" in results:
        attempts = _exam_attempts(results)
        tallest = results["rough_floor_height"] * 100
        return (f"- Exam ({attempts} fixed attempts, floor heights spread from flat to {tallest:g} cm, no wobble): "
                f"score {results['exam_score']:.0f}, falls {results['exam_falls']} of {attempts} "
                f"(lower stones {results['exam_lower_score']:.0f}, taller stones {results['exam_taller_score']:.0f})")
    if "exam_flat_score" not in results:
        return f"- Exam (12 fixed attempts, no wobble): score {results['exam_score']:.0f}, falls {results['exam_falls']} of 12"
    return (f"- Exam (24 fixed attempts, 12 flat and 12 rough, no wobble): score {results['exam_score']:.0f}, "
            f"falls {results['exam_falls']} of 24 (flat {results['exam_flat_score']:.0f}, rough {results['exam_rough_score']:.0f})")


# The card's lines about the kept brain; the rough floor lines appear only when measured
def _results_lines(results: dict, folder_name: str) -> list[str]:
    if not results:
        return ["No brain kept yet."]
    film_name = f"{folder_name}.mp4"
    lines = [
        f"Film: [{film_name}]({film_name}), {FILM_SECONDS:.0f} s of the best brain at {EVALUATION_TARGET_SPEED} m/s. "
        f"It is not kept in git; make it with `python -m duck.catalogue film duck_catalogue/{folder_name}`.",
        "",
        f"- From `{results['from_run']}` at {results['steps']:,} steps (measured {results['measured']})",
        _exam_line(results),
        f"- Walking test: {results['walking_test']} attempts walked",
    ]
    rough = _rough_floor_words(results)
    if "walking_test_rough" in results:
        lines.append(f"- Walking test on {rough}: {results['walking_test_rough']} attempts walked")
    lines.append(_gait_line(f"At {EVALUATION_TARGET_SPEED} m/s", results["gait_at_judging_speed"]))
    if "gait_on_rough_floor" in results:
        lines.append(_gait_line(f"At {EVALUATION_TARGET_SPEED} m/s on {rough}", results["gait_on_rough_floor"]))
    return lines


# The rough floor the walking test and gait used. Results from before Duck 3.1 do not say, and used 1 cm
def _rough_floor_words(results: dict) -> str:
    height = results.get("rough_floor_height", 0.01)
    return f"the rough floor ({height * 100:g} cm stones)"


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
        "## Recipe", "", *_recipe_lines(story["recipe"], _started_from_brain(story)), "",
        "## Training history", "", *_run_lines(story["runs"]), "",
        "## Best brain", "", *_results_lines(results, duck_dir.name), "",
        "## Notes", "", story["notes"] or "(none)", "",
    ]
    card_path = duck_dir / CARD_NAME
    card_path.write_text("\n".join(lines), encoding="utf-8")
    return card_path


# One line of the list page
def _index_line(duck_dir: Path) -> str:
    story = _read_json(duck_dir / STORY_NAME)
    results = _read_json(duck_dir / RESULTS_NAME)
    score = f"{results['exam_score']:.0f} ({results['exam_falls']} falls of {_exam_attempts(results)})" if results else "none yet"
    return f"| [{story['name']}]({duck_dir.name}/{CARD_NAME}) | {story['started'][:10]} | {story['summary']} | {score} |"


# Write duck_catalogue/README.md with one line per duck, oldest first
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
    film = commands.add_parser("film", help="make the film of the duck's kept brain again")
    film.add_argument("duck_dir", type=Path)
    commands.add_parser("cards", help="rewrite all cards and the list page")
    args = parser.parse_args()

    if args.command == "new":
        new_duck(args.duck_dir, args.name)
    elif args.command == "keep":
        snapshot = choose_snapshot_or_exit(args.run_dir, args.snapshot)
        keep_brain(args.duck_dir, args.run_dir, snapshot, force=args.force)
    elif args.command == "film":
        film_duck(args.duck_dir)
    rewrite_all_cards(args.duck_dir.parent if hasattr(args, "duck_dir") else CATALOGUE_DIR)


if __name__ == "__main__":
    main()
