# Worlds and bodies

Date: 2026-10-30. Status: design, approved for planning once Francisco has read it.

## Why

A duck is made of three things: its **brain**, its **body** and the **world** it learns in. Brains already have a home: the catalogue keeps every duck's best brain, its recipe and its results. Bodies and worlds do not. The body is the one file `duck/assets/duck.xml`, and the world is a handful of shared settings in the code (`TRAINING_ROUGHNESS` in `duck/floor.py`, `TARGET_SPEED_RANGE` in `duck/duck_env.py`, the reward weights in `duck/rewards.py`).

Changing one of those changes it for every duck. On 2026-10-30, training Duck 3.1 on 4 cm stones meant switching the shared settings away from Duck 3.2's values, so training or keeping Duck 3.2 now would silently use the wrong floor, speeds and exam. Francisco then wanted to try a duck with much stronger legs, which today would change the body of every duck, old ones included.

Francisco mostly trains new ducks in new worlds and bodies, and will mostly ask Claude to make the worlds and bodies. So the files must be easy to read, and training a new combination must never need a code edit.

## Goals

- Worlds and bodies become named files, like ducks are named folders.
- A duck names its world and body. Training, the exam, the dashboard, films and the card all follow from that, with no shared settings to switch.
- Old ducks keep behaving exactly as they did: running Duck 3.2 means Duck 3.2's world and body.
- Any brain can be tested in another world or body with the same body shape (for example Duck 3.1's brain on flat ground, or in a stronger body).
- The reward belongs to the world (Francisco's choice): a world says where the duck is and what it is asked to do.

## Non-goals

- Changing the shape of the body: longer legs, new joints or new senses. Those need geometry work and, for joints and senses, a new brain. A later project.
- Making the strong legged duck itself. That is the first use of this design, done after it.
- Reproducing old exams exactly (Duck 3's full height rough part, Duck 3.1's first climbing exam). Their logs stay readable; new exams follow the world.
- Training under the forward reward of Ducks 1, 2 and 2.1 (before the overshoot rule). Their worlds record it as a note; only the current rule can be trained.
- Changing the floor shape numbers (`FLOOR_SHAPE`), the attempt length (20 s) or the physics step. They stay in the code for now.
- New kinds of ground (slopes, steps) and new extras beyond rocks and a backpack. The world format leaves room for them.
- Rewriting the challenges as worlds. They could be later (a challenge is a set of worlds a duck is tested in), but they work as they are.
- Giving the duck a sense of the extras (he cannot feel the backpack's weight or see rocks coming, as today).
- Updating the 3D joint explorer page for other bodies. It keeps showing the standard body.

## Anti-goals (what would make this a failure even if it works)

- An old duck's results change because a world or body file was edited later.
- Francisco has to edit Python to try a new world or body.
- Old runs, logs, cards or brains stop opening.

## The three parts

### Worlds: `worlds/<name>.json`

A world is everything around the duck and everything asked of him: the ground, what happens to him (thrown rocks, a load to carry) and the task (speeds and reward). One small JSON file per world, kept in git. Examples:

```json
{
  "name": "stones_up_to_4cm",
  "description": "Cobblestones from flat up to 4 cm, walking speeds 0.2 to 0.7 m/s",
  "floor": {"tallest_stones": 0.04},
  "target_speed_range": [0.2, 0.7],
  "reward": {"forward": 3.0, "alive": 0.2, "energy": 1.5, "sideways": 0.5},
  "extras": {},
  "notes": ""
}
```

```json
{
  "name": "stones_up_to_2cm_with_rocks",
  "description": "Cobblestones up to 2 cm, and a 100 g rock thrown at him every 2 s",
  "floor": {"tallest_stones": 0.02},
  "target_speed_range": [0.2, 0.7],
  "reward": {"forward": 3.0, "alive": 0.2, "energy": 1.5, "sideways": 0.5},
  "extras": {"rocks": {"mass": 0.1, "every_seconds": 2.0, "speed": 4.0}},
  "notes": ""
}
```

- `floor.tallest_stones` in metres: each training attempt draws a height evenly from 0 up to it. 0 means the plain flat plane. `floor` is a section so other ground (slopes, steps) can be added later.
- `target_speed_range`: each training attempt asks for a random speed in this range.
- `reward`: the four weights of `RewardWeights`. The forward rule is the current one (most points at the target speed).
- `extras`: optional things that happen in the world. Built in this project, reusing today's challenge code:
  - `rocks`: a stone ball thrown at the body (`duck/rocks.py`): `mass` in kg, `every_seconds`, `speed` in m/s; the first throw, the random side, the aim and the parking after 1 s work as in the rocks challenge today. Works on the flat plane and on the stone floor.
  - `backpack`: a box fixed on the back (`BackpackWorld` in `duck/challenges.py`): `mass` in kg. It is in the world, not the body, because it is something the duck is asked to carry, like a task; the body is the duck himself.
  - An unknown extra is refused with a plain message. More extras (wind, pushes, slopes) can be added later as new keys, each with its own small module.
- `notes`: free text, for example why the world exists.

The exam follows from the world: the same floor rule as today (12 flat attempts when `floor.tallest_stones` is 0, otherwise the stone exam: 24 attempts spread from flat to `tallest_stones`, lower and taller halves), speeds spread over `target_speed_range`, and the world's extras switched on in every exam attempt, each attempt's seed fixing where the rocks come from.

### Bodies: `bodies/<name>.json`

One small JSON file per body, describing only how it differs from the standard duck (`duck/assets/duck.xml`, which stays untouched). Example:

```json
{
  "name": "strong_legs",
  "description": "Hip and knee motors twice as strong",
  "motor_force_limits": {"hips": 8, "knees": 8},
  "notes": ""
}
```

- The standard body is `bodies/standard.json` with no changes.
- What a body can change, to start with:
  - `motor_force_limits`: the most force of a motor group, replacing `forcerange` (stays symmetric, `-x x`)
  - `motor_stiffness`: `kp` of a motor group
  - `part_masses`: mass in kg of named bodies (torso, thighs, and so on), for heavier or lighter parts
  - `foot_friction`: the sliding friction of the feet and toes
- Motor groups, so a body file stays short: `neck` (neck_base, neck_top), `hips` (both hip_swing and hip_side), `knees`, `ankles`, `toes`. Single motor names are also accepted.
- A body is applied by editing the standard XML text (for example with `xml.etree.ElementTree`) before MuJoCo loads it. Every place that reads `DUCK_XML_PATH` today (the flat model, the floor model, the backpack and rock worlds) builds from the body's XML instead.
- Because a body only changes numbers, every body has the same 12 motors, joints and senses, so every brain fits every body. A body file that names an unknown motor, group or part is refused with a plain message.

### Ducks: the catalogue, as now

`duck.json` gains two names and two frozen copies:

```json
"world": "stones_up_to_4cm",
"body": "standard",
"world_settings": { ...a copy of the world file when the duck was made... },
"body_settings": { ...a copy of the body file... }
```

The frozen copies are what training and testing use, never the files in `worlds/` and `bodies/`. So editing a world file later can never change an old duck (anti-goal 1). To change a duck's world, make a new duck, or a new run with a note, as now.

`python -m duck.catalogue new` takes `--world NAME --body NAME` (body defaults to `standard`) and freezes both. The card shows the world and body names and their descriptions next to the recipe.

## Runs

A run folder gets `run_settings.json` when it is created: the duck folder it belongs to and frozen copies of its world and body. Resuming reads it, so a resumed run always continues in its own world.

Training becomes:

`.venv/Scripts/python -m duck.train --duck duck_catalogue/duck_04 --run runs/duck_04/from_duck_03_1 --steps 4000000 --start-from duck_catalogue/duck_03_1/brain`

`--duck` is needed for a new run; the world and body come from it. Resume needs only `--run` (and reads `run_settings.json`). Starting from another duck's brain in a different body is allowed (every brain fits every body); the card's "starts from" line says which body the brain came from.

## Passing world and body through the code

- `duck/worlds.py` (new): `World` (a frozen dataclass), `load_world(name)`, reading and checking the file; `duck/bodies.py` (new): `Body`, `load_body(name)`, `body_xml(body) -> str`.
- `DuckWalkEnv(world=..., body=..., ...)`: draws stones from `world.tallest_stones`, speeds from `world.target_speed_range`, rewards from `world.reward`, and loads the body's XML. `World` and `Body` are small dataclasses, so they pass to `SubprocVecEnv` workers.
- The shared settings `TRAINING_ROUGHNESS`, `TARGET_SPEED_RANGE` and the default `RewardWeights()` stop being read by training, the exam, the catalogue and the dashboard. `MAX_ROUGHNESS` stays as the floor model's height (0.04 m, raised by tests that need taller stones, as `TallStoneWorld` does now), so floors stay exactly as they are today.
- `take_full_exam(model, vec_normalize, world, body)`.
- The world's extras are added to the model when the environment loads it (`add_rock`, `add_backpack` already edit the XML text, so they apply after the body and floor). The rock's throwing schedule moves from `RockWorld` into the environment, switched on by the world, so `RockWorld` becomes a thin wrapper or goes away.
- Challenges keep their own floors, speeds, rocks and backpacks (that is the challenge), but use the duck's body, and turn energy points into joules with the world's energy weight instead of the default `RewardWeights()`. The drive tool keeps its flat floor and speed slider and uses the duck's body.
- The tools (`evaluate`, `view_snapshot`, `video`, `drive`, `challenges`, `catalogue film` and `keep`, `dashboard`) find the world and body from the folder they are given: a duck folder's `duck.json`, a run folder's `run_settings.json` (or the `best/` folder's parent). Each also takes `--world NAME` and `--body NAME` to try another combination. A folder with no settings (runs from before this change) uses `standard` and asks for `--world`, with a plain message.

## Keeping a brain

`keep` today compares a new brain's exam score with the kept brain's stored score, which is wrong when the exam changed (Duck 3.1's kept 981 came from the old exam). `results.json` records the world name the exam used. If the kept brain was examined in another world, `keep` gives the kept brain the current exam first and compares like with like, and says so in its message.

## Moving the existing ducks over

Write these worlds, each from the recipes in the ducks' `duck.json` files:

All with no extras.

| World | floor.tallest_stones | speeds | Used by |
|---|---|---|---|
| `flat_walking` | 0 | 0.2 to 0.7 | Duck 1, 2, 2.1 (notes: they trained under the older forward reward, without the overshoot rule) |
| `stones_up_to_1cm` | 0.01 | 0.2 to 0.7 | Duck 3 |
| `stones_up_to_3cm` | 0.03 | 0.2 to 0.7 | Duck 3.1, first run |
| `stones_up_to_4cm` | 0.04 | 0.2 to 0.7 | Duck 3.1, second run (`from_duck_03_1`) and from now on |
| `flat_running` | 0 | 0.2 to 1.5 | Duck 3.2 |

All with reward 3.0, 0.2, 1.5, 0.5, and all ducks with the `standard` body. Each `duck.json` gets its names and frozen copies; each existing run folder gets a `run_settings.json` where its world is known. Duck 3.1's `duck.json` names `stones_up_to_4cm` (its newest run); its card says the first run used `stones_up_to_3cm`. After the move, every duck's walking test on seeds 0 to 9, flat and on its own stones, must give the same result as the code just before the move (not the cards: raising the floor model to 4 cm already moved Duck 3.1's 3 cm test from 7 to 6 of 10). Record those results before starting, then compare.

## Testing

Only the test files of the changed modules are run (Francisco's rule, the full suite is too slow). New tests: loading and checking world and body files; a world with rocks throws them on both floors and a world with a backpack carries it, in training and in the exam; a body's XML has the changed numbers and is refused for unknown names; the environment uses the world's stones, speeds and reward; `run_settings.json` is written and used on resume; `keep` re-examines a kept brain from another world; the tools pick the world from a duck folder and a run folder.

## Documents

Update `docs/guide.md` (a new "Worlds and bodies" section, the new training command), the README table, `CLAUDE.md` (layout, the end of shared per duck settings) and `duck_catalogue/README.md` through the cards.

## Decided by Claude (Francisco can overrule)

- JSON files in `worlds/` and `bodies/` at the project root, kept in git.
- Rocks and a backpack as the first two world extras (Francisco: thrown stones are a property of the world); the backpack counts as world, not body.
- Bodies as changes from the standard duck, with motor groups, rather than full XML copies.
- Bodies change numbers only (strength, stiffness, masses, foot grip) in this project.
- Frozen copies in `duck.json` and `run_settings.json` are what gets used; the files are templates.
- `--duck` on training for new runs; `--world` and `--body` on the test tools for trying combinations.
- Ducks 1, 2 and 2.1 get `flat_walking` with the current forward rule and a note, rather than code for the old rule.
- `keep` re-examines the kept brain when the exam's world differs.

## Left to the builder

- The exact helper names, and how the tools find the settings from a `best/` folder.
- Which body parts `part_masses` accepts (the bodies in `duck.xml`).
- The wording of messages and card lines (no dashes, as always).
- The order of work. A sensible one: worlds and bodies with the environment; then training, the exam and runs; then the catalogue and `keep`; then the tools and dashboard; then moving the existing ducks; then the documents.

## Before starting

Today's uncommitted work (the stone exam, the dashboard started by training) is committed first, and Duck 3.1's second run is compared on new seeds and kept or not, so this project starts from a clean state.
