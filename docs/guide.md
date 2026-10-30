# Guide

Everything about using the project, from how the duck learns to how a new duck is made, trained, kept and tested. The README has the short version; notes for working on the code are in `CLAUDE.md`.

All commands run from the project folder and start with `.venv/Scripts/python`. Type them in PowerShell or Git Bash; the old cmd.exe window does not understand the forward slashes.

## How the duck learns

### The duck

A 3D duck in the MuJoCo physics engine (`duck/assets/duck.xml`): a long body, legs set far back, a bendy neck, a head and beak, folded wings and webbed feet. It weighs 1.53 kg and stands 25 cm tall. It has 12 motors: two in the neck, and in each leg a hip swing, a hip side, a knee, an ankle and toes. The wings are fixed to the body. The motors are position motors with limited strength, so falling and struggling are real. Open `docs/duck_joints.html` in a browser to drag each joint and see its range and strength.

### Its senses

Before every move the duck senses 39 numbers: its height, how far it has drifted sideways, its tilt, how fast it moves and turns, the angle and speed of each of its 12 joints, whether each foot touches the ground, and the speed it is asked to walk at. It is blind: it feels the floor only through its feet and joints.

### Its brain

A small neural network (two layers of 256 cells) chooses the 12 motor targets 50 times per second. It learns by trial and error with PPO (reinforcement learning, from the Stable-Baselines3 library). While learning it adds a random wobble to its moves to try new things; at the exam and when it is filmed, the wobble is off.

The brain is left/right symmetric (the "NET" method of Abdolhosseini and others, 2019): it always answers a mirrored situation with the mirrored command, so the duck learns to walk once instead of once for each leg. The mirror table is in `duck/mirror.py` and the brain in `duck/symmetric_policy.py`.

### The floor

From Duck 3 on, every training attempt gets a freshly drawn cobblestone floor: rolling bumps with square 6 cm stones on top, at a random height from flat up to the most (1 cm for Duck 3, 3 cm from Duck 3.1 on). A height of exactly 0 is the old flat floor. The floor code is in `duck/floor.py`.

### How it is scored

Every attempt asks for a random walking speed from 0.2 to 0.7 m/s. On each of the 50 steps per second the duck earns:

- **Forward:** 3 points per m/s of forward speed, at most at the asked speed. Going faster loses points the same way as going slower: the reward is 3 x (target - |speed - target|).
- **Staying up:** 0.2 points.
- **Energy:** minus 1.5 points for every joule its motors spend. This is what makes stepping win over hopping or lunging.
- **Drift:** minus 0.5 points per metre away from the straight line.

An attempt ends after 20 seconds, or earlier if the duck falls (its body drops below 60% of its standing height or tips more than about 50 degrees). Standing still for 20 seconds scores only about 200.

### The exam

Training scores jump around because of the wobble and the random speeds, so training also gives the brain a fixed, noise-free exam: at the start, every 250,000 steps and at the end. It has two parts of 12 attempts each, always the same attempts (seeds 1000 to 1011, each with its own asked speed):

- **Flat:** 12 attempts on the flat floor.
- **Rough:** the same 12 attempts on cobblestones, climbing evenly up to the most height (at 3 cm: 0.25 cm, 0.5 cm, and so on up to 3 cm). Duck 3's exam used the full 1 cm for all 12.

The exam score is the average of the two parts; the falls are added up, out of 24. The best possible score is about 1,150, which would need walking at exactly the asked speed with no energy at all. A realistic best for this duck is around 1,045.

### What counts as walking

The walking test (`duck.evaluate`) asks for 0.4 m/s in 10 attempts of 20 seconds. The duck walks if at least 8 attempts finish without falling, at an average speed of at least 0.2 m/s, and within 0.5 m of the straight line.

## Making a new duck

A duck is a training recipe (the rules, the brain and the settings), not a single brain. A new duck starts only when the recipe changes; training longer is the same duck. The catalogue in `duck_catalogue/` holds every duck: its story, its frozen recipe, its best brain and its card. `duck_catalogue/README.md` lists them all.

1. **Change the recipe in the code** and commit it (the recipe records the code version).
2. **Start the duck** in the catalogue, which freezes the current recipe:

   `.venv/Scripts/python -m duck.catalogue new duck_catalogue/duck_04 --name "Duck 4"`

3. **Write its story** in `duck_catalogue/duck_04/duck.json`: the summary, what changed, why, and where it starts from.
4. **Train it**, from zero or from another duck's brain (see Training). Runs of a duck go in `runs/duck_04/<run name>/`, with the training output saved as `train.log` inside the run folder.
5. **Follow it** on the dashboard (see Watching and judging).
6. **Keep its best brain**: this measures the brain (exam, walking test and gait on both floors), copies it into the catalogue only if its exam beats the kept one, makes its film and rewrites the cards:

   `.venv/Scripts/python -m duck.catalogue keep duck_catalogue/duck_04 runs/duck_04/from_duck_03_1/best`

7. **Finish the story**: add the run and what happened to `duck.json`, then rewrite the cards:

   `.venv/Scripts/python -m duck.catalogue cards`

Each kept duck has a 16 second film next to its card (`duck_catalogue/duck_03/duck_03.mp4`, 1280 x 960). Films are not kept in git; make one again from the kept brain with:

`.venv/Scripts/python -m duck.catalogue film duck_catalogue/duck_03`

## Training

Train a new run from zero for 4 million steps:

`.venv/Scripts/python -m duck.train --run runs/duck_04/from_zero --steps 4000000`

Train a new run starting from another duck's kept brain (its brain and sense scaling; the new run counts its steps from 0):

`.venv/Scripts/python -m duck.train --run runs/duck_04/from_duck_03_1 --steps 4000000 --start-from duck_catalogue/duck_03_1/brain`

Continue a stopped run. The number of steps is the total for the whole run, so a bigger number trains on past the first goal:

`.venv/Scripts/python -m duck.train --run runs/duck_04/from_zero --steps 6000000 --resume`

Good to know:

- **Stopping is safe.** Ctrl+C, a sleep or a reboot only loses the steps since the last saved brain; `--resume` picks up from there. Training keeps the laptop awake while it runs, so keep it plugged in.
- **A used run folder is refused** for a new run: use `--resume`, or pick a new folder name.
- **Saved brains** land in `<run>/snapshots/` once every million steps (change with `--snapshot-every`), and the best brain by exam in `<run>/best/`, replacing the previous best. `<run>/best` works like a run folder in every tool. Step counts land on rollout boundaries, for example 1,007,616 instead of 1,000,000.
- **Training stops by itself** after 8 exams in a row (2 million steps) without a new best. Change that with `--patience`, or turn it off with `--patience 0`. Change how often the exam runs with `--exam-every`.
- **The step size adapts by itself**: after each brain update, if the brain changed more than twice the target amount the learning rate is divided by 1.5, and if it changed less than half the target it is multiplied by 1.5. Without this, Duck 1's updates grew to 50 to 100 times the size PPO is designed for and the duck fell apart.
- **The logs** in the run folder: `exam.csv` (every exam, with the flat and rough parts), `progress.csv` (training score and distance), `training_stats.csv` (PPO's own numbers after each update: how far the brain moved, the wobble, the step size), `train.log` (what training printed).
- **Speed:** with 8 duck worlds side by side, about 1,000 steps per second on the flat floor (Duck 2.1) and about 650 to 750 on the cobblestones, which cost about twice the physics. 4 million steps take about 1.5 to 2 hours.

### When the score stops improving

If `duck.progress` says there has been no improvement for an hour:

1. Watch the latest brain first (dashboard, live viewer or video) to see what the duck is actually doing.
2. Then resume with more curiosity, the amount of random experimenting: add `--resume --ent-coef 0.01` to the training command.

The stall clock counts real time, so after resuming a run that was stopped for a while, wait a full hour before trusting the stall message. If more curiosity does not help, the design (`docs/designs/2026-10-28-duck-walker-design.md`) lists what to try next: fix a reward that rewards a bad habit, add training wheels, or go back to an earlier brain. Changing the duck's body comes last.

## Watching and judging

Follow training live in the browser. In a second terminal, while training runs (`--steps` should match the training command):

`.venv/Scripts/python -m duck.dashboard runs/duck_04/from_zero --steps 4000000`

It opens http://127.0.0.1:8765/ and refreshes itself: steps trained, time left, distance per attempt, whether the score is still improving, score and distance charts, and a 20 second film of the best brain so far on the roughest floor, with its exam score. Buttons slow the film down; a click pauses it. Stopping the dashboard does not affect training.

The other tools take a run folder; the walking test, the live viewer and the video also take `<run>/best` or a duck's `brain` folder:

| What | Command |
|---|---|
| Training chart and the stall check | `.venv/Scripts/python -m duck.progress runs/duck_04/from_zero` |
| The walking test | `.venv/Scripts/python -m duck.evaluate runs/duck_04/from_zero/best` |
| Watch live in a window | `.venv/Scripts/python -m duck.view_snapshot runs/duck_04/from_zero/best` |
| A film of the whole run, brain by brain | `.venv/Scripts/python -m duck.video runs/duck_04/from_zero` (saved as `evolution.mp4` in the run folder) |

They use the newest brain in the folder; add `--snapshot` with a step count for another one, for example `--snapshot 1007616`. They use the flat floor unless you add `--roughness` with the stone height in metres, for example `--roughness 0.03`. The live viewer also takes `--speed`, for example `--speed 0.6`.

## Testing a kept duck

### Challenges

Challenges put a duck's kept brain in situations it never trained for. They are tests, not training: the brain does not change. Run one or more for a duck:

`.venv/Scripts/python -m duck.challenges duck_catalogue/duck_03_1 speed backpack stones rocks`

| Challenge | What it asks |
|---|---|
| `speed` | Walk at one speed from 0.4 up to 1.5 m/s on the flat floor; training asked for 0.2 to 0.7 |
| `backpack` | Carry a box of 100 to 800 g on its back, on the flat floor at 0.4 m/s |
| `stones` | Walk on cobblestones of one height, from 1 up to 8 cm, at 0.4 m/s (30 attempts per height) |
| `rocks` | Keep walking while a stone ball of 25 to 400 g is thrown at its body every 2 seconds |

Each challenge saves a table, for example `duck_catalogue/duck_03_1/challenges/speed/results.csv`: for each value, how many attempts walked the full 20 seconds, how long it stayed up on average, its actual speed and its energy per metre. The same command always gives the same numbers, so ducks can be compared. `challenges/README.md` is rewritten from all of a duck's tables after every run; remarks written by hand go in `challenges/notes.md` and are copied into it.

Add `--films` to also film attempt 1 of every value of the challenges you run, kept in each challenge's own folder with its own page (`challenges/speed/films/index.html`), and one page for the duck that plays every challenge's films in sections (`challenges/index.html`). Films are not kept in git. Add `--film 150` to the rocks challenge to also film all 10 attempts with 150 g rocks (`challenges/rocks/films/round_150g/`), with a page that plays them side by side (`challenges/rocks/films/index.html`, not kept in git). Change the number of attempts with `--attempts`.

### The joints

`docs/duck_joints.html` shows the duck in 3D: drag its legs, feet, toes and neck to see how far each joint turns and how strong each motor is. Its numbers were copied from `duck/assets/duck.xml`, so it needs a new export if the body changes.

## Where things are

| Folder | What is in it | In git |
|---|---|---|
| `duck/` | The code: the duck's world, training, exam, tools | Yes |
| `tests/` | One test file per code file; run them all with `.venv/Scripts/python -m pytest` (about 25 minutes) | Yes |
| `duck_catalogue/` | One folder per duck: `duck.json` (story and recipe), `brain/` (its best brain), `results.json`, `card.md`, its film, and `challenges/` with the results of the challenges it took | Yes, except the films |
| `runs/duck_NN/<run>/` | Each duck's training runs: saved brains, `best/`, logs | No |
| `runs/early_experiments/` | The runs from the first day, before the catalogue; its `README.md` explains each one | Only the README |
| `docs/designs/`, `docs/plans/` | The design notes and build plans | Yes |

### Keeping it tidy

A finished run's brains can be slimmed, which empties each brain's optimizer memory (two thirds of its size):

`.venv/Scripts/python -m duck.snapshots slim runs/duck_04`

The brains still load and walk exactly the same; only a run that is still training needs that memory to carry on exactly where it stopped. The catalogue slims the brains it keeps by itself.

## Ideas for later

- **Falling less:** measure the real fall rate per stone height with many more attempts; give the brain a sense of the ground just ahead of its feet; or count a fall as a direct cost.
- **More challenges:** slopes, a slippery floor, wind, tired motors, changing the asked speed during a walk.
- **Pushes in training:** throw rocks during training so the duck learns to catch a shove, which none of the ducks met so far.
- **Obstacles and steering:** going around things needs the duck to turn.
- **Moving wings,** trained on top of the walking brain.
