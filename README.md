# learn_to_walk

A 3D duck in a physics simulation teaches itself to walk.

The duck has a long body, legs set far back, a bendy neck, a head and beak, folded wings and webbed feet. It has 12 motors: two in the neck, and in each leg a hip swing, a hip side, a knee, an ankle and toes. The motors have limited strength, so falling and struggling are real.

A small neural network is the duck's brain. It learns by trial and error (reinforcement learning, using a method called PPO). All the stumbling and falling happens during training. While it trains, the project saves snapshots of the brain. At the end you can make one video where each snapshot walks for a few seconds, one after another, so you can watch the duck go from flopping over to walking.

The brain is left/right symmetric (the "NET" method of Abdolhosseini and others, 2019). Mirroring the world, swapping the left and right legs and flipping the sideways signs, gives a mirrored duck, and the brain is built so it always answers a mirrored situation with the mirrored command. The duck therefore only has to learn to walk once, not once for each leg, and the senses that belong together (a left foot and a right foot, say) share their scaling. The mirror table lives in `duck/mirror.py` and the brain in `duck/symmetric_policy.py`. Runs started before this change keep their plain brain and plain sense scaling when resumed; everything (watching, judging, the video, the dashboard) still works on them.

The floor can be rough. From Duck 3 on, every attempt in training gets a freshly drawn cobblestone floor: rolling bumps with square stones on top, a random roughness from 0 up to the most (1 cm for Duck 3, 3 cm from Duck 3.1 on). A roughness of exactly 0 keeps the old flat plane, so older ducks and tools behave as before. The floor code lives in `duck/floor.py`.

Everything runs on a CPU only Windows laptop.

## Setup

Do this once, from the project folder.

1. Make a virtual environment with Python 3.12:

   `py -3.12 -m venv .venv`

2. Install PyTorch from the CPU index:

   `.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

3. Install the other libraries:

   `.venv/Scripts/python -m pip install -r requirements.txt`

4. Check that everything works:

   `.venv/Scripts/python -m pytest`

## Commands

All commands run from the project folder, and start with `.venv/Scripts/python`. Type them in PowerShell (or Git Bash). They do not work in the old cmd.exe window, because of the forward slashes.

Start training for 30 million steps:

`.venv/Scripts/python -m duck.train --run runs/duck_v1 --steps 30000000`

Continue a stopped run:

`.venv/Scripts/python -m duck.train --run runs/duck_v1 --steps 30000000 --resume`

To keep training past 30 million steps, resume with a bigger `--steps` value. The number is the total for the whole run, not extra steps. For example, this trains on until 50 million:

`.venv/Scripts/python -m duck.train --run runs/duck_v1 --steps 50000000 --resume`

Start a new run from another run's or duck's brain instead of from zero (a new run folder is needed):

`.venv/Scripts/python -m duck.train --run runs/duck_03/from_duck_02_1 --steps 4000000 --start-from ducks/duck_02_1/brain`

Continue with more curiosity (more random experimenting):

`.venv/Scripts/python -m duck.train --run runs/duck_v1 --steps 30000000 --resume --ent-coef 0.01`

Training also gives the duck a noise-free exam every 250,000 steps (and at the start and end): the same 24 attempts each time, in two parts (12 on the flat floor and 12 on the rough floor, climbing evenly from a twelfth of the most up to the most, at 3 cm: 0.25 cm, 0.5 cm, ... 3 cm; Duck 3's exam used 1 cm for all 12; the score is the average of the two parts and the falls are added up), with the brain's best moves and no random wobble. Each result is added to `exam.csv` in the run folder (with extra columns for the flat and rough parts), and whenever a brain beats the best exam score so far it is saved in the run's `best` folder, replacing the previous best. That folder works like a run folder, so you can judge or watch the best brain with `runs/duck_v1/best`. Change how often with `--exam-every`. If 8 exams in a row (2 million steps, about 35 minutes) bring no new best score, training stops by itself, because the duck has stopped improving. The best brain is then waiting in the `best` folder. Change the limit with `--patience`, or turn it off with `--patience 0`.

The step size adapts by itself: after each brain update, if the brain changed more than twice the target amount the learning rate is divided by 1.5, and if it changed less than half the target it is multiplied by 1.5. That keeps every update about the same size as the duck gets more precise. Without it, Duck 1's updates grew to 50 to 100 times the size PPO is designed for and the duck fell apart.

PPO's own numbers after each brain update (how far the brain moved, how often the safety limit kicked in, how well the judge predicts, the wobble and the step size) go to `training_stats.csv`.

Check progress (draws a chart in the run folder, prints the latest average distance per attempt, and says if training has stalled):

`.venv/Scripts/python -m duck.progress runs/duck_v1`

Or follow training live in the browser. In a second terminal, while training runs:

`.venv/Scripts/python -m duck.dashboard runs/duck_v1 --steps 30000000`

It opens a page at http://127.0.0.1:8765/ that refreshes itself every 15 seconds: steps trained, time left, distance per attempt, whether the score is still improving, score and distance charts (hover for exact values), and a film of the best brain so far walking at 0.4 m/s on the rough cobblestone floor (the most roughness, a fixed rough layout from seed 1000 that differs from the exam's first layout), captioned with the combined exam score and both parts, filmed again each time a new best brain is saved. Buttons slow the film to half or quarter speed, and a click pauses it. `--steps` should match the training command, so it can show the percentage and time left. Press Ctrl+C to stop it; training carries on.

Judge whether the duck really walks:

`.venv/Scripts/python -m duck.evaluate runs/duck_v1`

Watch the latest snapshot live in a window:

`.venv/Scripts/python -m duck.view_snapshot runs/duck_v1`

The judge, the live viewer and the video all use a flat floor unless you add `--roughness` with the cobblestone height in metres, for example `--roughness 0.01`.

Make the video of all the snapshots:

`.venv/Scripts/python -m duck.video runs/duck_v1`

The video is saved as `evolution.mp4` inside the run folder.

To judge or watch an older snapshot, add `--snapshot` and its step count, for example `--snapshot 1007616`.

## Where things are

- `duck/`: the code, and `tests/` its tests
- `ducks/`: the catalogue, one folder per duck with its story, recipe, best brain and card (kept in git)
- `runs/`: training output (not in git): `runs/duck_NN/<run>/` for each duck's runs, with the training output in `train.log`, and `runs/early_experiments/` for the runs before the catalogue
- `docs/designs/` and `docs/plans/`: the design notes and build plans; `docs/duck_joints.html`: the joint explorer

A finished run's brains can be slimmed: `.venv/Scripts/python -m duck.snapshots slim runs/duck_03` empties each brain's optimizer memory, two thirds of its size. The brains still load and walk exactly the same; only a run that is still training needs that memory.

## The duck's joints

Open `docs/duck_joints.html` in a browser to see the duck in 3D and drag its legs, feet, toes and neck. It shows how far each of the 12 joints can turn and how strong each motor is, with the numbers from `duck/assets/duck.xml`.

## Throwing rocks at a duck

A test of balance that is not part of training: stone balls are thrown at a duck's kept brain while it walks on the flat floor, one every 2 s from a random side at 4 m/s, 10 attempts of 20 s per rock size from 25 g to 400 g:

`.venv/Scripts/python -m duck.rocks ducks/duck_03_1`

The results go to `ducks/duck_03_1/rocks/results.csv`. Add `--film 150` to also film every attempt with 150 g rocks and make a page that plays them side by side (`ducks/duck_03_1/rocks/films/index.html`, not kept in git). `ducks/duck_03_1/rocks/README.md` explains the first round and compares three ducks.

## The duck catalogue

Every training recipe is a duck, with a card in `ducks/` (see `ducks/README.md` for the list). A card says what changed from the previous duck and why, the exact rules and settings, every training run, and how its best brain does. Next to the card is a 16 s film of the best brain with the duck's name on it (`ducks/duck_03/duck_03.mp4`, 1280 x 960). Films are not kept in git; make one again with `.venv/Scripts/python -m duck.catalogue film ducks/duck_03`. A new duck starts only when the recipe changes, not when training goes on longer.

Start a new duck with the current recipe, then fill in its story in `ducks/duck_02/duck.json`:

`.venv/Scripts/python -m duck.catalogue new ducks/duck_02 --name "Duck 2"`

Measure a snapshot and keep it as the duck's best brain (only if its exam score beats the kept one):

`.venv/Scripts/python -m duck.catalogue keep ducks/duck_02 runs/duck_02/main`

Rewrite the cards after editing a `duck.json`:

`.venv/Scripts/python -m duck.catalogue cards`

## Try a short pilot run first

Before the long run, do a short test of about 3 million steps (around 30 minutes) in its own folder:

`.venv/Scripts/python -m duck.train --run runs/pilot --steps 3000000`

When it finishes:

1. Run `.venv/Scripts/python -m duck.progress runs/pilot`. Check that the average distance per attempt is rising, not just the score. The score can rise while the duck only learns to stand still; the distance shows that it moves forward.
2. Run `.venv/Scripts/python -m duck.evaluate runs/pilot`. After only 3 million steps it will most likely say "Not there yet". That is fine, this just checks that everything works.
3. Watch it once in the live viewer: `.venv/Scripts/python -m duck.view_snapshot runs/pilot`
4. Make a video once: `.venv/Scripts/python -m duck.video runs/pilot`

If all of that works, start the real run in a new folder such as `runs/duck_v1`. Do not resume the pilot folder for the real run.

## How to start the real run

1. Plug the laptop in.
2. Open a terminal in the project folder.
3. Run the start training command above.
4. Leave it running. Do not shut down the laptop or close the lid.

Training keeps the laptop awake while it runs, so keep it plugged in.

Starting a new run in a folder that already has snapshots is refused. Use `--resume`, or pick a new folder name such as `runs/duck_v2`.

Stopping with Ctrl+C is safe. Resume picks up from the newest complete snapshot, even after a sleep or a reboot. Anything after that snapshot is redone.

Snapshot step counts land on rollout boundaries, not round numbers. For example you will see 1,007,616 instead of 1,000,000. That is normal.

## Speed

On this laptop, with 8 duck worlds side by side, training runs at about 1,600 steps per second (measured: 1,602 and 1,613 on a 1 million step test). That is about 5.8 million steps per hour, so 30 million steps take roughly 5 hours. Walking may need more or fewer steps than that. Training has no time limit, so let it run as long as the score keeps improving, and check progress at least once a day.

## When the score stops improving (the stall rule)

If `duck.progress` says there has been no improvement for an hour:

1. Watch the latest snapshot first, with the live viewer or by making a video. See what the duck is actually doing.
2. Then resume with more curiosity: add `--resume --ent-coef 0.01` to the training command.

One catch: the stall clock counts real time. If you resume a run that was stopped for a while, wait until training has been going for a full hour before you trust the stall message.

If more curiosity does not help, the spec lists what to try next: fix a reward that rewards a bad habit, add training wheels, or go back to an earlier snapshot. Changing the duck body comes last.

## How the duck is scored

On every control step (50 per second) the duck gets 3 points per metre per second of forward speed, measured from how far it really moved during that step, at its best at the target speed for that attempt: the reward is 3 x (target - |speed - target|), so going faster than the target loses the same points per metre per second that going slower does (overshoot penalty), plus 0.2 points for staying upright. It loses 1.5 points for every joule of work its motors do (force times joint speed), and 0.5 points per metre it has drifted from the straight line. Standing still for a whole 20 second attempt scores only about 200. The energy cost is there so the duck finds walking on its own: hopping lifts the whole body again and again and wastes energy, while stepping is cheap. Each training attempt asks for a random target speed between 0.2 and 0.7 metres per second, and the duck senses it. Going faster than asked loses points and costs energy, so the duck has to find the cheapest gait for each speed by itself. Judging and videos ask for 0.4 m/s; the live viewer takes `--speed`, for example `--speed 0.6`. The duck senses 39 numbers, including how far it has drifted sideways and its target speed. Snapshots from before the target speed (38 numbers) no longer load.

## What counts as walking

The duck walks when it finishes at least 8 out of 10 attempts. Each attempt is 20 seconds without falling, with an average speed of at least 0.2 metres per second, and staying within 0.5 metres of the straight line. `duck.evaluate` checks exactly this.

## Later goals

- Moving wings, trained on top of the walking brain
- Going around obstacles, which needs steering
- Sharing the results more easily, for example on a web page

## More detail

The design is in `docs/designs/2026-10-28-duck-walker-design.md`. Notes for working on the code are in `CLAUDE.md`.
