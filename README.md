# learn_to_walk

A 3D duck in a physics simulation teaches itself to walk.

The duck has a long body, legs set far back, a bendy neck, a head and beak, folded wings and webbed feet. It has 12 motors: two in the neck, and in each leg a hip swing, a hip side, a knee, an ankle and toes. The motors have limited strength, so falling and struggling are real.

A small neural network is the duck's brain. It learns by trial and error (reinforcement learning, using a method called PPO). All the stumbling and falling happens during training. While it trains, the project saves snapshots of the brain. At the end you can make one video where each snapshot walks for a few seconds, one after another, so you can watch the duck go from flopping over to walking.

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

Continue with more curiosity (more random experimenting):

`.venv/Scripts/python -m duck.train --run runs/duck_v1 --steps 30000000 --resume --ent-coef 0.01`

Check progress (draws a chart in the run folder, prints the latest average distance per attempt, and says if training has stalled):

`.venv/Scripts/python -m duck.progress runs/duck_v1`

Judge whether the duck really walks:

`.venv/Scripts/python -m duck.evaluate runs/duck_v1`

Watch the latest snapshot live in a window:

`.venv/Scripts/python -m duck.view_snapshot runs/duck_v1`

Make the video of all the snapshots:

`.venv/Scripts/python -m duck.video runs/duck_v1`

The video is saved as `evolution.mp4` inside the run folder.

To judge or watch an older snapshot, add `--snapshot` and its step count, for example `--snapshot 1007616`.

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

On every control step (50 per second) the duck gets 3 points per metre per second of forward speed, measured from how far it really moved during that step, plus 0.2 points for staying upright. It loses 3.0 points for every joule of work its motors do (force times joint speed), and 0.5 points per metre it has drifted from the straight line. Standing still for a whole 20 second attempt scores only about 200. The energy cost is there so the duck finds walking on its own: hopping lifts the whole body again and again and wastes energy, while stepping is cheap. The duck senses 38 numbers, including how far it has drifted sideways.

## What counts as walking

The duck walks when it finishes at least 8 out of 10 attempts. Each attempt is 20 seconds without falling, with an average speed of at least 0.2 metres per second, and staying within 0.5 metres of the straight line. `duck.evaluate` checks exactly this.

## Later goals

- Moving wings, trained on top of the walking brain
- Going around obstacles, which needs steering
- Sharing the results more easily, for example on a web page

## More detail

The design is in `docs/superpowers/specs/2026-10-28-duck-walker-design.md`. Notes for working on the code are in `CLAUDE.md`.
