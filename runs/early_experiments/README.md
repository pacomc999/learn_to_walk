# Early experiments

The training runs from the first day, 28 to 29 October 2026, before the duck catalogue (`ducks/`) and before the exam existed. Each one tried a change to the rewards or the duck, and what it taught led to the next one. Duck 1 (`ducks/duck_01/`) is the first catalogue duck; everything here came before it.

These brains are the old plain kind (not the symmetric brain), trained without an exam, so the scores below are training scores (with the random wobble), not exam scores. The brains were slimmed on 2026-10-30 (no optimizer memory): they load and act as before but cannot continue training exactly where they stopped.

## The runs, in order

| Run | When | Steps | What changed | What happened |
|---|---|---|---|---|
| `pilot/` | 28 Oct 17:30 to 18:05 | 3.0M | The first run: points for forward distance, a small bonus for staying up, a small penalty on large motor commands | Best training score 558 at 2.0M, about 2.5 m per attempt at the end. The penalty on commands did not measure real energy |
| `pilot_energy/` | 18:49 to 19:26 | 3.0M | The duck pays for real motor energy: 1.5 points per joule | Best 328. With no speed to aim for, the cheapest way forward was to hop |
| `pilot_energy3/` | 21:22 to 21:57 | 3.0M | The energy price doubled to 3 points per joule | Best 54, about 0.5 m per attempt: too expensive to walk, it dove forward and fell within a second. The price went back to 1.5 |
| `pilot_target_speed/` | 22:19 to 22:49 | 2.5M, stopped | Each attempt asks for a random speed from 0.2 to 0.7 m/s, and the duck senses that speed (its 39th sense) | Best 472. Stopped early when a bug was found: the duck read its joint angles in the wrong unit |
| `pilot_knees/` | 22:49 to 23:24 | 3.0M | The joint angle fix ("Read duck joint angles as radians"), with the target speed | Best 737 at 1.9M and about 8.3 m per attempt: the first run that clearly walked |
| `night1/` | 28 Oct 23:24 to 29 Oct 04:03 | 3.0M to 23.0M | `pilot_knees` continued overnight from its 3M brain | Peaked early and then got worse all night: 89 at the end, falling in most attempts. Only its text records are kept |
| `night2/` | 29 Oct, until 08:19 | 4.0M to 24.0M | `night1` continued in a second copy from its 4M brain | Best 756 at 4.4M, then also got worse: 210 at the end. Only its text records are kept |
| `best_4M/` | saved 29 Oct 08:20 | 4.0M | Not a run: the 4M brain where `night2` split from `night1`, close to the best moment of both nights, with a GIF | The best brain of the early days. It walked, but hopped with the right leg always leading |

## What they taught

- Real energy works as a cost, at 1.5 points per joule; at 3 the duck gives up walking.
- A target speed gives the duck a reason to walk at a steady pace instead of hopping or lunging.
- Training longer did not help: both nights peaked within the first few million steps and then unlearned. The brain updates turned out to be far too large, which Duck 2 fixed with an adaptive step size.
- The best brain always led with the same leg, which led to the symmetric brain of Duck 1.

## What is in each folder

- `snapshots/`: the saved brains, `snapshot_<steps>.zip` with its sense scaling `snapshot_<steps>_norm.pkl`, one about every 250,000 steps (the overnight runs' brains were deleted on 2026-10-30)
- `progress.csv`: the training score and distance over time; `progress.png` is a chart of it (the first two runs only)
- `train.log`: what training printed (from `pilot_energy3` on)
- `peak_check.txt` (the nights only): a later check of their saved brains, with and without the wobble, to find their best moments
- Films: `evolution.mp4` (the run from start to end), `latest_duck.gif` and other GIFs, `dashboard_latest.mp4` (the dashboard's last film)

The first three runs sensed 38 numbers, before the target speed was added, so their brains do not load into today's duck. The others have the 39 senses of today's duck but older rewards and the old plain brain.
