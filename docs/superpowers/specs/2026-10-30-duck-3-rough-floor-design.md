# Duck 3: rough floor and overshoot penalty

Date: 2026-10-30. Owner: Francisco. Builds on Duck 2.1 (`ducks/duck_02_1/`).

## Intent and why

Duck 2.1 walks on flat ground about as well as our scoring can tell: its best exam is 981 against a realistic ceiling of about 980 (the ceiling at 18 J per metre; the energy free ceiling is 1,150). The next goal is a duck that keeps walking when the ground is not flat, starting with a cobblestone kind of floor. Obstacles may come later and are not part of Duck 3.

Duck 3 also gets the overshoot penalty from the literature review (agreed earlier, kept for later until now): being faster than the asked speed loses points the same way as being slower. Duck 2.1 still walks about 7.4 m per exam attempt where the asked speeds would give 6.3 m.

Francisco chose to make both changes in one duck and to start from Duck 2.1's best brain. He accepted that this mixes two changes, so we cannot tell which one helped or hurt.

## Decisions Francisco made

- **Floor:** square cobblestones laid on rolling bumps, in one floor (the sketch in the scratchpad pictures `combined_*.png`, built from `floor_previews.py` and `floor_combined.py`).
- **Start:** from Duck 2.1's best brain, not from zero.
- **Overshoot penalty:** in Duck 3, together with the floor.
- **Best brain:** chosen by the flat exam and the rough exam together, so a brain that gets worse on flat ground cannot win.
- **Name:** Duck 3, a whole number because the world changes a lot.

## Goals

1. A duck that walks on the rough floor at the asked speed without falling.
2. It still walks on flat ground (the flat exam stays and counts half).
3. It no longer runs ahead of the asked speed.

## Non goals

- Obstacles, slopes, stairs, pushes. Later ducks.
- Eyes or a terrain map: the duck stays blind and feels the floor only through its feet and joints, as now. The 39 number observation does not change.
- Changing the brain, the PPO settings, the step size rule, the energy weight or the stride start.

## Anti goals (would be a failure even if it technically works)

- Many new hand tuned numbers. Francisco does not want walking to come from tailoring small parameters. Every new number must be listed in the recipe and on the card, and there should be as few as possible.
- A duck that only walks on rough ground and forgets flat ground.
- Exam scores that cannot be compared from one exam to the next (the rough exam must be repeatable).

## The floor

- A MuJoCo height field replaces the plane. The floor geom keeps the name `floor`, because foot contact sensing (`duck/model.py`, `_touches_floor`) matches that name.
- Shape, as in the sketch:
  - Rolling bumps: the sum of 8 cosine waves, each with a random direction, a random wavelength from 0.2 to 0.5 m and a random phase.
  - Square cobblestones: a grid of 6 cm stones (about one duck foot), each with a flat top at a random height from 0.35 to 1.0 of the stone height, with gaps of 7% of the stone size on each side at height 0.
  - Each part is scaled to 0 to 1 and the two are added with half weight each, then scaled to the attempt's roughness.
- **Roughness:** each training attempt picks one at random from 0 to `MAX_ROUGHNESS` = 0.01 m (1 cm), like the target speed. About one attempt in ten is under 1 mm, so flat ground stays in the training. This is the only new number that sets difficulty; the shape numbers above are copied from the sketch.
- **A new layout every attempt**, drawn from the environment's own random generator, so a seed gives the same floor.
- **Size:** long enough for the fastest attempt (0.7 m/s for 20 s is 14 m) plus margin behind the start, and wide enough that the duck cannot walk off the side before the sideways penalty and the fall rule matter. Resolution fine enough that a 6 cm stone has clear edges (the sketch used 4 mm).
- **Start pose:** the duck must not start inside the stones. Lift the torso by the floor height under the feet before the first step.
- **Flat ground:** roughness 0 must give an exactly flat floor at height 0, so the flat exam matches the old one.
- **Rendering:** films and the dashboard must show the current layout. In MuJoCo 3.17 the renderer keeps its own copy of the height field, so after a new layout the renderer has to upload it again (`mujoco.mjr_uploadHField`), or videos show a stale floor.

## The overshoot penalty

- Forward part of the reward becomes `forward_weight * (target - |speed - target|)`.
- Below the target speed this is exactly the current reward (`forward_weight * speed`), including walking backwards. Above the target it falls by the same amount it would rise below. The maximum stays at the target speed, so the exam ceilings above stay valid.
- No new number: it uses the existing forward weight of 3.0.

## Starting from Duck 2.1's brain

- A new way to start a run from another run's brain, for example `python -m duck.train --run runs/duck_03/from_duck_02_1 --steps 4000000 --start-from <folder with a snapshots/ folder>`.
- It loads the brain and its sense scaling and keeps learning from there in the new world. It keeps what the brain learned: its wobble size, its adapted step size and its sense scaling (which keeps updating as usual).
- The new run's step count starts at 0, so its logs, exams and snapshots read like any run. The card records which brain it started from.
- The source brain: Duck 2.1's kept best brain in `ducks/duck_02_1/brain/`.

## Exams

- Every exam (at the start, every 250,000 steps, at the end) has two parts, each the current 12 attempts with deterministic actions and the training sense scaling:
  - **Flat exam:** exactly the current exam (seeds 1000 to 1011, roughness 0).
  - **Rough exam:** 12 fixed stone layouts at full roughness (1 cm), each from its own fixed seed, with the same random target speeds as the flat exam.
- **Score:** the plain average of the two scores. This is what picks the best brain and counts for the patience rule (stop after 8 exams without a new best).
- `exam.csv` keeps `score`, `falls` and `distance` as the combined values, so the dashboard, phone page and catalogue keep working, and adds the flat and rough parts as extra columns. Old logs without the extra columns must still read.

## Catalogue, dashboard, docs

- The recipe in `duck.json` records the floor (shape numbers, `MAX_ROUGHNESS`), the forward reward form and the starting brain. Each card gets a "Floor" line; older ducks say "flat".
- New duck folder `ducks/duck_03/`, name "Duck 3". Runs in `runs/duck_03/...`.
- The dashboard caption shows the combined exam score and both parts.
- README.md and the project CLAUDE.md describe the floor, the overshoot rule, `--start-from` and the two part exam.

## Training run

- 4 million steps from Duck 2.1's best brain, with the usual patience of 8 exams.

## Calls made by Claude after the builder check (Francisco to confirm)

1. **Torso height stays absolute** (height above z = 0), in the observation and in the fall rule. The floor moves the torso by at most 1 cm against a standing height of 25 cm and a fall threshold of 0.6 of that, so the effect is small, and Duck 2.1's brain keeps the same meaning for its first input.
2. **Every floor shape number goes in the recipe and on the card**, not only `MAX_ROUGHNESS`: 8 waves, wavelengths 0.2 to 0.5 m, 6 cm stones, stone heights 0.35 to 1.0, gaps 7%, half bumps and half stones. Honesty about numbers is the point of the anti goal.
3. **Duck 3 starts its own records.** The exam at step 0 (Duck 2.1's brain, new floor, new reward) is Duck 3's first entry and can be its first best brain; the patience count starts there. Duck 3 has its own empty `brain/` folder, so `keep` compares only with Duck 3's own kept brain. Duck 2.1's kept brain is never touched.
4. **The catalogue's walking test and gait summary run on both floors**: flat as now, and rough at full roughness on fixed seeds. The card shows both.
5. **Viewing tools default to flat** (`evaluate.py`, `view_snapshot.py`, `video.py`) with a `--roughness` option in metres; the dashboard films the best brain on the rough floor at full roughness, on the first rough exam layout, because that is what Duck 3 is learning. (Note 2026-10-30: built as a fixed rough layout from seed 1000; it differs from the exam's first layout because the film fixes the speed.)

## Left to the builder

- How the height field is stored and regenerated quickly enough at each reset (8 worlds, one reset per attempt of up to 1,000 steps).
- The exact floor size and resolution, within the constraints above.
- How the duck's start height is lifted over the stones.
- The command line and function names, following the code's style.
