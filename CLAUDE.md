# CLAUDE.md

Walk Studio: a no-code web app around the learn_to_walk duck. MuJoCo physics, PPO from Stable-Baselines3, a local Python web server (standard library only) and a plain JavaScript page with three.js. Users never see a file; everything is a slider, switch or choice. See `README.md` for the user view.

## Layout

- `studio/creatures/kit.py`: `Setting`/`SettingGroup` (the page draws every control from these), `clean_values`, the finished `Build` (body XML, actuator XML, `Motor`s with mirror partner and sign, `Foot`s, stand pose, stride pairs, `symmetric_ok`, `motor_type`) and XML helpers
- `studio/creatures/duck.py`: with every setting at its default and `motor_type: position` it builds exactly learn_to_walk's `duck/assets/duck.xml` (tested). Sizes scale parts, masses follow volume, strength and springiness scale with size to the fourth power, direct (force) muscles get `FORCE_MOTOR_BOOST` 10 times the force limits (learn_to_walk's `bodies/force_motors.json`)
- `studio/creatures/puppy.py`: a quadruped, 14 motors (neck, tail wag with mirror sign -1, 3 per leg)
- `studio/creatures/custom.py`: build your own. `values["body"]` = `{"torso": {...}, "parts": [...]}`; each part has a parent, shape (stick, ball, plate), size, attach point (x/y/z shares of the parent's half sizes, or `along` for stick parents), direction (pitch down, yaw left, degrees), joint (fixed, swing = axis y, side = axis x, twist = axis z, hip = swing + side), motion (whole range, degrees), strength (N m), mirror (a twin with its whole subtree on the other side), foot, heaviness (% of water density), colour. Body frames are never rotated, so mirror signs are swing +1, side -1, twist -1. Motor order: middle, left, right in the same order. An unpaired part off the middle line makes `symmetric_ok` False and `brain.design_for_body` falls back to the free brain. `TEMPLATES` are the ready-made bodies
- `studio/creatures/__init__.py`: `ANIMALS`, `build_creature(key, values)`
- `studio/worlds.py`: world settings (ground, physics, task, reward with the switchable joint limit penalty `end_stops` + `end_stop_cost`, extras, looks), `World`, `builtin_worlds()` from `studio/world_templates/*.json` (values only list what differs from the defaults)
- `studio/terrain.py`: height maps: cobblestones (learn_to_walk's floor), hills, stairs, craters, blocks; `floor_layout` sizes the floor for the fastest speed; `capacity` (stairs pile up to `MOST_STEPS`)
- `studio/scene.py`: composes the MuJoCo file (floor plane or height field, gravity, grip, backpack, rock) and measures the stand height from the shapes. Floor friction is tiny and the walker's shapes carry grip x world grip (MuJoCo takes the larger friction of a pair)
- `studio/env.py`: `WalkEnv(build, world, senses)`. Observation: height above the ground under the torso, sideways position, orientation, velocities, joint angles, joint speeds, foot contacts (optional), ground scan ahead (optional, `GROUND_SCAN`), asked speed. Reward as learn_to_walk (speed, alive, energy, straight, end stops). Falls: torso below 0.6 of stand height above the ground, or tilted past 50 degrees. Gradually harder worlds read `difficulty`, set by training
- `studio/mirror.py`: mirror tables from the build (tested against the physics: a mirrored pose gives mirrored positions and senses), `MirroredRunningMeanStd`
- `studio/brain.py`: brain design (layers, judge layers, activation, symmetric, senses, starting wobble; `auto` wobble is -3.0 for direct muscles as Duck 4, -1.0 for springy), `SymmetricPolicy` (learn_to_walk's NET method made general: any depth, any activation, middle motors combine both views through the mirror signs), `policy_for`
- `studio/learning.py`: training settings in plain words, `ppo_settings`
- `studio/train.py`: `python -m studio.train FOLDER [--resume]`. Writes `status.json`, `progress.csv`, `exams.csv`, `brain/{best,latest}.zip` with `_scaling.pkl` (VecNormalize, for resuming) and `_norm.npz` (for acting), `replays/best.json` and `replays/exam_<steps>.json` (at most 30, thinned evenly), `log.txt`. Stops at `stop.request`. Adaptive step size as learn_to_walk. Exam: 8 attempts, speeds and heights spread evenly; quality = share of the asked distance covered
- `studio/replay.py`: 3D recordings for the page (shapes once, then 25 frames a second of every body's position and turn, plus a coarse ground patch); `holder` keeps a pose for drop tests (direct muscles act as a spring of the force limit per radian)
- `studio/manager.py`: starts training processes, pauses them, detects interrupted ones (pid check), reads everything back for the page, remembers measured speed in `library/system.json`
- `studio/previews.py`: creature preview, drop test, brain facts, watching a trained brain in any world
- `studio/library.py`: the `library/` folder (git-ignored; `STUDIO_LIBRARY` overrides it)
- `studio/server.py`: `ThreadingHTTPServer` on 127.0.0.1:8770, JSON API under `/api/` (catalog, preview/{creature,drop,brain}, creatures, worlds, brains, trainings and their stop/continue/rename/watch/replay)
- `studio/web/`: `index.html`, `css/studio.css` (tokens with a dark theme), `js/app.js` (state, router, sidebar), `js/ui.js` (elements, setting controls, toasts, dialogs), `js/viewer.js` (three.js, z up, playback, picking, highlighting), `js/charts.js` (SVG charts), `js/pages/*` (home, wizard, builder, network, walker, walkers, library, guide). `vendor/` holds three.js 0.160.0 and the Inter font so it works offline
- Launchers: `Start Walk Studio.bat` (Windows, CRLF), `Start Walk Studio.command` (macOS), `start_studio.sh`

## Commands

- Run: `python -m studio` (add `--no-browser`, `--port`)
- Tests: `python -m pytest` (about 30 seconds; `tests/test_train.py` runs two tiny real training sessions)

## Owner's rules (from learn_to_walk)

- Beginner readable code: clear names, short functions, a comment on each section
- No em or en dashes in any visible text
- Commit messages: short, present tense, no prefix
- Do not run long trainings in the cloud sessions: short smoke tests only
