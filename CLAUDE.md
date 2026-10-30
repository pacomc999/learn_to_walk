# CLAUDE.md

A 3D MuJoCo duck (design B, 12 motors) that learns to walk with PPO on a CPU-only Windows laptop. See `README.md` for usage and `docs/superpowers/specs/2026-10-28-duck-walker-design.md` for the design.

## Layout
- `duck/model.py`: loads `duck/assets/duck.xml`, body constants (standing height, `MIN_TORSO_HEIGHT` = 0.6 of standing height)
- `duck/floor.py`: the random cobblestone floor: `floor_heights` (rolling bumps from 8 cosine waves plus square stones, scaled to the roughness), `MAX_ROUGHNESS` (0.01 m) and `FLOOR_SHAPE` (all shape numbers), and the MuJoCo height field model
- `duck/rewards.py`: reward parts and the fall rule
- `duck/duck_env.py`: Gymnasium environment `DuckWalkEnv` (frame skip 4, 20 s attempts, 39 number observation: torso height, sideways position, orientation, velocities, 12 joint angles, 12 joint velocities, 2 foot contacts where toes count as foot, target speed)
- `duck/mirror.py`: the mirror table (reflection through the vertical plane along the walking line, y becomes -y): torso height and target speed stay, sideways position flips, quaternion (w, x, y, z) becomes (w, -x, y, -z), linear velocity (vx, vy, vz) becomes (vx, -vy, vz), angular velocity becomes (-wx, wy, -wz), neck joints stay, left and right leg joints swap and both hip_side values flip sign, foot contacts swap. Actions use `ACTION_ORDER = [0, 1, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6]` and `ACTION_SIGNS = [1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1]`. Also `mirror_observation`, `mirror_action` and `MirroredRunningMeanStd` (mirrored senses share their scaling)
- `duck/symmetric_policy.py`: `SymmetricPolicy`, the left/right symmetric brain (same 256 x 256 sizes). `MirrorExtractor.actor_cells` gives the 12 commands plus both hidden layers of the pass on the duck's real view (the brain's own forward pass uses it)
- `duck/train.py`: PPO training, snapshot and progress callbacks, resume. `--start-from FOLDER` starts a new run from another run's or duck's brain (weights and sense scaling). New runs use `SymmetricPolicy` and mirrored sense scaling; resumed old runs keep their plain `MlpPolicy` and plain sense scaling
- `duck/exam.py`: the noise-free exam in two parts (flat floor, then rough floor at `MAX_ROUGHNESS`; 12 attempts each, 24 in all; `take_full_exam`; the combined score is the average of the two part scores and the falls are summed) (per part: 12 attempts, seeds 1000 to 1011, random target speed, deterministic actions, the training sense scaling without updating it). `ExamCallback` in `train.py` takes it at the start, every `--exam-every` steps (default 250,000) and at the end, logs `<run>/exam.csv` (`time,timesteps,score,falls,distance` plus extra columns for the flat and rough parts) and saves a brain that beats the best logged score to `<run>/best/snapshots/` (opens like a run folder). Matches the scratchpad peak check: 1.61M symmetric scores 484, 7 falls. About 12 s per exam. After `--patience` exams in a row (default 8, 0 = never) without a new best, `_on_step` returns False and training stops early, still saving the final snapshot. The count starts at 0 in every session (a resume does not read it from the log)
- `duck/training_stats.py`: `TrainingStatsCallback` writes SB3's numbers of each brain update (`approx_kl, clip_fraction, explained_variance, entropy_loss, value_loss, std, learning_rate`) to `<run>/training_stats.csv`, read from `model.logger.name_to_value` at rollout start and training end (the logger is dumped before `train()`, so the latest update's values are still there)
- Adaptive step size (from Duck 2, 2026-10-29): `AdaptiveStepSizeCallback` in `train.py` reads `train/approx_kl` after each update and sets `model.learning_rate` for the next one with `adjusted_learning_rate` and `STEP_SIZE_RULE` (target change 0.01, factor 1.5, limits 1e-5 and 1e-2; Rudin and others 2021 defaults), then calls `model._setup_lr_schedule()`. It applies to every training session, resumed old runs included. The adapted rate is saved in snapshots. Reason: Duck 1's updates were 50 to 100 times bigger than PPO is designed for (approx_kl 0.5 to 1.1, clip_fraction above 0.8) and grew as the wobble shrank
- `duck/snapshots.py`: save, list and pick snapshot files
- `duck/progress.py`: progress log (`time,timesteps,mean_reward,mean_distance`), stall check, chart with score and distance panels. Broken rows are skipped; a locked log only prints a warning
- `duck/policy.py`: load a snapshot as a walking policy; it handles both the symmetric and the old plain brain
- `duck/evaluate.py`, `duck/view_snapshot.py`, `duck/video.py`: judge, watch live, make the video (all take `--roughness METRES`, default 0 = flat floor)
- `duck/dashboard.py` + `duck/dashboard.html`: live browser dashboard (standard library web server on 127.0.0.1:8765, `/status.json`, `/latest.mp4`); films 20 s at 1280 x 960 (`FILM_WIDTH`, `FILM_HEIGHT` in `duck_env.py`; `duck.xml` makes room for that size) of the best brain so far on the rough floor at full roughness, on a fixed rough layout from seed 1000 (not the exam's first layout, because the film fixes the speed), captioned with its combined exam score and both parts from `exam.csv` (`<run>/best/snapshots/`, or the newest snapshot before the first exam; regular snapshots come only once a million steps) to `<run>/dashboard_latest.mp4` in the main thread; `/latest.mp4` answers Range requests. The brain view (senses, cell grids, motor bars) was removed on 2026-10-30: it cost little but showed nothing the film and the exams do not. Measured motor directions, kept for reference: positive hip swing moves the foot back, positive ankle points it down. The server runs in a background thread. The page is vanilla JS with hand drawn SVG charts
- `duck/catalogue.py` + `ducks/` (kept in git): the duck catalogue. A duck is a training recipe, not a brain; a new duck starts only when the recipe changes. `ducks/duck_NN/duck.json` holds the story (summary, what changed, why, starts from, runs, notes) and the recipe frozen from the code at creation (`current_recipe`: `NEW_RUN_POLICY`, `NET_ARCH`, `DEFAULT_PPO_SETTINGS`, reward weights, git version). `keep` measures a snapshot (two part exam, walking test and gait at 0.4 m/s on both floors, flat and rough at full roughness) and replaces the kept brain in `brain/snapshots/` only if its exam score is higher (`--force` overrides), makes a captioned 16 s MP4 film (`<folder>.mp4`, git-ignored; `film DUCK_DIR` makes it again from the kept brain) and rewrites `card.md` and `ducks/README.md`. Edit `duck.json` by hand for the story, then `python -m duck.catalogue cards`. New runs of a duck go in `runs/duck_NN/...`
- `tests/`: one test file per module
- `runs/`: training output, git-ignored

## Commands
Run from the project root with the virtual environment:
- Tests: `.venv/Scripts/python -m pytest`
- Anything using `SubprocVecEnv` must be under `if __name__ == "__main__":`

## Motor order (12, fixed)
`neck_base, neck_top, left_hip_swing, left_hip_side, left_knee, left_ankle, left_toes, right_hip_swing, right_hip_side, right_knee, right_ankle, right_toes`
Position motors with limited force. Wings are fixed to the body.

## Reward (per step) and fall rules
- Target speed: each attempt picks one at random from `TARGET_SPEED_RANGE` (0.2 to 0.7 m/s); `DuckWalkEnv(target_speed=...)` fixes it. Evaluate and video use `EVALUATION_TARGET_SPEED` (0.4), the viewer takes `--speed`
- Forward: +3.0 per metre per second of forward speed at most, and the most at the target speed. Going faster or slower than the target loses the same points per metre per second off the top (at target 0.4, 0.5 m/s scores the same as 0.3 m/s). With no target speed the points just grow with speed. Speed is the distance moved over the control step (0.02 s) divided by that time, not a single velocity sample
- Alive: +0.2 per step (so standing still for 20 s scores only about 200)
- Energy: -1.5 per joule of motor work, summed over the 4 physics substeps of each control step as |actuator force x actuator velocity| x timestep. Replaced the old squared action penalty, which did not measure energy. At 1.5 with no speed target the duck hopped; at 3.0 it dove forward and fell after 0.9 s
- Sideways: -0.5 per metre of drift from the straight line
- Fall: torso height below `MIN_TORSO_HEIGHT`, or `up_z` below 0.64 (about 50 degrees of tilt)
- An attempt also ends after 20 simulated seconds
- Floor: drawn after the stride at every reset; roughness is random 0 to 1 cm in training (`DuckWalkEnv(roughness=None)`), evaluation tools pass it explicitly; the torso is lifted over the stones at the start; a roughness of exactly 0 keeps the old flat plane
- Start: the 'stand' keyframe, each joint nudged by up to `START_NOISE` (0.02 rad), then (from Duck 2.1) a stride stance: a coin flip picks the leading leg and the hip swings move `START_STRIDE` (2 x `START_NOISE`) apart each way. Reason: from a near mirror image pose the symmetric brain sends mirror image commands and dives forward (Duck 2 deterministic: 34 of 60 early falls); twice the nudge means the nudge can never close the gap
- Walking means: 8 of 10 attempts of 20 s, average speed at least 0.2 m/s, within 0.5 m of the line

## Snapshots
In `<run>/snapshots/`: `snapshot_<12 digit steps>.zip` (the brain) and `snapshot_<12 digit steps>_norm.pkl` (the statistics). Files are written under a temp name then renamed, statistics last, so a snapshot is complete only when both final files exist. Resume uses the newest complete one; if it fails to load it is renamed with a `.broken` ending and the one before it is used. Renames retry 5 times, 0.2 s apart, if Windows has the file locked. A new brain starts with `log_std_init` -1. Step counts land on rollout boundaries, not round numbers. The video uses at most 25 snapshots.

## Code style
Beginner readable: clear names, short single purpose functions, a comment on each section. Python 3.12 only, no TypeScript.

## Owner's rules
- No em or en dashes in any visible text (docs, captions, printed messages, chart labels)
- Commit messages: short, present tense, no prefix, for example `Add duck model`
