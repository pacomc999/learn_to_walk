# CLAUDE.md

A 3D MuJoCo duck (design B, 12 motors) that learns to walk with PPO on a CPU-only Windows laptop. See `README.md` for usage and `docs/superpowers/specs/2026-10-28-duck-walker-design.md` for the design.

## Layout
- `duck/model.py`: loads `duck/assets/duck.xml`, body constants (standing height, `MIN_TORSO_HEIGHT` = 0.6 of standing height)
- `duck/rewards.py`: reward parts and the fall rule
- `duck/duck_env.py`: Gymnasium environment `DuckWalkEnv` (frame skip 4, 20 s attempts, 38 number observation: torso height, sideways position, orientation, velocities, 12 joint angles, 12 joint velocities, 2 foot contacts where toes count as foot)
- `duck/train.py`: PPO training, snapshot and progress callbacks, resume
- `duck/snapshots.py`: save, list and pick snapshot files
- `duck/progress.py`: progress log (`time,timesteps,mean_reward,mean_distance`), stall check, chart with score and distance panels. Broken rows are skipped; a locked log only prints a warning
- `duck/policy.py`: load a snapshot as a walking policy
- `duck/evaluate.py`, `duck/view_snapshot.py`, `duck/video.py`: judge, watch live, make the video
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
- Forward: +3.0 per metre per second of forward speed, no cap. Speed is the distance moved over the control step (0.02 s) divided by that time, not a single velocity sample
- Alive: +0.2 per step (so standing still for 20 s scores only about 200)
- Energy: -1.5 per joule of motor work, summed over the 4 physics substeps of each control step as |actuator force x actuator velocity| x timestep. Replaced the old squared action penalty, which did not measure energy
- Sideways: -0.5 per metre of drift from the straight line
- Fall: torso height below `MIN_TORSO_HEIGHT`, or `up_z` below 0.64 (about 50 degrees of tilt)
- An attempt also ends after 20 simulated seconds
- Walking means: 8 of 10 attempts of 20 s, average speed at least 0.2 m/s, within 0.5 m of the line

## Snapshots
In `<run>/snapshots/`: `snapshot_<12 digit steps>.zip` (the brain) and `snapshot_<12 digit steps>_norm.pkl` (the statistics). Files are written under a temp name then renamed, statistics last, so a snapshot is complete only when both final files exist. Resume uses the newest complete one; if it fails to load it is renamed with a `.broken` ending and the one before it is used. Renames retry 5 times, 0.2 s apart, if Windows has the file locked. A new brain starts with `log_std_init` -1. Step counts land on rollout boundaries, not round numbers. The video uses at most 25 snapshots.

## Code style
Beginner readable: clear names, short single purpose functions, a comment on each section. Python 3.12 only, no TypeScript.

## Owner's rules
- No em or en dashes in any visible text (docs, captions, printed messages, chart labels)
- Commit messages: short, present tense, no prefix, for example `Add duck model`
