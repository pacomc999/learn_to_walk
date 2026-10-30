# Duck 3.2

**Duck 3.1's brain on a flat floor, asked to run at up to 1.5 m/s**

Started 2026-10-30 16:27. Starts from: Duck 3.1's best brain (duck_catalogue/duck_03_1/brain, 2.25M steps, exam 981 under Duck 3.1's exam).

## What changed and why

Three changes from Duck 3.1. The speeds: each training attempt asks for a random speed from 0.2 to 1.5 m/s instead of 0.2 to 0.7. The floor: training is flat again (training roughness 0; the cobblestone floor is kept for later ducks and the tests). The exam: only the flat part, 12 attempts, with asked speeds spread over the new range.

Asked for more than it trained for, Duck 3.1 walked up to about 0.74 m/s and dove forward and fell from 0.9 m/s on. The goal is a duck that runs as fast as it can, up to 1.5 m/s. Francisco chose to keep the slow speeds in training so one brain covers every pace, and a flat exam so the best brain is picked on what the duck trains for. A flat floor also roughly doubles the training speed. Running probably needs a new gait with both feet off the ground at times, which none of the ducks has yet.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Started from another brain: it keeps that brain's wobble, step size and PPO settings, so the starting values above apply only to runs from zero
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 1.5 m/s per attempt; attempts last 20 s
- Floor: flat in training (training roughness 0; the cobblestone floor is kept for tests). The exam is flat only
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: e162b47

## Training history

No runs yet.

## Best brain

No brain kept yet.

## Notes

Exam scores of Duck 3.2 cannot be compared with earlier ducks: the exam is flat only and asks for faster speeds, which can earn more forward points. Duck 3.2's first exam, at step 0, is Duck 3.1's brain on Duck 3.2's exam. A started run inherits the source brain's wobble, adapted step size and PPO settings, so the starting wobble and learning rate in the recipe above do not apply to this run.
