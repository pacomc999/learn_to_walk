# Duck 3.2

**Duck 3.1's brain on a flat floor, asked to run at up to 1.5 m/s**

Started 2026-10-30 16:27. Starts from: Duck 3.1's best brain (duck_catalogue/duck_03_1/brain, 2.25M steps, exam 981 under Duck 3.1's exam).

- World: flat_running, Flat floor, speeds 0.2 to 1.5 m/s, for learning to run
- Body: standard, The duck as designed in duck/assets/duck.xml

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

- `runs/duck_03_2/from_duck_03_1` (2026-10-30 16:28 to 17:37, 0 to 4.0M): From Duck 3.1's brain, with the first version of the flat exam, whose 12 attempts each drew a random speed from their seed: they happened to ask for 0.20 to 1.00 m/s, 0.50 on average, nothing faster. Started at 886 with 4 falls of 12 (Duck 3.1 dove forward above about 0.8 m/s); 1,220 at 0.25M, best 1,356 with 1 fall at 3.75M. Its best brain started run 2. About 1,000 steps per second
- `runs/duck_03_2/from_run_1` (2026-10-30 17:38 to 19:04, 0 to 4.0M): From run 1's best brain, with the exam's speeds spread evenly from 0.20 to 1.50 m/s (the same range as training, 0.85 on average; the best possible score is about 2,750). Started at 2,291 with no falls, then new bests at almost every exam, a few points at a time: 2,348 at 1.0M, 2,388 at 2.5M and the best, 2,391 with no falls (87% of the possible), at 2.75M. One weak exam (2,048, 2 falls) at 3.25M; 2,387 at the end

## Best brain

Film: [duck_03_2.mp4](duck_03_2.mp4), 16 s of the best brain at 0.4 m/s. It is not kept in git; make it with `python -m duck.catalogue film duck_catalogue/duck_03_2`.

- From `runs/duck_03_2/from_run_1/best` at 2,752,512 steps (measured 2026-10-30 19:06)
- Exam (12 fixed attempts, no wobble): score 2391, falls 0 of 12
- Walking test: 10 of 10 attempts walked
- At 0.4 m/s: speed 0.38 m/s, 12.0 J per metre, feet in the air 0%, one foot down 97%, both down 3%

## Notes

Exam scores of Duck 3.2 cannot be compared with earlier ducks: the exam is flat only and asks for faster speeds, which can earn more forward points. Duck 3.2's first exam, at step 0, is Duck 3.1's brain on Duck 3.2's exam. A started run inherits the source brain's wobble, adapted step size and PPO settings, so the starting wobble and learning rate in the recipe above do not apply to this run. The kept brain runs: on the flat floor, asked for 0.4 m/s it walks (one foot always down, 2.8 steps a second); from 0.8 m/s both feet are off the ground part of the time: 34% at 0.8, 44% at 1.0, 49% at 1.2 and 54% at 1.5 m/s, where it reaches 1.42 m/s at about 5.4 steps a second. Nobody told it to run: the switch from walking to running appeared by itself, between 0.4 and 0.8 m/s. The exam's speeds were spread evenly across the training range after run 1, because the random speeds of the first exam never asked for running.
