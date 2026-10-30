# Duck 2

**Duck 1 plus an adaptive step size, trained from zero**

Started 2026-10-29 16:53. Starts from: zero (same seed and untrained brain as Duck 1).

- World: flat_walking, Flat floor, walking speeds 0.2 to 0.7 m/s
- Body: standard, The duck as designed in duck/assets/duck.xml

## What changed and why

Same recipe as Duck 1, except the step size: after every brain update the learning rate is divided by 1.5 if the brain changed more than twice the target (0.01), or multiplied by 1.5 if it changed less than half of it (Rudin and others 2021).

Duck 1 collapsed after its best at 1.61M. A measurement run showed every update changing the brain 50 to 100 times more than PPO is designed for (approx_kl 0.5 to 1.1, safety limit hit by over 80% of moves), growing as the wobble shrank. Trained from zero with the same seed as Duck 1, so the two recipes can be compared fairly.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Reward per step: forward 3.0 per m/s up to the target speed, alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Floor: flat
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, no stride stance
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: 22a13ba

## Training history

- `runs/duck_02/from_zero` (2026-10-29 16:55 to 18:01, 0 to 4.0M): Slower start than Duck 1 (23 against 99 at 1.0M), then a jump to 395 at 1.5M and the best exam at 2.0M (557, 6 falls of 12). No collapse: after one dip to 151 at 2.25M it held between 381 and 448 up to 4.0M, where Duck 1 had fallen to 136. Each update changed the brain by about 0.015 (Duck 1: 0.5 to 1.1), the safety limit was hit by 20 to 24% of moves, and the learning rate fell from 0.0003 to about 0.00001

## Best brain

Film: [duck_02.mp4](duck_02.mp4), 16 s of the best brain at 0.4 m/s. It is not kept in git; make it with `python -m duck.catalogue film duck_catalogue/duck_02`.

- From `runs/duck_02/from_zero/best` at 2,007,040 steps (measured 2026-10-29 18:02)
- Exam (12 fixed attempts, no wobble): score 557, falls 6 of 12
- Walking test: 3 of 10 attempts walked
- At 0.4 m/s: speed 0.48 m/s, 17.3 J per metre, feet in the air 3%, one foot down 33%, both down 64%

## Notes

Walks with alternating steps and about 96% one foot down once it is moving, using 15 to 18 J per metre, about half of Duck 1. Its weak point is the start: with some starting poses (a difference of about one degree per joint) it dives forward and falls within 0.6 s. The step size kept falling all run, so learning slowed down; a higher target change or a different rule could be the next recipe.
