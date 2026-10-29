# Duck 2.1

**Duck 2 plus a stride stance start, trained from zero**

Started 2026-10-29 23:06. Starts from: zero (same seed and untrained brain as Duck 1 and Duck 2).

## What changed and why

Same recipe as Duck 2, except the start of each attempt: after the usual random nudge, a coin flip picks the leading leg and one hip swing starts 0.04 rad forward and the other 0.04 rad back (each twice the random nudge, so the nudge can never close the gap).

Duck 2 without its wobble fell in the first second in 34 of 60 starts. A check showed why: from a near mirror image pose the symmetric brain sends mirror image leg commands and dives forward, and a perfectly even pose always fell. On Duck 2's best brain, untrained for it, the stride start cut early falls to 2 of 60 and walked 30 of 30 attempts at 0.4 m/s. Keeping the wobble instead was rejected: it doubles the energy per metre (38 against 18 J). A point version because the change is small.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Reward per step: forward 3.0 per m/s up to the target speed, alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: d483a71

## Training history

- `runs/duck_02_1/from_zero` (, 0 to 4.0M): running

## Best brain

No brain kept yet.

## Notes

(none)
