# Duck 3.3 the Strong

**A body with every motor 10 times stronger, trained from zero on a flat floor**

Started 2026-10-31 10:50. Starts from: zero.

- World: flat_walking, Flat floor, walking speeds 0.2 to 0.7 m/s
- Body: ten_times_stronger, Every motor 10 times stronger: 10 times the force limit and 10 times the stiffness

## What changed and why

Two changes from Duck 3.1. The body: every motor has 10 times the force limit and 10 times the stiffness (hips and knees 40 from 4, ankles 30 from 3, neck 10 from 1, toes 5 from 0.5; stiffness 200 for hips, knees and ankles from 20, 40 for the neck from 4, 50 for the toes from 5). The brain: a new one trained from zero, not Duck 3.1's. The world is the flat floor with walking speeds 0.2 to 0.7 m/s and today's reward (flat_walking), so the exam is 12 flat attempts.

Francisco wanted to see what the duck is capable of with much stronger muscles. Raising only the force limits would have kept Duck 3.1's walk but given at most about 5 times the force in normal moves; raising the stiffness too gives the full 10 times, and Duck 3.1's brain fell every time in that body (its small commands turned into hard kicks), so this brain learns the body from the start.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Floor: flat in training (training roughness 0; the cobblestone floor is kept for tests). The exam is flat only
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: f90f501

## Training history

- `runs/duck_03_3/from_zero` (, ): 

## Best brain

No brain kept yet.

## Notes

The closest duck to compare with is Duck 2.1, also trained from zero on the flat floor at the same speeds, but under an older forward reward without the overshoot rule, so the scores are close but not identical in meaning. Before training, Duck 3.1's brain fell 10 of 10 times on flat ground in this body.
