# Duck 4

**Force motors: the brain sets each joint's force (up to 10 times the standard limits), loose at 0; learning to walk from zero on a flat floor**

Started 2026-10-31 22:51. Starts from: Nothing: a new brain, since old brains send angles, not forces.

- World: flat_walking, Flat floor, walking speeds 0.2 to 0.7 m/s
- Body: force_motors, Force motors with 10 times the standard force limits: the brain sets each joint's force, loose at 0

## What changed and why

The muscles. Every duck before had position motors, which pull each joint toward the angle the brain asks for like a spring. Duck 4 has force motors (the force_motors body): the brain's command is the force itself, from minus to plus the force limit, with the same limits as Duck 3.3 (10 times the standard). A command of 0 leaves the joint loose. Starting wobble 0.05 (log_std -3.0).

Duck 3.3's 10 times stiffer position motors made his legs rigid: asked for 2 m/s he ran about 1.5 m/s, and with free energy about 1.6 m/s, then stopped improving. Francisco wanted muscles that are not stiff but can push hard when the brain wants. The wobble was measured on random moves of a new brain (10 attempts each): with force motors a wobble of 0.05 costs 1.3 J per step, less than the standard duck's start (2.8 J at 0.37), so energy should not make falling on purpose pay, the trap Duck 3.3 fell into at first.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -3.0
- Started from another brain: it keeps that brain's wobble, step size and PPO settings, so the starting values above apply only to runs from zero
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Floor: flat in training (training roughness 0; the cobblestone floor is kept for tests). The exam is flat only
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: 861ce13

## Training history

- `runs/duck_04/from_zero` (, ): 

## Best brain

No brain kept yet.

## Notes

Exam and walking scores can be compared with the walking ducks in flat_walking. With no commands the duck collapses in about 0.4 s, so he has to learn to hold himself up first.
