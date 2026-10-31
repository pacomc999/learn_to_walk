# Duck 4

**Duck 3.1's brain in a body with every motor 10 times stronger, on cobblestones up to 4 cm**

Started 2026-10-31 10:05. Starts from: Duck 3.1's kept brain (duck_catalogue/duck_03_1/brain, exam 1206 in the stones_up_to_4cm world).

- World: stones_up_to_4cm, Cobblestones from flat up to 4 cm, walking speeds 0.2 to 0.7 m/s
- Body: ten_times_stronger, Every motor 10 times stronger: 10 times the force limit and 10 times the stiffness

## What changed and why

One change from Duck 3.1: the body. Every motor has 10 times the force limit and 10 times the stiffness: hips and knees 40 (from 4), ankles 30 (from 3), neck 10 (from 1), toes 5 (from 0.5); stiffness 200 for hips, knees and ankles (from 20), 40 for the neck (from 4) and 50 for the toes (from 5). The world is the same as Duck 3.1's second run: stones from flat up to 4 cm, walking speeds 0.2 to 0.7 m/s.

Duck 3.1 still falls on about two thirds of 4 cm floors, and Francisco wanted to see what the duck is capable of with much stronger muscles.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Started from another brain: it keeps that brain's wobble, step size and PPO settings, so the starting values above apply only to runs from zero
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Floor: random cobblestones up to 0.04 m high, drawn at every reset with a random roughness in training: 8 waves of 0.2 to 0.5 m wavelength, stones 0.06 m wide with tops 0.35 to 1.0 of the tallest stone, gaps 0.07 of the stone size on each side, bump weight 0.5 (the stones get the rest). The exam matches training: 24 attempts with floor heights spread evenly from flat up to the full height, scored in a lower and a taller half
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: f90f501

## Training history

- `runs/duck_04/from_duck_03_1` (, ):  (its brain came from the standard body)

## Best brain

No brain kept yet.

## Notes

Duck 3.1's brain falls 10 of 10 times on flat ground in this body before training: its small commands now kick hard. Exam scores compare directly with Duck 3.1's second run (same world, same exam).
