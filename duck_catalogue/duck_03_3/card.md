# Duck 3.3 the Strong

**A body with every motor 10 times stronger, trained from zero on a flat floor, then sent to running school up to 2 m/s**

Started 2026-10-31 11:30. Starts from: zero.

- World: flat_running_free_energy, Flat floor, speeds 0.2 to 2.0 m/s, and motor work costs nothing
- Body: ten_times_stronger, Every motor 10 times stronger: 10 times the force limit and 10 times the stiffness

## What changed and why

Three changes from Duck 3.1. The body: every motor has 10 times the force limit and 10 times the stiffness (hips and knees 40 from 4, ankles 30 from 3, neck 10 from 1, toes 5 from 0.5; stiffness 200 for hips, knees and ankles from 20, 40 for the neck from 4, 50 for the toes from 5). The brain: a new one trained from zero, not Duck 3.1's, starting with a 10 times smaller wobble (log_std -3.3, a spread of 0.037 instead of 0.37). The world is the flat floor with walking speeds 0.2 to 0.7 m/s and today's reward (flat_walking), so the exam is 12 flat attempts.

Francisco wanted to see what the duck is capable of with much stronger muscles. Raising only the force limits would have kept Duck 3.1's walk but given at most about 5 times the force in normal moves; raising the stiffness too gives the full 10 times, and Duck 3.1's brain fell every time in that body (its small commands turned into hard kicks), so this brain learns the body from the start.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -3.3
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 0.0 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 2.0 m/s per attempt; attempts last 20 s
- Floor: flat in training (training roughness 0; the cobblestone floor is kept for tests). The exam is flat only
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: f90f501

## Training history

- `runs/duck_03_3/from_zero` (2026-10-31 10:55 to 12:10, 0 to 6.5M (4M, then continued toward 8M and stopped by hand once his walk was steady)): Started with the 10 times smaller wobble (log_std -3.3). Every exam was a new best, but it climbed slowly: all 12 attempts fell up to 3.5M (score 124), as against 1.5M for Duck 2.1. Early on his attempts got shorter (1.1 s untrained, 0.23 s at 0.5M, with negative forward points), then longer again with forward points rising (0.53 s and +23 at 1.5M); by 2.25M his feet landed ahead of each other about twice per attempt. The jump came at the end of the first 4M: 346 with 11 falls at 3.75M and 655 with 7 falls at 4.0M. Continued from 4M: 1094 with 0 falls at 5.0M (above Duck 2.1's best of 986), then 0 falls in every exam, reaching 1188 at 6.5M, where it was stopped to move on to running. The wobble shrank by itself from 0.037 to about 0.014. About 1,000 to 1,100 steps per second.
- `runs/duck_03_3/from_run_1` (2026-10-31 12:20 to 21:40, 0 to 12.75M (planned 4M; stopped by low memory at 1.5M and resumed from 1.0M, then extended to 8M, 16M, and resumed once more; stopped early at 12.75M after 8 exams without a new best)): From his walking brain (exam 1188 in flat_walking). Running school exam: 455 with 8 falls at the start, 1,471 at 2.0M, 1,958 at 4.0M, the first exam with no falls at 6.5M (2,286), 2,401 with no falls at 8.0M and the best, 2,473 with 1 fall, at 9.0M; then 2,226 to 2,468 until it stopped. Asked for 2.0 m/s on 6 attempts of 10 s, every brain from 9M on cruised at about 1.46 to 1.49 m/s (8M: 1.40, 6M: 1.34); falls at 2 m/s: 3 of 6 for the 9M best brain, 0 of 6 for the 11M one. A saving bug lost one best brain (1,226 at 1,507,328 steps, where the brain from before the memory stop already had a file). About 690 steps per second
- `runs/duck_03_3/from_run_2` (2026-10-31 21:40 to 22:52, 0 to 3.3M (planned 8M; stopped by hand to start Duck 4)): From running school's best brain (9.0M), with no energy penalty. Exam (no energy points taken off, so not comparable with running school's): 2,882 at the start, best 3,016 with no falls at 1.5M, then 2,361 to 2,712 with 1 to 4 falls. Asked for 2.0 m/s (6 attempts of 10 s): 1.48 m/s at the start, 1.51 at 1M, 1.54 at 1.5M (0 falls), 1.61 at 2M and 1.60 at 3M (3 falls of 6). Asked for 1.5 m/s: 1.28 at the start, 1.40 at 2M, no falls. Free energy raised his top speed by about 0.1 m/s, then it levelled off, so Francisco moved on to force motors (Duck 4). About 690 to 870 steps per second

## Best brain

Film: [duck_03_3.mp4](duck_03_3.mp4), 16 s of the best brain at 0.4 m/s. It is not kept in git; make it with `python -m duck.catalogue film duck_catalogue/duck_03_3`.

- From `runs/duck_03_3/from_zero/best` at 6,504,448 steps (measured 2026-10-31 12:18)
- Exam (12 fixed attempts, no wobble): score 1188, falls 0 of 12
- Walking test: 10 of 10 attempts walked
- At 0.4 m/s: speed 0.38 m/s, 18.5 J per metre, feet in the air 43%, one foot down 55%, both down 2%

## Notes

A first try with the usual starting wobble (log_std -1.0, stopped at 1M steps) learned to fall on purpose: in this body the wobble whipped the stiff legs and burned about 180 joules per step, far more than staying up earns, and the strong legs could throw him down within 0.05 s. Measured on random wobble around standing, log_std -3.3 shakes the joints about as much as -1.0 shook the standard body, though it still burns about 5 times the energy. The closest duck to compare with is Duck 2.1, also trained from zero on the flat floor at the same speeds, but under an older forward reward without the overshoot rule. Running school (from_run_1, his idea once the walk was steady): his best walking brain (exam 1188 in flat_walking) continues on the flat floor at speeds 0.2 to 2.0 m/s (the flat_running_up_to_2ms world), faster than Duck 3.2's 1.5 since his legs are much stronger. The duck's world is now that one; the walking run keeps flat_walking in its own run_settings.json. Running exam scores cannot be compared with walking ones (faster speeds earn more forward points), nor directly with Duck 3.2's (his speeds only went to 1.5). Free energy (from_run_2, his idea): in running school he settled at about 1.5 m/s when asked for 2, so the energy penalty is removed (the flat_running_free_energy world, energy weight 0) and training continues from running school's best brain (9.0M). The duck's world is now that one.
