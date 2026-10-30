# Duck 3

**Duck 2.1's brain on a rough cobblestone floor, with an overshoot penalty**

Started 2026-10-30 03:12. Starts from: Duck 2.1's best brain (duck_catalogue/duck_02_1/brain, 4.0M steps, exam 986).

## What changed and why

Three changes from Duck 2.1. The floor: square cobblestones (6 cm) laid on rolling bumps, with a new layout every attempt and a random roughness from flat to 1 cm. The reward: going faster than the asked speed now loses forward points the same way as going slower. The exam: a flat part and a rough part of 12 attempts each, averaged; the average picks the best brain.

Duck 2.1 walks on flat ground about as well as the exam can tell (986, ceiling about 980 to 1,000), so the next goal is walking on uneven ground. The overshoot penalty comes from the literature review: Duck 2.1 still walked about 7.5 m per exam attempt where the asked speeds give 6.3 m. Francisco chose to make both changes in one duck and to start from Duck 2.1's best brain, accepting that the two effects cannot be told apart.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Started from another brain: it keeps that brain's wobble, step size and PPO settings, so the starting values above apply only to runs from zero
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Floor: random cobblestones up to 0.01 m high, drawn at every reset with a random roughness in training: 8 waves of 0.2 to 0.5 m wavelength, stones 0.06 m wide with tops 0.35 to 1.0 of the tallest stone, gaps 0.07 of the stone size on each side, bump weight 0.5 (the stones get the rest). The exam's rough part uses the full height for all 12 attempts
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: 953f4c0

## Training history

- `runs/duck_03/from_duck_02_1` (2026-10-30 03:13 to 04:43, 0 to 4.0M): Started at 732 (flat 798, rough 667, 3 falls of 24) with Duck 2.1's brain. Climbed to 877 by 0.5M, then jumped to 948 with 0 falls at 1.5M when the flat score rose from about 825 to 957 (it learned to slow to the asked speed). Held between 891 and 936 with 1 or 2 falls, then reached its best, 966 with 1 fall (flat 939, rough 992), at 3.5M; 965 at 3.75M and 929 at 4.0M. The rough part rose steadily all run (667 to 994). About 765 steps per second, slower than Duck 2.1 because of the height field floor

## Best brain

Film: [duck_03.mp4](duck_03.mp4), 16 s of the best brain at 0.4 m/s. It is not kept in git; make it with `python -m duck.catalogue film duck_catalogue/duck_03`.

- From `runs/duck_03/from_duck_02_1/best` at 3,506,176 steps (measured 2026-10-30 04:44)
- Exam (24 fixed attempts, 12 flat and 12 rough, no wobble): score 966, falls 1 of 24 (flat 939, rough 992)
- Walking test: 10 of 10 attempts walked
- Walking test on the rough floor (1 cm stones): 10 of 10 attempts walked
- At 0.4 m/s: speed 0.4 m/s, 9.8 J per metre, feet in the air 0%, one foot down 96%, both down 4%
- At 0.4 m/s on the rough floor (1 cm stones): speed 0.4 m/s, 10.4 J per metre, feet in the air 0%, one foot down 97%, both down 2%

## Notes

Duck 2.1's best brain scored 986 under the old forward rule. Under Duck 3's exam and reward the same brain scores 732 (flat 798, rough 667, 3 falls of 24): the flat drop is entirely the overshoot penalty, because Duck 2.1 walks faster than asked. Scores before and after the reward change cannot be compared. A started run inherits the source brain's wobble (std about 0.099), its adapted step size (about 0.0000117) and its PPO settings, so the starting wobble and learning rate in the recipe above do not apply to this run. Challenges (rocks, speed, backpack, taller stones): see challenges/README.md.
