# Duck 3.1

**Duck 3's brain on taller stones, up to 3 cm**

Started 2026-10-30 10:14. Starts from: Duck 3's best brain (ducks/duck_03/brain, 3.5M steps, exam 966 under Duck 3's exam).

## What changed and why

Two changes from Duck 3. The floor: the most roughness goes from 1 cm to 3 cm, still picked at random from flat up to the most for every training attempt, so about a third of attempts stay under 1 cm. The exam: the rough part climbs instead of using the full height for all 12 attempts; its attempts are spread evenly from 0.25 cm to 3 cm (each attempt keeps its seed, so only the stone height changes). The flat part is unchanged.

Duck 3 walks on 1 cm stones (10 of 10) and mostly on 2 cm (8 of 10), but fell on every 3 cm attempt, after about 7 s, pitching forward when a foot caught a stone. Its brain never felt a stone above 1 cm. Francisco chose 3 cm as the next step. The exam climbs so that it matches what training asks for (any height up to the most), still tests the hardest floor, and gives useful scores early on, when most 3 cm attempts still end in a fall. A minor change, so Duck 3.1 rather than Duck 4.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Started from another brain: it keeps that brain's wobble, step size and PPO settings, so the starting values above apply only to runs from zero
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Floor: random cobblestones up to 0.03 m high, drawn at every reset with a random roughness in training: 8 waves of 0.2 to 0.5 m wavelength, stones 0.06 m wide with tops 0.35 to 1.0 of the tallest stone, gaps 0.07 of the stone size on each side, bump weight 0.5 (the stones get the rest). The exam's rough part climbs: its 12 attempts are spread evenly up to the full height
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: ae36187

## Training history

No runs yet.

## Best brain

No brain kept yet.

## Notes

Exam scores of Duck 3 and Duck 3.1 cannot be compared, because the rough part changed. Duck 3.1's first exam, at step 0, is Duck 3's brain on Duck 3.1's exam, which gives the fair starting point. A started run inherits the source brain's wobble, adapted step size and PPO settings, so the starting wobble and learning rate in the recipe above do not apply to this run.
