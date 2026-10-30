# Duck 3.1

**Duck 3's brain on taller stones, up to 3 cm, then up to 4 cm**

Started 2026-10-30 10:14. Starts from: Duck 3's best brain (duck_catalogue/duck_03/brain, 3.5M steps, exam 966 under Duck 3's exam).

- World: stones_up_to_4cm, Cobblestones from flat up to 4 cm, walking speeds 0.2 to 0.7 m/s
- Body: standard, The duck as designed in duck/assets/duck.xml

## What changed and why

Two changes from Duck 3. The floor: the most roughness goes from 1 cm to 3 cm, still picked at random from flat up to the most for every training attempt, so about a third of attempts stay under 1 cm. The exam: the rough part climbs instead of using the full height for all 12 attempts; its attempts are spread evenly from 0.25 cm to 3 cm (each attempt keeps its seed, so only the stone height changes). The flat part is unchanged. Second run (from_duck_03_1): the most roughness goes from 3 cm to 4 cm, and the exam matches training: 24 attempts with floor heights spread evenly from flat to 4 cm, scored in a lower half (flat to about 1.9 cm) and a taller half (about 2.1 to 4 cm); each half asks for all 12 speeds, mixed. Still Duck 3.1, Francisco's choice, since it is the same idea as the first run.

Duck 3 walks on 1 cm stones (10 of 10) and mostly on 2 cm (8 of 10), but fell on every 3 cm attempt, after about 7 s, pitching forward when a foot caught a stone. Its brain never felt a stone above 1 cm. Francisco chose 3 cm as the next step. The exam climbs so that it matches what training asks for (any height up to the most), still tests the hardest floor, and gives useful scores early on, when most 3 cm attempts still end in a fall. A minor change, so Duck 3.1 rather than Duck 4. The second run: Duck 3.1 walked about two thirds of 3 cm attempts on new floors (53 of 80 in all tests on 2026-10-30), so Francisco asked to see how good it can get on 3 to 4 cm stones without losing normal walking.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Started from another brain: it keeps that brain's wobble, step size and PPO settings, so the starting values above apply only to runs from zero
- Reward per step: forward 3.0 x (target - |speed - target|) (speed over the target loses points too), alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- Floor: random cobblestones up to 0.04 m high, drawn at every reset with a random roughness in training: 8 waves of 0.2 to 0.5 m wavelength, stones 0.06 m wide with tops 0.35 to 1.0 of the tallest stone, gaps 0.07 of the stone size on each side, bump weight 0.5 (the stones get the rest). The exam matches training: 24 attempts with floor heights spread evenly from flat up to the full height, scored in a lower and a taller half
- Start: standing pose, each joint nudged randomly by up to 0.02 rad, then a stride stance: a coin flip picks the leading leg, one hip moves 0.04 rad forward and the other 0.04 rad back
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: 8068a17

## Training history

- `runs/duck_03_1/from_duck_03` (2026-10-30 10:16 to 11:58, 0 to 4.0M): Started at 869 (flat 939, rough 799, 5 falls of 24) with Duck 3's brain. Jumped to 966 with 1 fall at 0.25M, reached 979 with 0 falls at 1.0M and its best, 981 with 1 fall (flat 1002, rough 961), at 2.25M. After 1M the exams swung between about 870 and 981: good exams had 0 or 1 falls, weak ones (0.75M, 1.25M, 3.25M, 3.75M, 4.0M) had 3 or 4, split between flat and rough. The step size sat at its floor (0.00001) for most of the run while each update still changed the brain about twice the target, which fits the swings. About 660 steps per second; 4M steps took 1 h 42 min
- `runs/duck_03_1/from_duck_03_1` (2026-10-30 22:18 to 23:22, 0 to 2.5M (stopped early after 8 exams without a new best)): Started at 885 (lower 1162, taller 607, 8 falls of 24) with Duck 3.1's first best brain on the new exam. Jumped to 1052 at 0.25M and 1204 (4 falls) at 0.5M, then swung between 951 and 1168 without a new best until the final exam at 2,506,760 steps scored 1206 with 1 fall (lower 1302, taller 1110), only 8 steps after an exam of 1139 with 4 falls: the brain was the same, only the sense scaling had moved. The training score did not rise (about 950 throughout). The step size sat at its floor all run. About 700 steps per second. On 30 new floors (seeds 300 to 329) the kept brain walked 30 of 30 flat, 28 of 30 at 2 cm, 23 of 30 at 3 cm (first best: 19) and 10 of 30 at 4 cm (first best: 4). In the exam all its falls were on stones of 3.5 cm and taller, at slow and fast speeds alike.

## Best brain

Film: [duck_03_1.mp4](duck_03_1.mp4), 16 s of the best brain at 0.4 m/s. It is not kept in git; make it with `python -m duck.catalogue film duck_catalogue/duck_03_1`.

- From `runs/duck_03_1/from_duck_03_1/best` at 2,506,760 steps (measured 2026-10-30 23:48)
- Exam (24 fixed attempts, floor heights spread from flat to 4 cm, no wobble): score 1206, falls 1 of 24 (lower stones 1302, taller stones 1110)
- Walking test: 10 of 10 attempts walked
- Walking test on the rough floor (4 cm stones): 3 of 10 attempts walked
- At 0.4 m/s: speed 0.41 m/s, 11.2 J per metre, feet in the air 0%, one foot down 94%, both down 6%
- At 0.4 m/s on the rough floor (4 cm stones): speed 0.34 m/s, 16.0 J per metre, feet in the air 3%, one foot down 82%, both down 16%

## Notes

Exam scores of Duck 3 and Duck 3.1 cannot be compared, because the rough part changed. Duck 3.1's first exam, at step 0, is Duck 3's brain on Duck 3.1's exam, which gives the fair starting point. A started run inherits the source brain's wobble, adapted step size and PPO settings, so the starting wobble and learning rate in the recipe above do not apply to this run. Attempt by attempt, the best brain loses about 105 points per attempt to energy (about 11 J per metre on flat ground), 38 to 70 to forward points (starting from standstill, small speed errors, the one fall at 2.5 cm) and 3 to 5 to drift; the exam ceiling is about 1,150 with no energy, about 1,045 at its energy use. Challenges (speed, backpack, perilous road of tall stones, rocks): see challenges/README.md. The kept brain is from the second run, kept with --force because the first best's 981 came from the older exam; the second run's first exam (885) is the first best on the new exam. The recipe below is the second run's; the first run used a most roughness of 3 cm and the older exam (12 flat attempts plus 12 climbing to 3 cm). Worlds: Duck 3.1 names stones_up_to_4cm, the world of its second run; its first run (from_duck_03) trained in stones_up_to_3cm.
