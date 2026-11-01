# Duck 4

**Force motors: the brain sets each joint's force (up to 10 times the standard limits), loose at 0; learning to walk from zero on a flat floor**

Started 2026-10-31 22:51. Starts from: Nothing: a new brain, since old brains send angles, not forces.

- World: flat_walking_strong_end_stops, Flat floor, walking speeds 0.2 to 0.7 m/s, and a doubled cost for resting joints on their end stops
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

- `runs/duck_04/from_zero` (2026-10-31 22:53 to 2026-11-01 03:09, 0 to 14.5M (4M, then continued overnight toward 30M with no early stop; stopped at 14.5M once it levelled off)): Started at 9 with 12 falls of 12 (with no commands he collapses in about 0.4 s). Every exam a new best but slow at first: 57 at 1.75M, all attempts falling. The breakthrough: 101 (10 falls) at 2.0M, 161 (7) at 2.25M, 501 (4) at 2.5M and the first exam with no falls at 2.75M (799; Duck 2.1 needed 2.0M, Duck 3.3 5.0M). 1,077 at 4.0M, 1,193 at 5.0M (past Duck 3.3's best walk, 1,188), then a few points at a time with almost no falls: 1,262 at 8.0M and the best, 1,287 with no falls, at 12.25M. No new best in the 9 exams after it (1,274 to 1,287), so it was stopped at 14.5M. His gait at 3.25M (0.4 m/s): body tilted 30 degrees chest up, hip swings held still at their end stop (+57 degrees), neck and ankles at theirs too, walking with the knees (42 degree swing) at 7.1 small steps a second: resting on the joint stops costs no motor energy. About 650 (on battery) to 1,050 steps per second
- `runs/duck_04/from_run_1` (2026-11-01 03:10 to 08:31, 0 to 19.2M (sized to end around 08:30, no early stop)): From the energy run's best brain (12.25M). Free energy exam (no energy points taken off, so higher than the energy run's): 1,484 at the start, then 1,470 to 1,499 with no falls in most exams and a few dips to about 1,310 with 1 or 2 falls; best 1,499 with no falls at 17.25M. It did not end the crouch: at 0.4 m/s (3 attempts of 20 s) the 17.25M brain still holds both hip swings at their end stop (+57 degrees), hip sides at theirs (+24), the neck at its limits and the ankles near theirs (-43), with the body tilted 27 degrees, walking with knees, ankles and toes at 9.2 steps a second (8.1 before) and twice the motor work (0.27 J per step, 0.135 before). So the end-stop posture is not only an energy saving: it is the stable pose he found, and nothing in the reward asks him to leave it. About 990 steps per second
- `runs/duck_04/from_run_2` (2026-11-01 09:30 to 10:13, 0 to 2.0M (planned 4M; stopped by hand, it had stalled)): From the energy run's best brain (12.25M) with end stop weight 0.05. Exam (with the end stop cost): 915 at the start, best 930 with no falls at 0.5M, then 850 to 928 with no new best. At 0.4 m/s (3 attempts of 10 s) only the toes left their stops (19% of the time at a stop, then 1%); the neck (95 to 99%), hip swings (97%), hip sides (98%) and ankles (about 77%) stayed on theirs. The small cost could not break the habit: moving one hip a little off its stop costs balance at once, while the saving is small. About 1,000 steps per second
- `runs/duck_04/from_run_3` (, ): 

## Best brain

No brain kept yet.

## Notes

Exam and walking scores can be compared with the walking ducks in flat_walking. With no commands the duck collapses in about 0.4 s, so he has to learn to hold himself up first. Free energy (from_run_1): overnight, once his walk stopped improving (no new best for 2M steps), the energy penalty was removed (the flat_walking_free_energy world), as Francisco asked before going to sleep, and training continued from the best brain (12.25M). The hope is that it also ends the end-stop crouch, since leaning on the joint stops only pays because motor work costs energy. The duck's world is now that one. End stops (from_run_2, his idea after the free energy run): a small cost for joints resting on their end stops (the flat_walking_end_stops world, weight 0.05, with the energy cost back on), starting from the energy run's best walking brain (12.25M). A gentle spring toward the standing pose was tried first in a quick test and dropped: holding a joint still against a spring costs no motor work, so weak springs (1 N m per radian) left the crouch as it was, and springs strong enough to push him out of it (2 and up) made the walking brain fall; Francisco also did not want springs to make him stiff like the position motor ducks. The duck's world is now that one. Stronger end stops (from_run_3, his idea): the end stop weight doubled to 0.1 (the flat_walking_strong_end_stops world), from the end stop run's best brain (0.5M, toes already off their stops). The duck's world is now that one.
