# Duck Walker Design

Date: 2026-10-28
Project folder: `learn_to_walk/`
Owner: Francisco Martinez Checa (pacomc999), a beginner developer who knows HTML, CSS and JavaScript well and Python occasionally.

## Intent

A 3D duck in a physics simulation teaches itself to walk. The finished duck walks well; all the stumbling and falling happens during training. The heart of the project is watching that progress: a video of the duck at successive stages of training, one clip after another, from flopping over to walking.

Francisco wants to do this "the proper way": research-style reinforcement learning in Python, rather than a browser toy. Sharing the results with other people matters eventually, but is explicitly deferred.

## Goals (version one)

1. A 3D duck body with realistic duck proportions (see Body).
2. A neural network brain trained by reinforcement learning to walk forward in a straight line on flat ground without falling over.
3. The duck goes as fast as it safely can. Style does not matter: hopping, shuffling or any odd gait counts as success as long as it makes steady forward progress without falling.
4. Snapshots of the brain saved at regular points during training.
5. A video made from those snapshots: each snapshot walks for a few seconds, one after another, in training order, so the improvement is visible.
6. Trains on Francisco's laptop (CPU only). We measure how long it actually takes rather than assuming.

## Later goals (not version one)

- Going around obstacles. This needs the duck to steer and turn, so it is a separate training phase built on top of version one.
- Sharing results more easily (for example on a web page).
- Moving wings: one flap motor per wing, unlocked after walking works and trained on from the version one brain. The simulation has no air, so wings cannot give lift; the risk is the duck leaning on them like crutches, so wings must not be able to touch the ground (too short to reach it, or an attempt ends on wing contact).

## Non-goals

- Simulated muscles and tendons. Joints are driven by motors.
- Hills, steps, slopes or uneven terrain.
- A required or capped speed.
- Cloud or GPU training (Google Colab, MJX). Possible later if laptop training proves too slow.
- TensorFlow. See Tools.

## Anti-goals (would be a failure even if it technically works)

- Joints that teleport to an angle. Motors must apply limited force through the physics, so falling and struggling are real.
- A duck that "succeeds" by standing still or balancing in place. Forward progress is required.
- Only a final result with no visible progression. The snapshot video is the point.

## Constraints

- Machine: Windows 11 laptop, AMD Ryzen 5 5500U (6 cores, 12 threads), 16 GB RAM, integrated AMD Radeon graphics (no NVIDIA card, so no CUDA).
- Python 3.10.4 is installed. Use a virtual environment inside the project.
- Code must be beginner-readable: clear names, short functions that each do one thing, comments explaining each section.
- No em dashes or en dashes in visible text (on-screen labels, video captions, docs).
- Commit messages in present tense, short ("Add duck model").

## Decided by Francisco

- **Platform:** Python.
- **Body:** design B from the mockups, "real duck proportions": a long body tilted slightly forward, legs set far back (which is what produces a duck waddle), a bendy neck with a head and beak, folded wings, a small tail, and webbed feet.
- **Actuation:** motors at the joints, not muscles.
- **Learning:** a neural network trained by reinforcement learning.
- **Ground:** flat.
- **Effort penalty:** yes, include a penalty for wasted effort.
- **Speed:** as fast as it safely can; forward reward is not capped.
- **Gait style:** does not matter.
- **Viewing:** a video of each snapshot walking, one after another.
- **Hardware:** train on his laptop first and see how fast it is.
- **Motors:** the full duck, all 12 motors active from the start (seen in the motor mockup, `duck-joints.html`): neck base, neck top, and on each leg hip swing, hip side, knee, ankle and toes. Simpler 10, 8 and 6 motor versions were shown and not chosen.
- **Wings:** locked to the body in version one, with no motors.
- **If training is slow:** waiting longer is the first choice. Training must therefore show a progress chart (average score over time) so improvement can be seen.
- **If training stalls:** if the score shows no improvement for one hour, first watch a video of the latest snapshot to see what the duck is doing, then increase the curiosity setting (the amount of random experimenting, PPO's entropy coefficient) and continue training from the latest snapshot.

## Calls made by Claude (Francisco can reject any of these)

Francisco saw a summary covering MuJoCo, PyTorch with Stable-Baselines3 instead of TensorFlow, the snapshots-as-generations idea and CPU training, and replied "looks right". The rest of this list is presented to him with this document for review.

- **Physics engine: MuJoCo.** The standard for walking research, free, runs on CPU, has a built-in 3D viewer and offscreen rendering for videos.
- **Learning library: Stable-Baselines3 with PPO, on PyTorch.** Most current reinforcement learning tooling is built on PyTorch; TensorFlow has largely faded from this area. Francisco originally assumed TensorFlow, and was told about this switch.
- **Environment: a custom Gymnasium environment** wrapping the MuJoCo duck, modelled on Gymnasium's built-in Ant and Walker2d environments.
- **Motor control: position targets.** The brain outputs a target angle for each joint, and a strength-limited motor tries to reach it. Brains usually learn faster this way than when controlling raw force, and physics stays honest.
- **Side-to-side hips:** the first body mockup showed 10 motors whose leg joints only swing forward and back. A 3D duck also needs to stop tipping sideways, so a side-to-side hip joint was added to each leg, giving the 12 motors Francisco then chose.
- **What the duck senses:** body height, body tilt (orientation), body speed and spin, every joint angle and joint speed, and whether each foot touches the ground.
- **Reward, each time step:**
  - plus forward speed (distance moved forward per step, no cap);
  - plus a small constant for staying upright ("still alive" reward);
  - minus a small effort penalty (squared size of the motor commands);
  - minus a penalty for drifting sideways from the straight line.
- **An attempt ends** when the body drops below a height threshold or tips past a tilt limit, or after about 20 seconds of simulated time.
- **Parallel training:** run several simulated ducks at once to use the laptop's 12 threads.
- **Snapshots:** save the brain at regular intervals, aiming for roughly 20 to 30 snapshots over a full run, plus the first (untrained) and final ones.
- **Video:** for each snapshot, record about 5 to 8 seconds of the duck walking from the same starting position, with a caption such as "Snapshot 7 of 25, after 1.2 million steps". Join the clips into one MP4 in training order. Also provide a way to open any single snapshot in MuJoCo's live viewer.
- **If more curiosity does not help:** try, in this order: fix the reward that rewards the bad habit (for example shrink the upright reward if the duck stands still, end the attempt if the body or knees touch the ground, make falling cost more); add training wheels (a supporting force that fades over training, or easier goals first such as standing); go back to a snapshot from before the habit formed and continue with the fixed reward. Changing the duck B shape or locking motors comes last, and only with Francisco's agreement.
- **No total time limit:** training runs as long as it keeps improving. Check progress at least once a day.
- **When training is done:** the duck counts as walking when it finishes at least 8 of 10 full 20 second attempts without falling, moving forward steadily and staying near the straight line.
- **Long runs:** training must be able to resume from the latest saved snapshot after a stop, sleep or reboot, since runs may last overnight or longer. Keep the laptop from sleeping during runs.
- **Python version:** Python 3.10 reaches end of life in October 2026 and newer library releases may drop it. Install Python 3.12 and use it for the project's virtual environment. If that causes trouble, fall back to 3.10.4 with library versions pinned to ones that still support it.
- **Duck model file:** written by hand as a MuJoCo XML file from simple shapes (capsules, spheres, ellipsoids, thin boxes for feet), matching mockup B.

## Left to the builder

- Exact body dimensions, masses, joint ranges and motor strengths, as long as the duck keeps mockup B's look and proportions and can physically stand.
- Reward weights, termination thresholds and PPO settings.
- The exact snapshot interval and total training length.
- Project file and folder layout, and which video library to use.
- Whether to start with a simpler test (for example making sure the duck can stand) before full walking training.

## Reference

- Body mockups from brainstorming: `.superpowers/brainstorm/1526-1793197752/content/duck-bodies-v2.html` (git-ignored). Design B is the chosen one.
