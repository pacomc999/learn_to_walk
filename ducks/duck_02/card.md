# Duck 2

**Duck 1 plus an adaptive step size, trained from zero**

Started 2026-10-29 16:53. Starts from: zero (same seed and untrained brain as Duck 1).

## What changed and why

Same recipe as Duck 1, except the step size: after every brain update the learning rate is divided by 1.5 if the brain changed more than twice the target (0.01), or multiplied by 1.5 if it changed less than half of it (Rudin and others 2021).

Duck 1 collapsed after its best at 1.61M. A measurement run showed every update changing the brain 50 to 100 times more than PPO is designed for (approx_kl 0.5 to 1.1, safety limit hit by over 80% of moves), growing as the wobble shrank. Trained from zero with the same seed as Duck 1, so the two recipes can be compared fairly.

## Recipe

- Brain: SymmetricPolicy, actor layers [256, 256], judge layers [256, 256], starting wobble log_std -1.0
- Reward per step: forward 3.0 per m/s up to the target speed, alive 0.2, energy 1.5 per joule, sideways 0.5 per metre
- Target speed: random from 0.2 to 0.7 m/s per attempt; attempts last 20 s
- PPO: n_steps 1024, batch_size 256, n_epochs 10, learning_rate 0.0003, gamma 0.99, gae_lambda 0.95, clip_range 0.2
- Step size: adaptive, aiming for a change of 0.01 per update (divided or multiplied by 1.5, kept between 1e-05 and 0.01); the PPO learning rate is only the starting value
- Code version: 22a13ba

## Training history

No runs yet.

## Best brain

No brain kept yet.

## Notes

(none)
