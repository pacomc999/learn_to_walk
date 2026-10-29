# Symmetric Brain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New training runs use a brain that is exactly left/right symmetric (the NET method of Abdolhosseini and others 2019), so the duck cannot learn a lopsided gait.

**Architecture:** A mirror table (`duck/mirror.py`) says how each of the 39 senses and 12 motors changes when the world is mirrored. One actor network `F` reads `[s, M(s)]` and outputs 5 side commands plus 2 neck commands; it runs on both views and the results are combined as `u1 + M_a(u2)`, which is symmetric by construction. The critic is `G([s, Ms]) + G([Ms, s])`. Sense scaling statistics are updated with each batch and its mirror image, so scaling keeps the symmetry. Wobble is averaged with its mirror partner before use.

**Tech Stack:** Python 3.12, MuJoCo, Gymnasium, Stable-Baselines3 PPO, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-duck-walker-design.md` (see "Symmetric brain (2026-10-29)" under "Decided by Francisco").

## Global Constraints

- No em or en dashes in any visible text (docs, comments, printed messages, test names are fine as identifiers).
- Beginner readable code: clear names, short single purpose functions, a comment above each section and function, matching the style of `duck/train.py`.
- Python 3.12 only, no TypeScript.
- No reward, body, energy or PPO setting changes. `NET_ARCH` stays `{"pi": [256, 256], "vf": [256, 256]}`, `START_LOG_STD` stays -1.
- No new tuned number anywhere. The only inputs are the mirror table (fixed by the body) and existing settings.
- Old plain snapshots (all runs up to `runs/night2`) must still load for `load_policy`, `load_brain`, `duck.evaluate`, `duck.video`, the dashboard, and `--resume`.
- Run tests with `.venv/Scripts/python -m pytest` from the project root.
- Commit messages: short, present tense, no prefix, ending with the trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_018W1SX6YrvxzJhw4vRRNR56`. Commit directly on main.

## The mirror table (fixed values, used verbatim)

Mirroring is a reflection through the vertical plane along the walking line (y becomes -y).

Observation (39), index: meaning, mirrored as
- 0 torso height: same
- 1 sideways position: sign flips
- 2 to 5 quaternion (w, x, y, z): becomes (w, -x, y, -z)
- 6 to 8 linear velocity (vx, vy, vz): becomes (vx, -vy, vz)
- 9 to 11 angular velocity (wx, wy, wz): becomes (-wx, wy, -wz)
- 12 to 23 joint angles and 24 to 35 joint velocities, each in motor order: neck_base and neck_top stay; left leg (hip_swing, hip_side, knee, ankle, toes) swaps with the right leg, and both hip_side values change sign
- 36 left foot contact, 37 right foot contact: swap
- 38 target speed: same

Action (12), motor order `neck_base, neck_top, left_hip_swing, left_hip_side, left_knee, left_ankle, left_toes, right_hip_swing, right_hip_side, right_knee, right_ankle, right_toes`:
- `ACTION_ORDER = [0, 1, 7, 8, 9, 10, 11, 2, 3, 4, 5, 6]`
- `ACTION_SIGNS = [1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1]`
- mirrored action = `action[ACTION_ORDER] * ACTION_SIGNS`
- `LEFT_MOTORS = [2, 3, 4, 5, 6]`, `NECK_MOTORS = [0, 1]`

## Review Focus

1. A wrong sign in the mirror table (quaternion, angular velocity, hip side) silently teaches nonsense: the physics test in Task 1 is the guard, and it must fail if any single sign is wrong.
2. Stable-Baselines3's own orthogonal initialisation or its optimizer built before module replacement would undo or skip the new layers: Task 3 tests that every trainable parameter is in the optimizer and that a fresh brain gives near zero actions.
3. Sense scaling with per sense statistics breaks symmetry after normalisation: Task 2 tests `normalize(mirror(x)) == mirror(normalize(x))` after updates.
4. Old plain snapshots must keep loading everywhere: Task 4 and Task 5 keep the existing tests for plain brains passing and add one that a plain snapshot still resumes.
5. Saving and loading a symmetric snapshot (the class must be importable, scaling statistics keep their mirrored class) and resuming training from it: Task 4 tests a save, load and resume round trip with equal weights.

---

### Task 1: The mirror table

**Files:**
- Create: `duck/mirror.py`
- Test: `tests/test_mirror.py`

**Interfaces:**
- Produces: `OBSERVATION_ORDER`, `OBSERVATION_SIGNS` (length 39), `ACTION_ORDER`, `ACTION_SIGNS` (length 12), `LEFT_MOTORS`, `NECK_MOTORS`; `mirror_observation(x)` and `mirror_action(x)`, each accepting a numpy array or a torch tensor whose last axis is the 39 or 12 values (any leading batch shape), returning the same type and dtype.

- [ ] **Step 1: Write the failing tests** in `tests/test_mirror.py`:
  - `test_mirroring_twice_gives_back_the_original`: random numpy batch (5, 39) and (5, 12), and a torch batch; mirroring twice equals the input exactly.
  - `test_mirror_swaps_feet_and_flips_drift`: a vector with drift 0.1 at index 1 and left contact 1 at index 36 mirrors to drift -0.1 and right contact 1 at index 37.
  - `test_mirrored_duck_stays_a_mirror_image` (the physics guard): reset `DuckWalkEnv(target_speed=0.4)` with seed 0 as duck A; make duck B the same way, then set B's `data.qpos` and `data.qvel` to the mirror of A's (qpos: x same, y sign flips, z same, quaternion as in the table, 12 joints with the action order and signs; qvel: linear (vx, -vy, vz), angular (-wx, wy, -wz), 12 joints with the action order and signs) and call `mujoco.mj_forward`. Then for 25 control steps, with actions drawn from `np.random.default_rng(1).uniform(-1, 1, 12)`, step A with `a` and B with `mirror_action(a)`, and assert `B observation ≈ mirror_observation(A observation)` with `atol=1e-3` after every step. The qpos and qvel mirroring helpers live in the test file.
  - `test_physics_guard_catches_a_wrong_sign`: the same 25 step comparison with the hip side sign deliberately not flipped fails (assert the largest difference exceeds 1e-3). This proves the guard has teeth.

- [ ] **Step 2: Run the tests and see them fail** (`ModuleNotFoundError: duck.mirror`).

- [ ] **Step 3: Write `duck/mirror.py`** with the table above, built from `MOTOR_NAMES` comments for clarity. For a torch input, index with a `torch.as_tensor` of the order and multiply by signs of the input's dtype and device.

- [ ] **Step 4: Run the tests and see them pass.** If the physics test fails only because of slow drift late in the 25 steps (not at step 1), report it rather than loosening the tolerance.

- [ ] **Step 5: Commit** `Add the duck mirror table`.

### Task 2: Mirrored sense scaling

**Files:**
- Modify: `duck/mirror.py`
- Modify: `duck/train.py` (`build_vec_env`)
- Test: `tests/test_mirror.py`

**Interfaces:**
- Consumes: `mirror_observation` (Task 1).
- Produces: `MirroredRunningMeanStd(RunningMeanStd)` in `duck/mirror.py` whose `update(batch)` calls the parent with `np.concatenate([batch, mirror_observation(batch)])`. `build_vec_env` sets `vec_normalize.obs_rms = MirroredRunningMeanStd(shape=(OBSERVATION_SIZE,))` right after creating the `VecNormalize`.

- [ ] **Step 1: Write the failing tests:**
  - `test_mirrored_scaling_keeps_symmetry`: a `MirroredRunningMeanStd` updated with three random batches (8, 39) whose index 1 has mean 0.5; then for a random vector `x`, `(mirror(x) - mean) / sqrt(var + 1e-8)` equals `mirror((x - mean) / sqrt(var + 1e-8))` to 1e-6, and the mean at index 1 is 0.
  - `test_new_duck_worlds_use_mirrored_scaling`: `build_vec_env(2, seed=0, use_subprocess=False).obs_rms` is a `MirroredRunningMeanStd`.
- [ ] **Step 2: Run, see them fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the whole suite, see it pass.**
- [ ] **Step 5: Commit** `Share sense scaling between mirrored senses`.

### Task 3: The symmetric brain

**Files:**
- Create: `duck/symmetric_policy.py`
- Test: `tests/test_symmetric_policy.py`

**Interfaces:**
- Consumes: `mirror_observation`, `mirror_action`, `LEFT_MOTORS`, `NECK_MOTORS` (Task 1).
- Produces:
  - `MirrorExtractor(nn.Module)` with `latent_dim_pi = 12`, `latent_dim_vf = 1`, `forward(features) -> (actions, value)`, `forward_actor(features) -> actions (batch, 12)`, `forward_critic(features) -> value (batch, 1)`, and `actor_cells(features) -> (actions, layer1, layer2)` where the layers are the 256 cell activations of the pass on the real view `[s, Ms]`. Its actor `F` is `Linear(78, 256), Tanh, Linear(256, 256), Tanh` as `self.actor_trunk` and `Linear(256, 7)` as `self.actor_head` (5 side then 2 neck outputs); its critic `G` is the same shape with `Linear(256, 1)` as head. Hidden sizes come from `net_arch`.
  - Combination: `u1` is a zero (batch, 12) tensor with `F([s, Ms])` side outputs in `LEFT_MOTORS` and neck outputs in `NECK_MOTORS`; `u2` the same from `F([Ms, s])`; actions `= u1 + mirror_action(u2)`. Value `= G([s, Ms]) + G([Ms, s])`.
  - `SymmetricPolicy(ActorCriticPolicy)`: passes `ortho_init=False` to the parent, builds `MirrorExtractor` in `_build_mlp_extractor`, and after the parent's `_build` replaces `action_net` and `value_net` with `nn.Identity()` and recreates `self.optimizer` over `self.parameters()` (same class, `lr_schedule(1)`, `optimizer_kwargs`). It overrides `_get_action_dist_from_latent` to use `log_std` averaged with `log_std[ACTION_ORDER]`.
  - Initialisation inside `MirrorExtractor` matches Stable-Baselines3 defaults: hidden layers orthogonal with gain `sqrt(2)`, actor head gain 0.01, critic head gain 1, all biases 0.

- [ ] **Step 1: Write the failing tests** (build the model as `PPO(SymmetricPolicy, DummyVecEnv([DuckWalkEnv]), policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "log_std_init": -1.0}, n_steps=64, batch_size=32, device="cpu", seed=0)`; to make the checks meaningful, first perturb every parameter with `torch.randn_like * 0.5` under `no_grad`):
  - `test_mirrored_situation_gives_mirrored_commands`: for a random (16, 39) batch `s`, `forward_actor(mirror_observation(s)) ≈ mirror_action(forward_actor(s))` (atol 1e-5).
  - `test_judge_rates_mirror_images_equally`: `forward_critic(mirror(s)) ≈ forward_critic(s)`.
  - `test_left_and_right_wobble_match`: after setting `log_std` to random values, the distribution's std for left_knee equals right_knee, and for left_hip_side equals right_hip_side.
  - `test_every_weight_is_trained`: the set of parameters in `policy.optimizer.param_groups` equals `set(policy.parameters())`, and `action_net`/`value_net` are `nn.Identity`.
  - `test_new_brain_starts_calm`: without the perturbation, mean actions for random inputs are all below 0.1 in size.
  - `test_ppo_learns_with_the_symmetric_brain`: `model.learn(128)` runs, the weights change, and symmetry still holds afterwards.
  - `test_brain_cells_match_the_commands`: `actor_cells(s)[0]` equals `forward_actor(s)`, and both layers have 256 values between -1 and 1.
- [ ] **Step 2: Run, see them fail.**
- [ ] **Step 3: Implement** `duck/symmetric_policy.py`.
- [ ] **Step 4: Run the whole suite, see it pass.**
- [ ] **Step 5: Commit** `Add a symmetric brain`.

### Task 4: Train new runs with the symmetric brain

**Files:**
- Modify: `duck/train.py` (`_new_model`)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `SymmetricPolicy` (Task 3), mirrored scaling (Task 2).
- Produces: every new run uses `PPO(SymmetricPolicy, ...)` with the same `policy_kwargs` and settings as now. Resumed runs keep whatever brain their snapshot has (plain or symmetric), because `PPO.load` and `VecNormalize.load` restore the saved classes.

- [ ] **Step 1: Write the failing tests** in `tests/test_train.py`:
  - `test_new_run_uses_the_symmetric_brain`: `train(tmp_path, total_steps=0, **FAST).policy` is a `SymmetricPolicy`.
  - `test_symmetric_run_resumes_with_the_same_brain`: train to 1024, then `load_latest_snapshot`; the loaded policy is a `SymmetricPolicy`, all weights equal (as in `test_resumed_brain_keeps_its_learning`), and `vec_normalize.obs_rms` is a `MirroredRunningMeanStd`; then resume training to 2048 with `train(..., resume=True)`.
  - `test_plain_snapshot_still_resumes`: build a plain run by calling `PPO("MlpPolicy", build_vec_env(...), ...)` and saving it with the existing snapshot helper at 0 steps (or copy the pattern from `_new_model` with `MlpPolicy`), then `train(..., total_steps=1024, resume=True, **FAST)` continues it and the policy is a plain `ActorCriticPolicy`, not a `SymmetricPolicy`.
  - Existing tests stay as they are and must pass (`test_new_brain_starts_with_log_std_minus_one` still sees 12 values of -1).
- [ ] **Step 2: Run, see the new ones fail.**
- [ ] **Step 3: Implement.** Update the module docstring and the `NET_ARCH` comment to say new runs use the symmetric brain.
- [ ] **Step 4: Run the whole suite, see it pass.**
- [ ] **Step 5: Commit** `Train new runs with the symmetric brain`.

### Task 5: Brain view, docs and timing

**Files:**
- Modify: `duck/policy.py` (`load_brain`)
- Modify: `tests/test_policy.py`, `README.md`, `CLAUDE.md`
- Modify: `duck/dashboard.html` only if a caption must say the brain view shows the duck's own view pass

**Interfaces:**
- Consumes: `MirrorExtractor.actor_cells` (Task 3).
- Produces: `load_brain` works for both brains: for a `SymmetricPolicy` it returns `BrainReading(actions, layer1, layer2)` from `actor_cells` (actions clamped to -1..1 like now); for a plain brain it behaves exactly as today.

- [ ] **Step 1: Write the failing test** `test_symmetric_brain_matches_policy` in `tests/test_policy.py`: a module fixture trains a symmetric run with `total_steps=0` (as the existing fixture, which is now symmetric too) and a plain snapshot fixture (built as in Task 4); `load_brain(...)(obs).action ≈ load_policy(...)(obs)` (abs 1e-5) for both, with 256 cells per layer.
- [ ] **Step 2: Run, see it fail.**
- [ ] **Step 3: Implement** the branch in `load_brain`, then update `README.md` (what the symmetric brain is and that old runs are plain) and `CLAUDE.md` (layout: `duck/mirror.py`, `duck/symmetric_policy.py`; the mirror table summary; that the dashboard brain view shows the real view pass of the symmetric brain). In the dashboard brain panel, add one short caption line under the cell grids for symmetric brains only if the page has no place that already explains the cells; otherwise leave the page alone.
- [ ] **Step 4: Run the whole suite, see it pass.** Then time 6 training rounds of a new symmetric run against the plain brain numbers (walking about 5.3 s, learning about 4.6 s per round) with the scratchpad script `time_round.py` pointed at a 0 step symmetric snapshot, and put the numbers in the task report.
- [ ] **Step 5: Commit** `Show the symmetric brain on the dashboard and document it`.

---

## After the plan (controller, not a task)

Start the pilot: `.venv/Scripts/python -m duck.train --run runs/pilot_symmetric --steps 3000000 --snapshot-every 250000`, move the dashboard to it, and compare with `runs/pilot_knees` (same reward, plain brain) on test style score, falls, foot pattern and energy per metre.
