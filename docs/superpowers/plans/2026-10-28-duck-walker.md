# Duck Walker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 12 motor MuJoCo duck (design B) that learns to walk forward in a straight line with PPO on a CPU laptop, saving snapshots along the way and turning them into one captioned "evolution" video.

**Architecture:** A hand written MuJoCo XML duck is wrapped in a custom Gymnasium environment whose reward and fall rules live in a small pure module. A training script runs PPO (Stable-Baselines3) over several parallel ducks, saves snapshot pairs (brain plus observation statistics) at fixed step intervals, logs a progress CSV, and can resume with a new curiosity setting. Separate scripts chart progress, judge whether a snapshot walks, replay a snapshot live, and record the snapshot video.

**Tech Stack:** Python 3.12, MuJoCo, Gymnasium, Stable-Baselines3 (PPO), PyTorch (CPU build), NumPy, Matplotlib, Pillow, imageio with ffmpeg, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-duck-walker-design.md`

## Global Constraints

- Python 3.12 in a project virtual environment at `.venv`. Fallback only if 3.12 fails: Python 3.10.4 with library versions pinned to ones that support it.
- CPU only. Install PyTorch from the CPU index: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Platform: Windows 11, 12 threads, 16 GB RAM. Anything using `SubprocVecEnv` must run under `if __name__ == "__main__":`.
- Duck: design B proportions, 12 position motors, in this exact order: `neck_base, neck_top, left_hip_swing, left_hip_side, left_knee, left_ankle, left_toes, right_hip_swing, right_hip_side, right_knee, right_ankle, right_toes`. Wings fixed to the body, no wing joints.
- Motors are position actuators with limited force; joints never teleport.
- Flat ground only.
- Reward per step: plus forward speed (no cap), plus small upright bonus, minus effort penalty (squared actions), minus sideways drift penalty.
- An attempt ends on falling (torso too low or tilted too far) or after 20 seconds of simulated time.
- Done test: at least 8 of 10 attempts of 20 seconds without falling, moving forward steadily, staying near the straight line.
- Snapshots: first (untrained), regular intervals, final. Video uses at most 25, evenly spread, in training order, 5 to 8 seconds each, captioned like "Snapshot 7 of 25, after 1.2 million steps".
- Stall rule: no improvement for 60 minutes means watch the latest snapshot video, then resume with more curiosity (`ent_coef`).
- Code is beginner readable: clear names, short single purpose functions, a comment on each section.
- No em dashes or en dashes in any visible text: captions, printed messages, chart labels, docs.
- Commit messages: short, present tense, no prefix (for example `Add duck model`), ending with the lines `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_018W1SX6YrvxzJhw4vRRNR56`.
- Run tests with `.venv/Scripts/python -m pytest` from the project root.

## Review Focus

- Laptop sleeps or the process is killed while a snapshot is being written: resume must use the newest complete snapshot, never a half written file. Test added in Task 4.
- Resuming with a different curiosity value: the loaded brain must keep its learning and use the new value. Test added in Task 6.
- A run with only one snapshot (stopped early): the video must still be made, with one clip. Test added in Task 8.
- Strong motors make the simulation blow up (NaN values): the environment must keep returning finite numbers. Test added in Task 3.
- Scripts started from a folder other than the project root: the duck XML must still be found. Test added in Task 1.

---

### Task 1: Environment setup and the duck model

**Files:**
- Create: `requirements.txt`, `duck/__init__.py`, `duck/assets/duck.xml`, `duck/model.py`, `tests/__init__.py`, `tests/test_model.py`

**Interfaces:**
- Produces (in `duck/model.py`):
  - `DUCK_XML_PATH: pathlib.Path`, resolved from `Path(__file__).parent / "assets" / "duck.xml"`
  - `MOTOR_NAMES: list[str]`, the 12 names in Global Constraints order
  - `load_duck_model() -> mujoco.MjModel`
  - `torso_height(data: mujoco.MjData) -> float`: z position of body `torso`
  - `torso_up_z(data: mujoco.MjData) -> float`: z component of the torso's up axis (1.0 upright, 0.0 on its side)
  - `foot_contacts(model, data) -> tuple[bool, bool]`: whether geoms `left_foot` and `right_foot` touch geom `floor`
  - `STANDING_HEIGHT: float`, torso height in the `stand` keyframe
  - `MIN_TORSO_HEIGHT: float`, equal to `0.6 * STANDING_HEIGHT`
- Model conventions used by later tasks: z is up, +x is forward, free joint on body `torso`, keyframe `stand`, tracking camera `follow` (`mode="trackcom"`), ground geom `floor`.

- [ ] **Step 1: Install Python 3.12 and create the virtual environment**

Run `py -3.12 --version`. If it is missing, ask Francisco before running `winget install -e --id Python.Python.3.12` (it installs software system wide). Then `py -3.12 -m venv .venv`.

- [ ] **Step 2: Write `requirements.txt` and install**

Lower bounds: `mujoco>=3.2`, `gymnasium>=1.0`, `stable-baselines3>=2.4`, `numpy`, `matplotlib>=3.9`, `pillow>=10`, `imageio[ffmpeg]>=2.35`, `pytest>=8`. Install torch from the CPU index first (Global Constraints), then `.venv/Scripts/python -m pip install -r requirements.txt`.
Expected: `.venv/Scripts/python -c "import mujoco, gymnasium, stable_baselines3, torch; print(torch.cuda.is_available())"` prints `False` and no errors.

- [ ] **Step 3: Write the failing tests in `tests/test_model.py`**

```python
import mujoco
from duck.model import (load_duck_model, MOTOR_NAMES, STANDING_HEIGHT,
                        MIN_TORSO_HEIGHT, torso_height, torso_up_z)

def test_model_loads_from_any_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert load_duck_model().nu == 12

def test_motor_names_and_order():
    model = load_duck_model()
    assert [model.actuator(i).name for i in range(model.nu)] == MOTOR_NAMES

def test_wings_have_no_joints():
    model = load_duck_model()
    names = [model.joint(i).name for i in range(model.njnt)]
    assert not any("wing" in name for name in names)

def test_follow_camera_exists():
    assert load_duck_model().camera("follow") is not None

def test_min_height_is_60_percent_of_standing():
    assert MIN_TORSO_HEIGHT == 0.6 * STANDING_HEIGHT

def test_duck_stands_when_holding_pose():
    model = load_duck_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    while data.time < 2.0:
        mujoco.mj_step(model, data)
    assert torso_height(data) > 0.8 * STANDING_HEIGHT
    assert torso_up_z(data) > 0.9
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'duck.model'`

- [ ] **Step 5: Write `duck/assets/duck.xml`**

Build design B from the mockup `duck-joints.html` (git ignored, under `.superpowers/brainstorm/`) converted to z up: long ellipsoid torso tilted slightly forward, tail, folded wing ellipsoids as geoms on the torso, two capsule neck segments, head sphere, box beak, legs set far back, thin box or mesh webbed feet named `left_foot`/`right_foot`. Timestep `0.005`. Hinge joints and `<position>` actuators named as in `MOTOR_NAMES`, each with a `ctrlrange` equal to its joint range and a `forcerange`. Hip side joints rotate about the x axis; all other joints about the y axis. Add `floor` plane, `follow` camera on the torso, and a `stand` keyframe whose `ctrl` holds the standing pose. Dimensions, masses, ranges, `kp` and force limits are the builder's choice, keeping the B look; if it cannot stand, a heavier tail or mass moved backward is allowed, moving legs forward is not.

- [ ] **Step 6: Implement `duck/model.py`** with the interface above. `STANDING_HEIGHT` is computed once at import from the `stand` keyframe.

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: 6 passed. Also open it by eye: `.venv/Scripts/python -m mujoco.viewer --mjcf=duck/assets/duck.xml` and check it looks like duck B.

- [ ] **Step 8: Commit** `requirements.txt duck tests` with message `Add duck model`.

---

### Task 2: Reward and fall rules

**Files:**
- Create: `duck/rewards.py`, `tests/test_rewards.py`

**Interfaces:**
- Produces (in `duck/rewards.py`):
  - `@dataclass RewardWeights(forward: float = 1.0, alive: float = 0.5, effort: float = 0.01, sideways: float = 0.5)`
  - `compute_reward(forward_velocity: float, sideways_offset: float, action: np.ndarray, weights: RewardWeights = RewardWeights()) -> tuple[float, dict[str, float]]`. The dict has keys `forward`, `alive`, `effort`, `sideways` (penalties stored as negative numbers) and they sum to the total.
  - `has_fallen(height: float, up_z: float, min_height: float, min_up_z: float = 0.64) -> bool`. `0.64` is about 50 degrees of tilt.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from duck.rewards import compute_reward, has_fallen, RewardWeights

def test_moving_forward_upright_with_no_effort():
    total, parts = compute_reward(1.0, 0.0, np.zeros(12))
    assert total == pytest.approx(1.5)

def test_effort_penalty_uses_squared_actions():
    total, parts = compute_reward(1.0, 0.0, np.ones(12))
    assert parts["effort"] == pytest.approx(-0.12)
    assert total == pytest.approx(1.38)

def test_sideways_penalty_same_on_both_sides():
    left, _ = compute_reward(0.0, -0.4, np.zeros(12))
    right, _ = compute_reward(0.0, 0.4, np.zeros(12))
    assert left == right == pytest.approx(0.3)

def test_moving_backward_loses_points():
    total, _ = compute_reward(-0.5, 0.0, np.zeros(12))
    assert total == pytest.approx(0.0)

def test_parts_add_up_to_total():
    total, parts = compute_reward(0.7, 0.2, np.full(12, 0.5), RewardWeights())
    assert sum(parts.values()) == pytest.approx(total)

def test_fall_rules():
    assert has_fallen(0.3, 1.0, min_height=0.4)
    assert has_fallen(0.6, 0.5, min_height=0.4)
    assert not has_fallen(0.6, 0.9, min_height=0.4)
```

- [ ] **Step 2: Run to verify failure**: `.venv/Scripts/python -m pytest tests/test_rewards.py -v`, expected `ModuleNotFoundError`.
- [ ] **Step 3: Implement `duck/rewards.py`.** Sideways penalty is `weights.sideways * abs(sideways_offset)`.
- [ ] **Step 4: Run to verify pass**: expected 6 passed.
- [ ] **Step 5: Commit** with message `Add reward and fall rules`.

---

### Task 3: Gymnasium environment

**Files:**
- Create: `duck/duck_env.py`, `tests/test_duck_env.py`

**Interfaces:**
- Consumes: everything in Task 1 `duck/model.py`; `compute_reward`, `has_fallen`, `RewardWeights` from Task 2.
- Produces (in `duck/duck_env.py`):
  - `FRAME_SKIP = 4` (control at 50 Hz), `MAX_EPISODE_SECONDS = 20.0`
  - `class DuckWalkEnv(gymnasium.Env)` with `__init__(self, render_mode: str | None = None, reward_weights: RewardWeights = RewardWeights(), width: int = 640, height: int = 480)`, `metadata = {"render_modes": ["rgb_array"], "render_fps": 50}`
  - attributes `model`, `data`, `max_steps` (= 1000)
  - `action_space`: `Box(-1, 1, (12,))`, scaled to each actuator's `ctrlrange`
  - `observation_space`: `Box` of size 37: torso height (1), torso orientation quaternion (4), torso linear velocity (3), torso angular velocity (3), 12 joint angles, 12 joint velocities, 2 foot contacts (0.0 or 1.0)
  - `reset(seed)` starts from keyframe `stand` with uniform noise of plus or minus 0.02 on joint angles from the seeded generator
  - `step(action)` returns `(obs, reward, terminated, truncated, info)`; `terminated` from `has_fallen(..., MIN_TORSO_HEIGHT)`, `truncated` when step count reaches `max_steps`; reward from `compute_reward(torso x velocity, torso y position, action)`; `info` has `x_position`, `y_position`, `reward_parts`. Stepping after termination keeps simulating (Task 8 relies on this to film the fall).
  - `render()` returns an RGB frame from camera `follow` using `mujoco.Renderer`
  - `has_fallen` is imported by name into this module so tests can replace it

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
from gymnasium.utils.env_checker import check_env
import duck.duck_env as duck_env_module
from duck.duck_env import DuckWalkEnv

def test_passes_gymnasium_checks():
    check_env(DuckWalkEnv(), skip_render_check=True)

def test_spaces():
    env = DuckWalkEnv()
    assert env.observation_space.shape == (37,)
    assert env.action_space.shape == (12,)
    assert env.action_space.low.min() == -1 and env.action_space.high.max() == 1

def test_same_seed_same_start():
    env = DuckWalkEnv()
    first, _ = env.reset(seed=0)
    second, _ = env.reset(seed=0)
    assert np.array_equal(first, second)

def test_attempt_stops_after_20_seconds(monkeypatch):
    monkeypatch.setattr(duck_env_module, "has_fallen", lambda *args, **kwargs: False)
    env = DuckWalkEnv()
    env.reset(seed=0)
    for step in range(1, 1001):
        _, _, terminated, truncated, _ = env.step(np.zeros(12))
        assert not terminated
        assert truncated == (step == 1000)

def test_lying_down_ends_the_attempt():
    env = DuckWalkEnv()
    env.reset(seed=0)
    env.data.qpos[2] = 0.05
    _, _, terminated, _, _ = env.step(np.zeros(12))
    assert terminated

def test_extreme_actions_stay_finite():
    env = DuckWalkEnv()
    obs, _ = env.reset(seed=0)
    for step in range(1000):
        action = np.ones(12) if step % 2 else -np.ones(12)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()

def test_render_gives_an_image():
    env = DuckWalkEnv(render_mode="rgb_array", width=160, height=120)
    env.reset(seed=0)
    assert env.render().shape == (120, 160, 3)
```

- [ ] **Step 2: Run to verify failure**: `.venv/Scripts/python -m pytest tests/test_duck_env.py -v`, expected `ModuleNotFoundError`.
- [ ] **Step 3: Implement `DuckWalkEnv`** as specified. Torso x velocity comes from `data.qvel[0]`.
- [ ] **Step 4: Run to verify pass**: expected 7 passed.
- [ ] **Step 5: Commit** with message `Add duck walking environment`.

---

### Task 4: Snapshot files

**Files:**
- Create: `duck/snapshots.py`, `tests/test_snapshots.py`

**Interfaces:**
- Produces (in `duck/snapshots.py`):
  - `@dataclass(frozen=True) Snapshot(steps: int, model_path: Path, norm_path: Path)`
  - `snapshot_paths(run_dir: Path, steps: int) -> Snapshot`: files `run_dir/"snapshots"/f"snapshot_{steps:012d}.zip"` and `..._{steps:012d}_norm.pkl`
  - `save_snapshot(model, vec_normalize, run_dir: Path, steps: int) -> Snapshot`: calls `model.save(temp_zip)` and `vec_normalize.save(temp_pkl)` with temp names `snapshot_{steps:012d}.tmp.zip` and `..._norm.tmp.pkl`, then `os.replace` each onto the final name, norm file last
  - `list_snapshots(run_dir: Path) -> list[Snapshot]`: only snapshots with both final files, sorted by steps
  - `latest_snapshot(run_dir: Path) -> Snapshot | None`
  - `choose_for_video(snapshots: list[Snapshot], max_count: int = 25) -> list[Snapshot]`: all if `len <= max_count`, else `max_count` evenly spaced by index, always including first and last

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
from duck.snapshots import (snapshot_paths, save_snapshot, list_snapshots,
                            latest_snapshot, choose_for_video)

class FakeSaver:
    def save(self, path):
        Path(path).write_text("x")

def test_saving_leaves_only_final_files(tmp_path):
    snap = save_snapshot(FakeSaver(), FakeSaver(), tmp_path, 2048)
    assert snap.model_path.exists() and snap.norm_path.exists()
    assert not list((tmp_path / "snapshots").glob("*.tmp*"))

def test_half_written_snapshot_is_ignored(tmp_path):
    save_snapshot(FakeSaver(), FakeSaver(), tmp_path, 1000)
    broken = snapshot_paths(tmp_path, 2000)
    broken.model_path.write_text("x")  # brain saved, statistics missing
    assert [s.steps for s in list_snapshots(tmp_path)] == [1000]
    assert latest_snapshot(tmp_path).steps == 1000

def test_snapshots_sorted_by_steps(tmp_path):
    for steps in (3000, 0, 1000):
        save_snapshot(FakeSaver(), FakeSaver(), tmp_path, steps)
    assert [s.steps for s in list_snapshots(tmp_path)] == [0, 1000, 3000]

def test_no_snapshots(tmp_path):
    assert list_snapshots(tmp_path) == []
    assert latest_snapshot(tmp_path) is None

def test_choose_for_video_spreads_evenly():
    snaps = [snapshot_paths(Path("run"), i * 1000) for i in range(100)]
    chosen = choose_for_video(snaps)
    assert len(chosen) == 25
    assert chosen[0] == snaps[0] and chosen[-1] == snaps[-1]
    assert choose_for_video(snaps[:3]) == snaps[:3]
    assert choose_for_video(snaps[:1]) == snaps[:1]
```

- [ ] **Step 2: Run to verify failure**: expected `ModuleNotFoundError`.
- [ ] **Step 3: Implement `duck/snapshots.py`.** `run_dir/"snapshots"` is created if missing.
- [ ] **Step 4: Run to verify pass**: expected 5 passed.
- [ ] **Step 5: Commit** with message `Add snapshot saving and listing`.

---

### Task 5: Progress log, stall check and chart

**Files:**
- Create: `duck/progress.py`, `tests/test_progress.py`

**Interfaces:**
- Produces (in `duck/progress.py`):
  - `@dataclass ProgressRow(time: datetime, timesteps: int, mean_reward: float)`
  - `append_progress(run_dir: Path, timesteps: int, mean_reward: float, when: datetime) -> None`: appends to `run_dir/"progress.csv"` with header `time,timesteps,mean_reward`, time in ISO format
  - `read_progress(run_dir: Path) -> list[ProgressRow]` (empty list if no file)
  - `minutes_since_improvement(rows: list[ProgressRow]) -> float`: minutes from the row with the highest `mean_reward` (earliest if tied) to the last row; `0.0` for no rows
  - `STALL_MINUTES = 60`
  - `plot_progress(run_dir: Path) -> Path`: saves `run_dir/"progress.png"`, score against hours of training, title "Duck training progress"
  - CLI `python -m duck.progress <run_dir>`: saves the chart and prints minutes since improvement; if at least `STALL_MINUTES`, prints exactly: `No improvement for an hour. Watch the latest snapshot video, then resume with more curiosity, for example --ent-coef 0.01`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timedelta
import pytest
from duck.progress import (ProgressRow, append_progress, read_progress,
                           minutes_since_improvement, plot_progress)

START = datetime(2026, 10, 28, 20, 0)

def rows(rewards, minutes_apart=10):
    return [ProgressRow(START + timedelta(minutes=i * minutes_apart), i * 1000, r)
            for i, r in enumerate(rewards)]

def test_still_improving():
    assert minutes_since_improvement(rows([1, 2, 3])) == 0.0

def test_stalled_for_90_minutes():
    assert minutes_since_improvement(rows([5, 4, 4, 3, 4, 4, 4, 4, 4, 4])) == pytest.approx(90)

def test_no_rows_means_no_stall():
    assert minutes_since_improvement([]) == 0.0

def test_log_round_trip(tmp_path):
    append_progress(tmp_path, 2048, 12.5, START)
    append_progress(tmp_path, 4096, 15.0, START + timedelta(minutes=5))
    assert read_progress(tmp_path) == [
        ProgressRow(START, 2048, 12.5),
        ProgressRow(START + timedelta(minutes=5), 4096, 15.0)]

def test_chart_is_saved(tmp_path):
    for row in rows([1, 2, 3]):
        append_progress(tmp_path, row.timesteps, row.mean_reward, row.time)
    assert plot_progress(tmp_path).exists()
```

- [ ] **Step 2: Run to verify failure**: expected `ModuleNotFoundError`.
- [ ] **Step 3: Implement `duck/progress.py`.** Use Matplotlib's `Agg` backend so it works without a window.
- [ ] **Step 4: Run to verify pass**: expected 5 passed.
- [ ] **Step 5: Commit** with message `Add training progress log and chart`.

---

### Task 6: Training with snapshots, resume and curiosity

**Files:**
- Create: `duck/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `DuckWalkEnv` (Task 3); `save_snapshot`, `latest_snapshot`, `list_snapshots` (Task 4); `append_progress` (Task 5).
- Produces (in `duck/train.py`):
  - `build_vec_env(n_envs: int, seed: int, use_subprocess: bool = True) -> VecNormalize`: `n_envs` copies of `Monitor(DuckWalkEnv())` in `SubprocVecEnv` (or `DummyVecEnv`), wrapped in `VecNormalize(norm_obs=True, norm_reward=True)`
  - `train(run_dir: Path, total_steps: int, snapshot_every: int = 1_000_000, n_envs: int = 8, ent_coef: float | None = None, resume: bool = False, use_subprocess: bool = True, ppo_overrides: dict | None = None) -> PPO`
    - new run: create PPO (`MlpPolicy`, `device="cpu"`, `ent_coef` if given), save snapshot at step 0 before learning; with `total_steps=0` it stops there (Task 8's test relies on this)
    - resume: load `latest_snapshot` brain and `VecNormalize.load(norm_path, venv)`; if `ent_coef` given, set `model.ent_coef`; continue with `reset_num_timesteps=False` until `total_steps` in total
    - a callback saves a snapshot each time `num_timesteps` passes a multiple of `snapshot_every`, plus the final step
    - a callback appends a progress row at each rollout end when `model.ep_info_buffer` has episodes (mean of `r`)
    - prints steps per second at each snapshot so laptop speed can be measured
    - keeps Windows awake while training with `ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)` and restores `0x80000000` at the end; no op on other systems
  - CLI `python -m duck.train --run runs/duck_v1 --steps 30000000 [--resume] [--ent-coef 0.01] [--envs 8] [--snapshot-every 1000000]`
- PPO settings beyond these are the builder's choice.

- [ ] **Step 1: Write the failing tests** (small and fast: `ppo_overrides={"n_steps": 256, "batch_size": 64}`, `use_subprocess=False`)

```python
from duck.train import train
from duck.snapshots import list_snapshots
from duck.progress import read_progress

FAST = dict(n_envs=2, use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})

def test_new_run_saves_untrained_regular_and_final(tmp_path):
    train(tmp_path, total_steps=2048, snapshot_every=1024, **FAST)
    steps = [s.steps for s in list_snapshots(tmp_path)]
    assert steps[0] == 0
    assert 1024 in steps
    assert steps[-1] >= 2048

def test_resume_continues_with_more_curiosity(tmp_path):
    train(tmp_path, total_steps=1024, snapshot_every=1024, **FAST)
    model = train(tmp_path, total_steps=2048, snapshot_every=1024, ent_coef=0.02, resume=True, **FAST)
    assert model.num_timesteps >= 2048
    assert model.ent_coef == 0.02
    assert list_snapshots(tmp_path)[-1].steps >= 2048

def test_zero_steps_saves_only_untrained_snapshot(tmp_path):
    train(tmp_path, total_steps=0, **FAST)
    assert [s.steps for s in list_snapshots(tmp_path)] == [0]

def test_progress_is_logged(tmp_path):
    train(tmp_path, total_steps=4096, snapshot_every=4096, **FAST)
    assert len(read_progress(tmp_path)) >= 1
```

- [ ] **Step 2: Run to verify failure**: expected `ModuleNotFoundError`.
- [ ] **Step 3: Implement `duck/train.py`.**
- [ ] **Step 4: Run to verify pass**: expected 4 passed. Then try the real launcher for 1 minute: `.venv/Scripts/python -m duck.train --run runs/try --steps 50000 --snapshot-every 25000` and check it runs with `SubprocVecEnv` and prints steps per second. Delete `runs/try` afterwards.
- [ ] **Step 5: Commit** with message `Add PPO training with snapshots`.

---

### Task 7: Loading a snapshot, judging it, and the live viewer

**Files:**
- Create: `duck/policy.py`, `duck/evaluate.py`, `duck/view_snapshot.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Snapshot`, `list_snapshots`, `latest_snapshot` (Task 4); `DuckWalkEnv` (Task 3); `train` (Task 6, test only).
- Produces:
  - `duck/policy.py`: `load_policy(snapshot: Snapshot) -> Callable[[np.ndarray], np.ndarray]`: loads PPO and the observation statistics (`VecNormalize.load` on a one duck `DummyVecEnv`, `training=False`, `norm_reward=False`), returns a deterministic function from a raw observation to an action
  - `duck/evaluate.py`:
    - `MIN_AVERAGE_SPEED = 0.2` (metres per second), `MAX_SIDEWAYS = 0.5` (metres), `REQUIRED_SUCCESSES = 8`
    - `judge_attempt(fell: bool, seconds: float, distance_x: float, final_y: float) -> bool`: success if not fell, `distance_x / seconds >= MIN_AVERAGE_SPEED`, `abs(final_y) <= MAX_SIDEWAYS`
    - `@dataclass EvaluationResult(successes: int, attempts: int, passed: bool)`
    - `evaluate_snapshot(snapshot: Snapshot, attempts: int = 10, seed: int = 0) -> EvaluationResult`: attempt `i` uses `reset(seed=seed + i)`, `passed` is `successes >= REQUIRED_SUCCESSES * attempts / 10` (8 of 10 for the real test)
    - CLI `python -m duck.evaluate <run_dir> [--snapshot STEPS]`, defaulting to the latest; prints `8 of 10 attempts walked. The duck can walk!` or `5 of 10 attempts walked. Not there yet.`
  - `duck/view_snapshot.py`: CLI `python -m duck.view_snapshot <run_dir> [--snapshot STEPS]`, runs the policy in `mujoco.viewer.launch_passive`, resetting after each fall or 20 seconds, in real time

- [ ] **Step 1: Write the failing tests**

```python
from duck.evaluate import judge_attempt, evaluate_snapshot
from duck.snapshots import latest_snapshot
from duck.train import train

def test_judge_attempt():
    assert judge_attempt(False, 20.0, 5.0, 0.1)
    assert not judge_attempt(True, 20.0, 5.0, 0.1)    # fell
    assert not judge_attempt(False, 20.0, 3.0, 0.1)   # 0.15 m/s is too slow
    assert not judge_attempt(False, 20.0, 5.0, -0.6)  # drifted off the line

def test_evaluate_a_real_snapshot(tmp_path):
    train(tmp_path, total_steps=512, snapshot_every=512, n_envs=2,
          use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})
    result = evaluate_snapshot(latest_snapshot(tmp_path), attempts=2)
    assert result.attempts == 2 and 0 <= result.successes <= 2
```

- [ ] **Step 2: Run to verify failure**: expected `ModuleNotFoundError`.
- [ ] **Step 3: Implement `duck/policy.py`, `duck/evaluate.py`, `duck/view_snapshot.py`.**
- [ ] **Step 4: Run to verify pass**: expected 2 passed. Check the viewer by hand on the snapshot from a short training run: a window opens and the untrained duck flops.
- [ ] **Step 5: Commit** with message `Add snapshot evaluation and live viewer`.

---

### Task 8: The evolution video

**Files:**
- Create: `duck/video.py`, `tests/test_video.py`

**Interfaces:**
- Consumes: `list_snapshots`, `choose_for_video` (Task 4); `load_policy` (Task 7); `DuckWalkEnv` with `render_mode="rgb_array"` (Task 3).
- Produces (in `duck/video.py`):
  - `caption_text(position: int, count: int, steps: int) -> str`, `position` counted from 1
  - `add_caption(frame: np.ndarray, text: str) -> np.ndarray`: white text on a dark band at the bottom, using Pillow
  - `record_clip(policy, seconds: float = 6.0, fps: int = 30, width: int = 640, height: int = 480, seed: int = 0) -> list[np.ndarray]`: resets with `seed`, steps the env and keeps a frame whenever simulated time passes the next 1/fps mark; after a fall it keeps simulating with zero actions so the fall and the duck lying down stay on screen
  - `make_video(run_dir: Path, out_path: Path | None = None, max_snapshots: int = 25, seconds_per_clip: float = 6.0, fps: int = 30) -> Path`: default output `run_dir/"evolution.mp4"`; raises `ValueError(f"No snapshots found in {run_dir}")` when empty
  - CLI `python -m duck.video <run_dir> [--seconds 6]`

- [ ] **Step 1: Write the failing tests**

```python
import imageio.v3 as iio
import numpy as np
import pytest
from duck.video import caption_text, add_caption, make_video
from duck.train import train

def test_captions():
    assert caption_text(1, 25, 0) == "Snapshot 1 of 25, untrained"
    assert caption_text(7, 25, 1_200_000) == "Snapshot 7 of 25, after 1.2 million steps"
    assert caption_text(3, 25, 250_000) == "Snapshot 3 of 25, after 250 thousand steps"
    assert caption_text(25, 25, 30_000_000) == "Snapshot 25 of 25, after 30 million steps"

def test_caption_keeps_image_size():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert add_caption(frame, "Snapshot 1 of 1, untrained").shape == (120, 160, 3)

def test_no_snapshots_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match="No snapshots found"):
        make_video(tmp_path)

def test_video_with_a_single_snapshot(tmp_path):
    train(tmp_path, total_steps=0, snapshot_every=1024, n_envs=2,
          use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})
    path = make_video(tmp_path, seconds_per_clip=0.5, fps=30)
    assert path.exists()
    assert len(iio.imread(path)) == 15
```

- [ ] **Step 2: Run to verify failure**: expected `ModuleNotFoundError`.
- [ ] **Step 3: Implement `duck/video.py`.** Write the MP4 with `imageio` (ffmpeg plugin), H.264.
- [ ] **Step 4: Run to verify pass**: expected 4 passed. Play the test video by hand and check the caption is readable.
- [ ] **Step 5: Commit** with message `Add evolution video`.

---

### Task 9: First real training run and project docs

**Files:**
- Modify: `README.md`
- Create: `CLAUDE.md`

- [ ] **Step 1: Run the whole test suite**: `.venv/Scripts/python -m pytest -v`, expected all pass.
- [ ] **Step 2: Start the real run in the background** with the laptop plugged in: `.venv/Scripts/python -m duck.train --run runs/duck_v1 --steps 30000000`.
- [ ] **Step 3: Measure speed** after about 15 minutes from the printed steps per second, and estimate hours to 30 million steps.
- [ ] **Step 4: Update `README.md`** with: what the project is, setup commands, the commands for training, resuming with more curiosity, progress chart, evaluation, live viewer and video, the stall rule, and the measured speed. No em or en dashes.
- [ ] **Step 5: Create `CLAUDE.md`** for the project: folder layout, how to run tests, motor order, the reward and fall rules, snapshot file format, and the beginner readable code style.
- [ ] **Step 6: Commit** with message `Add usage docs and first run timing`.
- [ ] **Step 7: While training runs**, check `python -m duck.progress runs/duck_v1` at least once a day. Apply the stall rule from the spec. When it finishes or walks, run `python -m duck.evaluate runs/duck_v1` and `python -m duck.video runs/duck_v1`.
