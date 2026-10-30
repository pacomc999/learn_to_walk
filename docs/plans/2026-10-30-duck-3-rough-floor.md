# Duck 3: Rough Floor and Overshoot Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the duck train and be examined on a random cobblestone floor, score overshooting the asked speed like undershooting it, and start a run from another run's brain.

**Architecture:** A new `duck/floor.py` makes height maps and a MuJoCo model whose floor is a height field; `DuckWalkEnv` draws a new layout and roughness at every reset (roughness exactly 0 keeps the old flat plane model). The exam becomes two parts (flat and rough, 12 attempts each) averaged into one score. `train` gains `--start-from`, and the viewing tools, dashboard and catalogue learn about the floor.

**Tech Stack:** Python 3.12, MuJoCo 3.17 (height fields), Gymnasium, Stable-Baselines3 PPO, pytest.

**Spec:** `docs/designs/2026-10-30-duck-3-rough-floor-design.md` (read it first, including "Calls made by Claude after the builder check").

## Global Constraints

- No em or en dashes in any visible text (docs, comments that are printed, captions, messages, chart labels).
- Beginner readable code: clear names, short single purpose functions, a comment above each function and section, matching the existing files.
- Python 3.12 only, no TypeScript. Run tests with `.venv/Scripts/python -m pytest` from the project root.
- Commit messages short, present tense, no prefix, ending with these two lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_018W1SX6YrvxzJhw4vRRNR56`
- Work directly on `main`. Do not push (the controller pushes).
- The 39 number observation, the brain, PPO settings, step size rule, energy weight and stride start do not change.
- The floor geom must stay named `floor` (foot contact sensing in `duck/model.py` matches that name).
- Floor shape numbers, copied from the sketch: 8 cosine waves, wavelengths 0.2 to 0.5 m with random direction and phase; square stones 0.06 m, flat tops at 0.35 to 1.0 of the stone height, gaps 7% of the stone size on each side at height 0; bumps and stones each scaled to 0..1 and added with weight 0.5 each; then scaled so the highest point equals the roughness. `MAX_ROUGHNESS = 0.01` m.
- Overshoot rule: forward part = `forward_weight * (target - abs(speed - target))` when a target speed is set.

## Review Focus

1. A resumed old run whose `exam.csv` has the old 5 column header: new exam rows must still line up with that header (Task 4 test `test_exam_rows_keep_an_old_log_readable`).
2. A rough reset must never start the duck inside the stones, at any seed (Task 2 test `test_rough_start_never_starts_inside_the_floor`).
3. Films of a rough env must show the layout of the current attempt, not the first one (Task 2 test `test_new_layout_is_sent_to_the_renderer`).
4. `--start-from` must not change the source folder, and must refuse a run folder that already has snapshots and refuse being combined with `--resume` (Task 5 tests).
5. Roughness exactly 0 must reproduce the old flat world exactly, so Duck 2.1's flat score stays comparable (Task 2 test `test_zero_roughness_is_the_old_flat_world`).

---

### Task 1: Floor height maps and the height field model

**Files:**
- Create: `duck/floor.py`
- Test: `tests/test_floor.py`

**Interfaces:**
- Produces:
  - `MAX_ROUGHNESS: float = 0.01`
  - `FLOOR_SHAPE: dict` with keys `waves` (8), `wavelength_range` ((0.2, 0.5)), `stone_size` (0.06), `stone_height_range` ((0.35, 1.0)), `gap_share` (0.07), `bump_share` (0.5)
  - `floor_heights(rng: np.random.Generator, roughness: float) -> np.ndarray`: heights in metres, shape `(FLOOR_ROWS, FLOOR_COLS)`, min 0, max exactly `roughness` (all zeros when roughness is 0)
  - `load_floor_model() -> mujoco.MjModel`: the duck model with the plane replaced by a height field geom named `floor` (elevation `MAX_ROUGHNESS`)
  - `set_floor(model: mujoco.MjModel, heights: np.ndarray) -> None`: writes heights into `model.hfield_data` (as heights / `MAX_ROUGHNESS`)
  - `highest_floor_near(heights: np.ndarray, x: float, y: float, radius: float) -> float`: highest point in the square of half size `radius` around (x, y)
  - `FLOOR_ROWS, FLOOR_COLS: int` and `floor_index(x: float, y: float) -> tuple[int, int]`: the grid size and the (row, column) of the cell under a world point, matching how MuJoCo lays the field out (rows along y, columns along x)

Floor size: x from -2 m to 16 m (fastest attempt is 14 m), y from -1 m to 1 m, about 5 mm resolution (builder may choose 4 to 5 mm). Build the model from the text of `duck/assets/duck.xml` by adding an `<hfield>` asset and swapping the plane geom, as the sketch did; the geom keeps `contype="1" conaffinity="1"` and gets its position so the field covers the x range above.

- [ ] **Step 1: Write the failing tests**

```python
def test_zero_roughness_is_flat():
    assert np.all(floor_heights(np.random.default_rng(0), 0.0) == 0.0)

def test_highest_point_is_the_roughness():
    heights = floor_heights(np.random.default_rng(0), 0.006)
    assert heights.min() == pytest.approx(0.0) and heights.max() == pytest.approx(0.006)

def test_same_seed_same_layout_and_new_seed_new_layout():
    a = floor_heights(np.random.default_rng(1), MAX_ROUGHNESS)
    assert np.array_equal(a, floor_heights(np.random.default_rng(1), MAX_ROUGHNESS))
    assert not np.array_equal(a, floor_heights(np.random.default_rng(2), MAX_ROUGHNESS))

def test_floor_model_has_a_height_field_named_floor():
    model = load_floor_model()
    assert model.geom("floor").type == mujoco.mjtGeom.mjGEOM_HFIELD
    assert model.hfield_size[0][2] == pytest.approx(MAX_ROUGHNESS)

def test_set_floor_round_trips():
    model = load_floor_model()
    heights = floor_heights(np.random.default_rng(3), MAX_ROUGHNESS)
    set_floor(model, heights)
    assert model.hfield_data.max() == pytest.approx(1.0)

def test_highest_floor_near():
    heights = np.zeros((FLOOR_ROWS, FLOOR_COLS))
    heights[floor_index(1.0, 0.2)] = 0.008
    assert highest_floor_near(heights, 1.03, 0.2, radius=0.05) == pytest.approx(0.008)
    assert highest_floor_near(heights, 1.5, 0.2, radius=0.05) == 0.0
```

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_floor.py -v`; expect FAIL (no module).
- [ ] **Step 3: Implement `duck/floor.py`** with the interfaces above; the wave and stone formulas are the ones in the Global Constraints (the sketch is `C:\Users\pacoe\AppData\Local\Temp\claude\C--Users-pacoe-coding-projects\ef7fb3c1-25e0-40cd-93f0-0c3be2e96765\scratchpad\floor_previews.py` and `floor_combined.py`; reuse the shapes, not the throwaway structure). Generating one layout must take well under 0.2 s.
- [ ] **Step 4: Run** the tests; expect PASS.
- [ ] **Step 5: Commit** `git commit -m "Add rough floor height maps"` (with the trailer lines).

### Task 2: The duck world draws a floor at every reset

**Files:**
- Modify: `duck/duck_env.py` (constructor, `reset`, `render`)
- Test: `tests/test_duck_env.py`

**Interfaces:**
- Consumes: Task 1 (`floor_heights`, `load_floor_model`, `set_floor`, `highest_floor_near`, `MAX_ROUGHNESS`).
- Produces: `DuckWalkEnv(..., roughness: float | None = None)`. `None` means a random roughness from 0 to `MAX_ROUGHNESS` each attempt (the training default); a number fixes it. Exactly `0.0` loads the old plane model (`load_duck_model`) and draws no floor. The current attempt's value is `env.roughness`; its height map is `env.floor` (None on the plane model).

Draw order in `reset` (so seeds keep their old target speed, nudge and stride): target speed, keyframe, nudge, stride, then roughness and layout, then lift the torso by `highest_floor_near` around the start (radius 0.15 m) and `mj_forward`. After a new layout, if a renderer exists, upload the field again with `mujoco.mjr_uploadHField(self.model, <renderer context>, hfield id)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_zero_roughness_is_the_old_flat_world():
    old = DuckWalkEnv(roughness=0.0)
    assert old.model.geom("floor").type == mujoco.mjtGeom.mjGEOM_PLANE
    first, _ = old.reset(seed=0)
    # same as before this change: compare with a hard coded copy of the seed 0 start taken before editing reset,
    # or with a plane model env built the old way

def test_random_roughness_each_attempt():
    env = DuckWalkEnv()
    values = [(env.reset(seed=s), env.roughness)[1] for s in range(30)]
    assert all(0.0 <= v <= MAX_ROUGHNESS for v in values) and len(set(values)) == 30

def test_new_layout_every_attempt_same_for_same_seed():
    env = DuckWalkEnv(roughness=MAX_ROUGHNESS)
    env.reset(seed=4); a = env.floor.copy()
    env.reset(seed=5); assert not np.array_equal(a, env.floor)
    env.reset(seed=4); assert np.array_equal(a, env.floor)

def test_rough_start_never_starts_inside_the_floor():
    env = DuckWalkEnv(roughness=MAX_ROUGHNESS)
    for seed in range(40):
        env.reset(seed=seed)
        assert min((c.dist for c in env.data.contact[:env.data.ncon]), default=0.0) > -0.001

def test_new_layout_is_sent_to_the_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(duck_env_module.mujoco, "mjr_uploadHField", lambda *args: calls.append(args))
    env = DuckWalkEnv(render_mode="rgb_array", roughness=MAX_ROUGHNESS)
    env.reset(seed=0); env.reset(seed=1)
    assert len(calls) >= 2
```

For the first test, capture the seed 0 observation of the current code before editing (run it once and paste the numbers, or build an env with the plane model through the old path) and assert equality.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_duck_env.py -v`; expect the new tests to FAIL.
- [ ] **Step 3: Implement** the constructor parameter, the reset order above and the renderer upload.
- [ ] **Step 4: Run** the whole suite `.venv/Scripts/python -m pytest -q`; expect all PASS (the existing stride test, Gymnasium check and energy tests included).
- [ ] **Step 5: Measure** speed: time 2,000 random action steps for `DuckWalkEnv(roughness=0.0)` and `DuckWalkEnv(roughness=MAX_ROUGHNESS)` including resets every 1,000 steps; write both numbers in the report. If the rough one is more than 30% slower, report it as a concern (do not tune).
- [ ] **Step 6: Commit** `git commit -m "Draw a rough floor at every reset"`.

### Task 3: Overshooting the asked speed loses points

**Files:**
- Modify: `duck/rewards.py` (`_forward_part`)
- Test: `tests/test_rewards.py`

**Interfaces:** `compute_reward` keeps its signature. With `target_speed` infinite (no target), the forward part stays `forward_weight * speed`.

- [ ] **Step 1: Write the failing tests**

```python
def test_forward_points_below_target_are_unchanged():
    _, parts = compute_reward(0.3, 0.0, 0.0, target_speed=0.4)
    assert parts["forward"] == pytest.approx(3.0 * 0.3)

def test_overshoot_loses_points_like_undershoot():
    _, fast = compute_reward(0.5, 0.0, 0.0, target_speed=0.4)
    _, slow = compute_reward(0.3, 0.0, 0.0, target_speed=0.4)
    assert fast["forward"] == pytest.approx(slow["forward"])

def test_most_points_at_the_target_speed():
    _, exact = compute_reward(0.4, 0.0, 0.0, target_speed=0.4)
    assert exact["forward"] == pytest.approx(1.2)

def test_backwards_is_unchanged():
    _, parts = compute_reward(-0.2, 0.0, 0.0, target_speed=0.4)
    assert parts["forward"] == pytest.approx(-0.6)
```

Update any existing test that asserts the old cap (`min(speed, target)` above the target).

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_rewards.py -v`; expect FAIL on the overshoot test.
- [ ] **Step 3: Implement** the rule from the Global Constraints; update the comment above `_forward_part`.
- [ ] **Step 4: Run** the whole suite; expect PASS.
- [ ] **Step 5: Commit** `git commit -m "Score overshooting the asked speed like undershooting it"`.

### Task 4: The exam has a flat part and a rough part

**Files:**
- Modify: `duck/exam.py`, `duck/train.py` (`ExamCallback._examine` only)
- Test: `tests/test_exam.py`

**Interfaces:**
- Consumes: `DuckWalkEnv(roughness=...)` from Task 2, `MAX_ROUGHNESS`.
- Produces:
  - `take_exam(model, vec_normalize, attempts: int = EXAM_ATTEMPTS, roughness: float = 0.0) -> ExamResult` (one part, seeds 1000 to 1011 for both parts)
  - `@dataclass FullExamResult(score: float, falls: int, distance: float, attempts: int, flat: ExamResult, rough: ExamResult)`; score and distance are the averages of the parts, falls the sum, attempts 24
  - `take_full_exam(model, vec_normalize) -> FullExamResult`
  - `EXAM_LOG_HEADER = ["time", "timesteps", "score", "falls", "distance", "attempts", "flat_score", "flat_falls", "rough_score", "rough_falls"]`
  - `append_exam(run_dir, timesteps, result: FullExamResult, when)`: when the log exists, write only the columns of its own header (an old 5 column log stays 5 columns)
  - `read_exams(run_dir) -> list[dict]`: each dict has `steps, score, falls, attempts` (12 when the column is missing) and `flat_score, rough_score` (None when missing)
  - `ExamCallback` uses `take_full_exam`; its message reads `Exam at N steps: score S, falls F of 24 (flat A, rough B).`; best brain and patience use the combined score.

- [ ] **Step 1: Write the failing tests**

```python
def test_full_exam_averages_the_two_parts(): ...
    # with a small trained or untrained model from tests.test_train FAST settings:
    # result.score == mean(flat.score, rough.score), result.falls == flat.falls + rough.falls, result.attempts == 24

def test_rough_part_is_repeatable(): ...
    # two take_exam(..., roughness=MAX_ROUGHNESS) calls on the same brain give the same score

def test_exam_rows_keep_an_old_log_readable(tmp_path):
    (tmp_path / "exam.csv").write_text("time,timesteps,score,falls,distance\n2026-10-29T00:00:00,0,10.0,12,0.1\n")
    append_exam(tmp_path, 100, <FullExamResult with score 50>, datetime.now())
    rows = read_exams(tmp_path)
    assert [r["score"] for r in rows] == [10.0, 50.0] and rows[1]["attempts"] == 12

def test_new_log_has_both_parts(tmp_path):
    append_exam(tmp_path, 100, <FullExamResult flat 40 rough 60>, datetime.now())
    assert read_exams(tmp_path)[0]["flat_score"] == 40.0 and read_exams(tmp_path)[0]["rough_score"] == 60.0
```

Write the `...` tests fully, following the existing exam tests' fixtures. Existing tests (`test_training_takes_exams_and_keeps_best`, the patience tests with their `flat_exam` fixture) must be updated to the new result type, not deleted.

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_exam.py -v`; expect FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run** the whole suite; expect PASS. Update `best_exam_score` and `exam_at` (dashboard) only as far as needed to keep working on both log shapes.
- [ ] **Step 5: Commit** `git commit -m "Examine the duck on flat and rough floors"`.

### Task 5: Start a run from another run's brain

**Files:**
- Modify: `duck/train.py` (`train`, `main`, a new `_started_model`)
- Test: `tests/test_train.py`

**Interfaces:**
- Produces: `train(..., start_from: Path | None = None)` and CLI `--start-from FOLDER` (a folder holding `snapshots/`, such as a run folder, a run's `best/` or a duck's `brain/`). It loads the newest complete snapshot there with its sense scaling, attaches new duck worlds (seed 0), sets the step count to 0, saves it as the new run's step 0 snapshot and prints `Starting from the brain at N steps in FOLDER.` It keeps the brain's wobble, adapted step size and sense scaling (which keeps updating).
- Errors (plain messages, exit code 1 like the existing `FileExistsError` path): the run folder already has snapshots; `--start-from` together with `--resume`; no complete snapshot in the source.

- [ ] **Step 1: Write the failing tests**

```python
def test_start_from_another_runs_brain(tmp_path): ...
    # train a FAST source run, then train(target, steps, start_from=source):
    # target snapshot at 0 has the same policy weights as the source's newest snapshot,
    # target exam.csv first row is at 0 steps, the source folder's file list and modification times are unchanged

def test_start_from_refuses_a_used_run_folder(tmp_path): ...
def test_start_from_and_resume_together_is_refused(tmp_path): ...
def test_start_from_an_empty_folder_is_refused(tmp_path): ...
```

Write them fully with the existing `FAST` settings from `tests/test_train.py`.

- [ ] **Step 2: Run** the new tests; expect FAIL.
- [ ] **Step 3: Implement**, reusing `_load_snapshot`, `_refuse_used_run_folder` and `_save_snapshot_once`.
- [ ] **Step 4: Run** the whole suite; expect PASS.
- [ ] **Step 5: Commit** `git commit -m "Start a run from another run's brain"`.

### Task 6: Viewing tools and dashboard know the floor

**Files:**
- Modify: `duck/video.py` (`record_clip`, `main`), `duck/evaluate.py` (`evaluate_snapshot`, `main`), `duck/view_snapshot.py` (`main`), `duck/dashboard.py` (`make_snapshot_clip`), `duck/dashboard.html` (film caption)
- Test: `tests/test_video.py`, `tests/test_evaluate.py`, `tests/test_dashboard.py`

**Interfaces:**
- `record_clip(policy, seconds, fps, width, height, seed=0, on_frame=None, roughness: float = 0.0)`; `evaluate_snapshot(snapshot, attempts=10, seed=0, roughness: float = 0.0)`; CLI option `--roughness METRES` (default 0) on `duck.video`, `duck.evaluate` and `duck.view_snapshot`.
- The dashboard films with `roughness=MAX_ROUGHNESS` and `seed=EXAM_FIRST_SEED` (the first rough exam layout). Its film caption shows the combined score with both parts when the exam row has them, for example `After 2.5 million steps, exam score 900 (flat 950, rough 850), 1 fall of 24, asked to walk at 0.4 m/s, no wobble, on cobblestones`, and keeps the old wording (`of 12`, no floor words) for old rows. `run_status["clip_exam"]` carries `attempts`, `flat_score`, `rough_score` from `read_exams`.

- [ ] **Step 1: Write the failing tests**: `record_clip` with `roughness=MAX_ROUGHNESS` builds a rough env (monkeypatch `DuckWalkEnv` in `duck.video` to capture its arguments); `evaluate_snapshot(..., roughness=0.01)` passes it through the same way; `run_status` includes the parts for a new style row and `None` parts for an old row.
- [ ] **Step 2: Run** them; expect FAIL.
- [ ] **Step 3: Implement**; the page caption function `clipCaption` in `dashboard.html` handles both row shapes.
- [ ] **Step 4: Run** the whole suite; expect PASS. Then render one real dashboard clip of `duck_catalogue/duck_02_1/brain` on the rough floor into a temporary folder with `make_snapshot_clip` and open one frame image to check that stones are visible (report what you saw).
- [ ] **Step 5: Commit** `git commit -m "Show the rough floor in films and the dashboard"`.

### Task 7: Catalogue and docs

**Files:**
- Modify: `duck/catalogue.py` (`current_recipe`, `_recipe_lines`, `measure_brain`, `_results_lines`), `README.md`, `CLAUDE.md`
- Test: `tests/test_catalogue.py`

**Interfaces:**
- `current_recipe()` adds `"floor": {"max_roughness": MAX_ROUGHNESS, **FLOOR_SHAPE}` and `"forward_reward": "3.0 x (target - |speed - target|)"` (built from the weight, not hard coded text).
- Card recipe gets a `Floor` line: older recipes without `floor` say `- Floor: flat`; new ones list every shape number. A `Forward reward` wording follows the recipe; older recipes say the old cap rule.
- `measure_brain` stores `exam_score, exam_falls` (combined), `exam_flat_score, exam_rough_score`, `walking_test` (flat) and `walking_test_rough` (at `MAX_ROUGHNESS`), and gait summaries on both floors (`gait_at_judging_speed`, `gait_on_rough_floor`). `_results_lines` shows the rough lines only when present, so old cards read as before.

- [ ] **Step 1: Write the failing tests**: recipe has the floor and forward rule; an old recipe dict renders `- Floor: flat`; a results dict without rough keys renders like before.
- [ ] **Step 2: Run** them; expect FAIL.
- [ ] **Step 3: Implement**; then update README.md (floor, overshoot rule, `--start-from`, `--roughness`, two part exam) and the project CLAUDE.md (layout entry for `duck/floor.py`, the reward line for forward, the exam description, the start rule, the dashboard line).
- [ ] **Step 4: Run** the whole suite and `.venv/Scripts/python -m duck.catalogue cards`; expect PASS and unchanged cards for Duck 1, 2 and 2.1 apart from the new `Floor: flat` line.
- [ ] **Step 5: Commit** `git commit -m "Record the floor and reward rule in the catalogue"`.

---

After Task 7 the controller (not a subagent) creates `duck_catalogue/duck_03` ("Duck 3", summary "Duck 2.1's brain on a rough cobblestone floor, with an overshoot penalty"), starts `python -m duck.train --run runs/duck_03/from_duck_02_1 --steps 4000000 --start-from duck_catalogue/duck_02_1/brain`, and points the dashboard and phone page at it.
