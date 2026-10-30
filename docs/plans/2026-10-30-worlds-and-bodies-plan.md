# Worlds and Bodies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Worlds and bodies become named JSON files; every duck and run carries frozen copies, and every tool reads them instead of shared settings in the code.

**Architecture:** Two small data modules (`duck/worlds.py`, `duck/bodies.py`) load and check the JSON files. A scene module builds the MuJoCo XML from a body, a floor and the world's extras. `DuckWalkEnv`, the exam, training, the catalogue and the tools all take a `World` and a `Body`, found from a folder through `duck/run_settings.py`.

**Tech Stack:** Python 3.12, MuJoCo, Gymnasium, Stable Baselines3, pytest, `xml.etree.ElementTree`.

**Spec:** `docs/designs/2026-10-30-worlds-and-bodies-design.md`

## Global Constraints

- No em or en dashes in any visible text (docs, captions, printed messages, card lines).
- Beginner readable code: clear names, short single purpose functions, a comment on each section, matching the surrounding files. Python 3.12, no TypeScript.
- Commit messages: short, present tense, no prefix, ending with the session's attribution lines. Commit only with Francisco's go ahead for this plan.
- Run only the test files of the modules a task changes (`.venv/Scripts/python -m pytest -q -x tests/test_<module>.py`), never the whole suite.
- Anything using `SubprocVecEnv` stays under `if __name__ == "__main__":`.
- The frozen copies in `duck.json` and `run_settings.json` are what training and testing use; the files in `worlds/` and `bodies/` are templates only.
- The standard body must give exactly today's model (the XML text of `duck/assets/duck.xml` unchanged), and floors keep `MAX_ROUGHNESS` 0.04 as their model height, so existing results do not move.
- Every brain fits every body (bodies change numbers only).

## Review Focus

1. A tool given an old run folder with no `run_settings.json` and no `--world`: expect one plain sentence asking for `--world`, no traceback. Pinned in Task 7.
2. A world file edited after a duck was made: the duck keeps its frozen world. Pinned in Task 6.
3. Rocks on the stone floor: the parked rock must sit clear of the tallest stones and not touch the duck before the first throw. Pinned in Task 3.
4. A run resumed after its duck's `duck.json` changed: the run keeps its own `run_settings.json`. Pinned in Task 5.
5. Typos in a world or body file (unknown key, unknown motor group or part, a speed range with low above high, a negative mass): refused with a plain message naming the file and the key. Pinned in Tasks 1 and 2.

---

### Task 0: Start from a clean state (Francisco's go ahead needed)

**Files:** none new.

- [ ] **Step 1:** Ask Francisco to commit today's work (stone exam, dashboard started by training) and to decide whether Duck 3.1's second run (`runs/duck_03_1/from_duck_03_1/best`) is kept. Do not continue until the working tree is clean.
- [ ] **Step 2: Record the baseline.** For every duck in `duck_catalogue/`, run the walking test on seeds 0 to 9 flat, and for ducks trained on stones also at their own tallest stones (Duck 3: 0.01, Duck 3.1: 0.03 and 0.04). Save the lines to the scratchpad as `baseline_walking.txt`, for example `duck_03_1 flat 10 of 10`, `duck_03_1 0.03 6 of 10`.

Run: `.venv/Scripts/python -m duck.evaluate duck_catalogue/duck_03_1/brain --roughness 0.03`
Expected: a line like `6 of 10 attempts walked. Not there yet.`

---

### Task 1: Worlds

**Files:**
- Create: `duck/worlds.py`, `worlds/flat_walking.json`, `worlds/stones_up_to_1cm.json`, `worlds/stones_up_to_3cm.json`, `worlds/stones_up_to_4cm.json`, `worlds/flat_running.json`
- Test: `tests/test_worlds.py`

**Interfaces:**
- Produces:
  - `WORLDS_DIR = Path("worlds")`
  - `@dataclass(frozen=True) class Rocks: mass: float; every_seconds: float = 2.0; speed: float = 4.0`
  - `@dataclass(frozen=True) class Backpack: mass: float`
  - `@dataclass(frozen=True) class World: name: str; description: str; tallest_stones: float; target_speed_range: tuple[float, float]; reward: RewardWeights; rocks: Rocks | None = None; backpack: Backpack | None = None; notes: str = ""`
  - `world_from_dict(data: dict, source: str = "world") -> World` (raises `ValueError` with a plain message naming `source` and the key)
  - `world_to_dict(world: World) -> dict` (the file layout of the spec: `floor`, `extras` sections)
  - `load_world(name: str, folder: Path = WORLDS_DIR) -> World`
- `RewardWeights` (in `duck/rewards.py`) becomes `@dataclass(frozen=True)` so a `World` can hold it.

- [ ] **Step 1: Write the failing tests** in `tests/test_worlds.py`:

```python
def test_world_file_round_trip():
    world = load_world("stones_up_to_4cm")
    assert world.tallest_stones == 0.04 and world.target_speed_range == (0.2, 0.7)
    assert world.reward == RewardWeights(3.0, 0.2, 1.5, 0.5) and world.rocks is None and world.backpack is None
    assert world_from_dict(world_to_dict(world)) == world

def test_world_with_extras():
    data = {**world_to_dict(load_world("flat_walking")),
            "extras": {"rocks": {"mass": 0.1}, "backpack": {"mass": 0.2}}}
    world = world_from_dict(data)
    assert world.rocks == Rocks(0.1, 2.0, 4.0) and world.backpack == Backpack(0.2)

@pytest.mark.parametrize("change, word", [
    ({"colour": "red"}, "colour"),
    ({"extras": {"wind": {}}}, "wind"),
    ({"target_speed_range": [0.7, 0.2]}, "target_speed_range"),
    ({"extras": {"rocks": {"mass": -1}}}, "mass"),
    ({"floor": {"tallest_stones": -0.01}}, "tallest_stones"),
])
def test_bad_world_is_refused_plainly(change, word):
    data = {**world_to_dict(load_world("flat_walking")), **change}
    with pytest.raises(ValueError, match=word):
        world_from_dict(data, source="worlds/test.json")

def test_the_five_worlds_match_the_old_recipes():
    expected = {"flat_walking": (0.0, (0.2, 0.7)), "stones_up_to_1cm": (0.01, (0.2, 0.7)),
                "stones_up_to_3cm": (0.03, (0.2, 0.7)), "stones_up_to_4cm": (0.04, (0.2, 0.7)),
                "flat_running": (0.0, (0.2, 1.5))}
    for name, (stones, speeds) in expected.items():
        world = load_world(name)
        assert (world.tallest_stones, world.target_speed_range) == (stones, speeds)
```

- [ ] **Step 2: Run them and see them fail.** Run: `.venv/Scripts/python -m pytest -q -x tests/test_worlds.py`. Expected: FAIL, `No module named 'duck.worlds'`.
- [ ] **Step 3: Implement `duck/worlds.py`** with the interfaces above, and write the five world files in the spec's layout (one line description each; `flat_walking` notes: Ducks 1, 2 and 2.1 trained under an older forward reward, without the overshoot rule). Checks: only the known top keys (`name, description, floor, target_speed_range, reward, extras, notes`), known extras (`rocks, backpack`), masses and speeds above 0, `tallest_stones` from 0 up to `MAX_ROUGHNESS`, speed range low below high, the four reward weights present.
- [ ] **Step 4: Run them and see them pass.** Same command. Expected: all pass.
- [ ] **Step 5: Commit** `duck/worlds.py duck/rewards.py worlds/ tests/test_worlds.py`: `Add world files and their loader`

---

### Task 2: Bodies

**Files:**
- Create: `duck/bodies.py`, `bodies/standard.json`
- Test: `tests/test_bodies.py`

**Interfaces:**
- Produces:
  - `BODIES_DIR = Path("bodies")`
  - `MOTOR_GROUPS = {"neck": [...], "hips": [4 hip motors], "knees": [...], "ankles": [...], "toes": [...]}` (names from `MOTOR_NAMES`)
  - `FOOT_GEOMS = ["left_foot", "left_toes", "right_foot", "right_toes"]`
  - `@dataclass(frozen=True) class Body: name: str; description: str; motor_force_limits: dict[str, float]; motor_stiffness: dict[str, float]; part_masses: dict[str, float]; foot_friction: float | None = None; notes: str = ""` (dicts default empty; use `field(default_factory=dict)` and `frozen=True` with `hash=False` where needed)
  - `body_from_dict(data: dict, source: str = "body") -> Body`, `body_to_dict(body: Body) -> dict`, `load_body(name: str, folder: Path = BODIES_DIR) -> Body`
  - `STANDARD_BODY: Body` (loaded from `bodies/standard.json`)
  - `body_xml(body: Body) -> str`: the standard XML text, unchanged when the body has no changes; otherwise edited with ElementTree: `forcerange="-x x"` and `kp` on the named actuators, `mass` on the named bodies' geoms (MuJoCo body mass comes from its geoms), the first number of `friction` on `FOOT_GEOMS`.

- [ ] **Step 1: Write the failing tests** in `tests/test_bodies.py`:

```python
def test_standard_body_is_todays_duck():
    assert body_xml(STANDARD_BODY) == DUCK_XML_PATH.read_text()

def test_strong_legs_double_the_hips_and_knees():
    body = body_from_dict({"name": "strong_legs", "description": "x",
                           "motor_force_limits": {"hips": 8, "knees": 8}})
    model = mujoco.MjModel.from_xml_string(body_xml(body))
    assert list(model.actuator("left_hip_side").forcerange) == [-8, 8]
    assert list(model.actuator("right_knee").forcerange) == [-8, 8]
    assert list(model.actuator("left_ankle").forcerange) == [-3, 3]  # untouched

def test_stiffness_mass_and_friction():
    body = body_from_dict({"name": "b", "description": "x", "motor_stiffness": {"left_knee": 30},
                           "part_masses": {"torso": 1.0}, "foot_friction": 1.5})
    model = mujoco.MjModel.from_xml_string(body_xml(body))
    assert model.actuator("left_knee").gainprm[0] == 30
    assert model.body("torso").mass[0] == pytest.approx(1.0)
    assert model.geom("left_toes").friction[0] == 1.5

@pytest.mark.parametrize("data, word", [
    ({"motor_force_limits": {"wings": 5}}, "wings"),
    ({"part_masses": {"tail": 1}}, "tail"),
    ({"part_masses": {"torso": -1}}, "torso"),
    ({"size": 2}, "size"),
])
def test_bad_body_is_refused_plainly(data, word):
    with pytest.raises(ValueError, match=word):
        body_from_dict({"name": "b", "description": "x", **data}, source="bodies/test.json")
```

- [ ] **Step 2: Run and see them fail.** `.venv/Scripts/python -m pytest -q -x tests/test_bodies.py`. Expected: FAIL, no module.
- [ ] **Step 3: Implement `duck/bodies.py` and `bodies/standard.json`** (`{"name": "standard", "description": "The duck as designed in duck/assets/duck.xml", "notes": ""}`). For `part_masses`, look up the body names in `duck.xml`; if a body has several geoms, share the new mass in proportion to their old masses (write the rule in a comment). A position actuator's `kp` sets both `gainprm[0]` and `biasprm[1]` (minus `kp`); set the `kp` attribute and let MuJoCo do it.
- [ ] **Step 4: Run and see them pass.**
- [ ] **Step 5: Commit** `duck/bodies.py bodies/ tests/test_bodies.py`: `Add body files and build the duck from them`

---

### Task 3: The environment takes a world and a body (with rocks and a backpack)

**Files:**
- Create: `duck/scene.py`
- Modify: `duck/floor.py` (`_floor_xml`, `load_floor_model` take the body XML; remove `TRAINING_ROUGHNESS`), `duck/duck_env.py` (constructor, `_load_model`, speeds, reward, rocks; remove `TARGET_SPEED_RANGE`), `duck/rocks.py` (the thrower), `duck/challenges.py` (`add_backpack` moves to `duck/scene.py`, re-exported)
- Test: `tests/test_scene.py`, `tests/test_duck_env.py`, `tests/test_floor.py`, `tests/test_rocks.py`

**Interfaces:**
- Consumes: `World`, `Rocks`, `Backpack`, `load_world` (Task 1); `Body`, `body_xml`, `STANDARD_BODY` (Task 2).
- Produces:
  - `scene_xml(world: World, body: Body, flat: bool, most: float = MAX_ROUGHNESS) -> str` and `load_scene(...) -> mujoco.MjModel` in `duck/scene.py`: the body XML, the height field floor unless `flat`, then the backpack, then the rock (rock parked at z `most + radius + 0.01` on the stone floor).
  - `DuckWalkEnv(world: World, body: Body, render_mode=None, target_speed=None, width=FILM_WIDTH, height=FILM_HEIGHT, roughness=None, floor_most=MAX_ROUGHNESS)`: `world` and `body` are required keywords (no hidden default). Flat plane when `roughness == 0.0`, or when `roughness is None` and `world.tallest_stones == 0`. Draws stones from 0 to `world.tallest_stones`, speeds from `world.target_speed_range`, rewards with `world.reward`. `self.rock_thrower: RockThrower | None`.
  - `class RockThrower` in `duck/rocks.py`: `__init__(model, rocks: Rocks)`, `reset(rng: np.random.Generator)`, `before_step(data)`, `after_step(data)`, `hits: int`, `throws: int`. The throwing rules are today's `RockWorld` (`FIRST_THROW`, side, lead, arc, `ROCK_LIFETIME`), with `every_seconds` and `speed` from `Rocks`. The env passes its own `np_random` at reset, so the attempt's seed fixes the throws.
  - `RockWorld` and `BackpackWorld` become thin helpers that build a `DuckWalkEnv` with `dataclasses.replace(world, rocks=...)` or `backpack=...`; `TallStoneWorld` passes `floor_most`.

- [ ] **Step 1: Write the failing tests:**

```python
# tests/test_scene.py
def test_standard_flat_scene_is_todays_model():
    assert scene_xml(load_world("flat_walking"), STANDARD_BODY, flat=True) == DUCK_XML_PATH.read_text()

def test_rock_parks_clear_of_the_tallest_stones():
    world = replace(load_world("stones_up_to_4cm"), rocks=Rocks(0.4))
    env = DuckWalkEnv(world=world, body=STANDARD_BODY, roughness=0.04)
    env.reset(seed=0)
    for _ in range(50):  # 1 s, before the first throw at 2 s
        env.step(np.zeros(12))
    assert env.rock_thrower.hits == 0
    assert not any("rock" in pair for pair in contact_geom_names(env))  # helper in the test file

# tests/test_duck_env.py
def test_env_follows_the_world():
    world = replace(load_world("stones_up_to_4cm"), target_speed_range=(0.3, 0.31))
    env = DuckWalkEnv(world=world, body=STANDARD_BODY)
    env.reset(seed=1)
    assert 0.3 <= env.target_speed <= 0.31 and 0 < env.roughness <= 0.04

def test_flat_world_uses_the_plain_plane():
    env = DuckWalkEnv(world=load_world("flat_running"), body=STANDARD_BODY)
    env.reset(seed=0)
    assert env.floor is None

def test_rock_world_throws_on_both_floors():
    for name, roughness in (("flat_walking", 0.0), ("stones_up_to_4cm", 0.02)):
        world = replace(load_world(name), rocks=Rocks(0.1))
        env = DuckWalkEnv(world=world, body=STANDARD_BODY, roughness=roughness, target_speed=0.4)
        env.reset(seed=0)
        for _ in range(150):  # 3 s: one throw at 2 s
            env.step(np.zeros(12))
        assert env.rock_thrower.throws == 1

def test_same_seed_same_throws():  # two envs, same seed, same rock position after 2.2 s
    ...  # assert the rock qpos arrays are equal

def test_backpack_world_carries_it():
    world = replace(load_world("flat_walking"), backpack=Backpack(0.5))
    env = DuckWalkEnv(world=world, body=STANDARD_BODY)
    assert env.model.geom("backpack").mass[0] == pytest.approx(0.5)
```

Existing tests in `test_duck_env.py`, `test_floor.py` and `test_rocks.py` that build `DuckWalkEnv()` get `world=` and `body=` (a `FLAT` and a `STONES` world fixture at the top of each file).
- [ ] **Step 2: Run and see them fail.** `.venv/Scripts/python -m pytest -q -x tests/test_scene.py tests/test_duck_env.py`
- [ ] **Step 3: Implement** the interfaces above. Keep `floor_heights` and `set_floor` unchanged so floors stay identical.
- [ ] **Step 4: Run the four test files and see them pass.** `.venv/Scripts/python -m pytest -q -x tests/test_scene.py tests/test_duck_env.py tests/test_floor.py tests/test_rocks.py`
- [ ] **Step 5: Commit**: `Let the environment take a world and a body`

(Other modules break between Tasks 3 and 7; that is expected, and each task fixes its own callers.)

---

### Task 4: The exam follows the world

**Files:**
- Modify: `duck/exam.py`
- Test: `tests/test_exam.py`

**Interfaces:**
- Consumes: `DuckWalkEnv(world=, body=)` (Task 3).
- Produces: `_exam_attempt_on(model, vec_normalize, world, body, seed, roughness, speed) -> tuple[float, bool, float]`, `exam_speeds(world: World, attempts: int = EXAM_ATTEMPTS)`, `stone_exam_heights(world: World, attempts=STONE_EXAM_ATTEMPTS)`, `stone_exam_speeds(world: World, attempts=STONE_EXAM_ATTEMPTS)`, `take_exam(model, vec_normalize, world, body, attempts=EXAM_ATTEMPTS, roughness=0.0)`, `take_full_exam(model, vec_normalize, world: World, body: Body) -> FullExamResult`. Same rules as today, reading the world instead of `TRAINING_ROUGHNESS` and `TARGET_SPEED_RANGE`; the world's extras are on in every attempt.

- [ ] **Step 1: Update the tests:** the `monkeypatch.setattr(duck.exam, "TRAINING_ROUGHNESS", ...)` tests now pass a world instead (`load_world("stones_up_to_4cm")` or `load_world("flat_running")`); add:

```python
def test_exam_speeds_follow_the_world():
    speeds = exam_speeds(load_world("flat_running"))
    assert speeds[0] == pytest.approx(0.2) and speeds[-1] == pytest.approx(1.5)

def test_exam_uses_the_worlds_extras(monkeypatch):
    seen = []
    monkeypatch.setattr(duck.exam, "_exam_attempt_on",
        lambda model, vec_normalize, world, body, seed, roughness, speed: seen.append(world.rocks) or (1.0, False, 0.5))
    take_full_exam(None, None, replace(load_world("stones_up_to_4cm"), rocks=Rocks(0.1)), STANDARD_BODY)
    assert seen == [Rocks(0.1)] * 24
```

- [ ] **Step 2: Run and see them fail.** `.venv/Scripts/python -m pytest -q -x tests/test_exam.py`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and see them pass.**
- [ ] **Step 5: Commit**: `Take the exam in the duck's own world and body`

---

### Task 5: Run settings and training

**Files:**
- Create: `duck/run_settings.py`
- Modify: `duck/train.py`
- Test: `tests/test_run_settings.py`, `tests/test_train.py`, `tests/test_exam.py` (only the training tests that call `train`)

**Interfaces:**
- Consumes: Tasks 1 to 4.
- Produces:
  - `RUN_SETTINGS_NAME = "run_settings.json"`
  - `@dataclass(frozen=True) class RunSettings: duck_dir: Path | None; world: World; body: Body`
  - `write_run_settings(run_dir: Path, settings: RunSettings) -> None`, `read_run_settings(run_dir: Path) -> RunSettings | None`
  - `duck_settings(duck_dir: Path) -> RunSettings` (from the frozen `world_settings` and `body_settings` in `duck.json`; plain `ValueError` if the duck has none)
  - `settings_for_folder(folder: Path) -> RunSettings | None`: a duck folder, its `brain/`, a run folder, or its `best/`
  - `train(run_dir, total_steps, settings: RunSettings, snapshot_every=..., ...)`: writes `run_settings.json` for a new run; a resumed run reads its own and ignores `settings` (pass `None` on resume)
  - `make_duck_env(world, body)`, `build_vec_env(n_envs, seed, world, body, use_subprocess=True)`, `ExamCallback(..., world, body)`
  - Command line: `--duck DUCK_DIR` needed for a new run; `--resume` needs only `--run`; a resume without `run_settings.json` stops with a plain message.

- [ ] **Step 1: Write the failing tests:**

```python
# tests/test_run_settings.py
def test_run_settings_round_trip(tmp_path):
    settings = RunSettings(Path("duck_catalogue/duck_03_1"), load_world("stones_up_to_4cm"), STANDARD_BODY)
    write_run_settings(tmp_path, settings)
    assert read_run_settings(tmp_path) == settings
    assert settings_for_folder(tmp_path / "best") == settings

def test_folder_without_settings_gives_none(tmp_path):
    assert settings_for_folder(tmp_path) is None

# tests/test_train.py
def test_new_run_writes_its_settings(tmp_path):
    train(tmp_path, total_steps=0, settings=FLAT_SETTINGS, **FAST)
    assert read_run_settings(tmp_path) == FLAT_SETTINGS

def test_resume_keeps_its_own_settings(tmp_path):
    train(tmp_path, total_steps=512, snapshot_every=512, settings=FLAT_SETTINGS, **FAST)
    train(tmp_path, total_steps=1024, snapshot_every=512, settings=None, resume=True, **FAST)
    assert read_run_settings(tmp_path) == FLAT_SETTINGS
```

Every existing `train(...)` call in the tests gets `settings=FLAT_SETTINGS` (a module level `RunSettings(None, load_world("flat_walking"), STANDARD_BODY)`).
- [ ] **Step 2: Run and see them fail.** `.venv/Scripts/python -m pytest -q -x tests/test_run_settings.py tests/test_train.py`
- [ ] **Step 3: Implement.** `start_dashboard` is unchanged.
- [ ] **Step 4: Run and see them pass** (plus `tests/test_exam.py`).
- [ ] **Step 5: Commit**: `Give every run its own world and body`

---

### Task 6: The catalogue names worlds and bodies, and keep compares fairly

**Files:**
- Modify: `duck/catalogue.py`, `duck/evaluate.py` (only `evaluate_snapshot`; its command line is Task 7)
- Test: `tests/test_catalogue.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `duck_settings`, `settings_for_folder` (Task 5); `take_full_exam(model, vn, world, body)` (Task 4).
- Produces:
  - `evaluate_snapshot(snapshot, world: World, body: Body, attempts=10, seed=0, roughness=0.0) -> EvaluationResult` in `duck/evaluate.py` (Task 7 uses it)
  - `new_duck(duck_dir: Path, name: str, world: World, body: Body) -> Path`: writes `"world"`, `"body"`, `"world_settings": world_to_dict(world)`, `"body_settings": body_to_dict(body)` into `duck.json`. `current_recipe()` drops `floor`, `target_speed_range` and `reward` (they live in the world now); old recipes keep theirs and the card still reads them.
  - `measure_brain(snapshot, run_dir, world, body) -> dict`: results gain `"exam_world": world.name`.
  - `keep_brain(duck_dir, run_dir, snapshot, force=False)`: takes the world and body from the run's settings. If the kept results have no `exam_world` or another one, measure the kept brain again in this world before comparing, and print `The kept brain was examined in another world, so it took this world's exam first: <score>.`
  - Card: two new lines after the summary, `- World: <name>, <description>` and `- Body: <name>, <description>`. When a run started from another duck's brain trained in a different body, the run's line on the card says `(its brain came from the <body name> body)`.
  - CLI: `new DUCK_DIR NAME --world NAME [--body NAME]` (body defaults to `standard`).

- [ ] **Step 1: Write the failing tests:**

```python
def test_new_duck_freezes_world_and_body(tmp_path):
    duck_dir = new_duck(tmp_path / "duck_09", "Duck 9", load_world("stones_up_to_4cm"), STANDARD_BODY)
    story = json.loads((duck_dir / "duck.json").read_text())
    assert story["world"] == "stones_up_to_4cm" and story["body"] == "standard"
    assert world_from_dict(story["world_settings"]) == load_world("stones_up_to_4cm")

def test_editing_the_world_file_later_does_not_change_the_duck(tmp_path):
    worlds = tmp_path / "worlds"; worlds.mkdir()
    (worlds / "w.json").write_text(json.dumps(world_to_dict(load_world("flat_walking")) | {"name": "w"}))
    duck_dir = new_duck(tmp_path / "duck_09", "Duck 9", load_world("w", worlds), STANDARD_BODY)
    (worlds / "w.json").write_text(json.dumps(world_to_dict(load_world("flat_running")) | {"name": "w"}))
    assert duck_settings(duck_dir).world.target_speed_range == (0.2, 0.7)

def test_keep_reexamines_a_brain_from_another_world(tmp_path, small_run, capsys):
    ...  # kept results with "exam_world": "stones_up_to_3cm"; keep in a stones_up_to_4cm run
    assert "took this world's exam first" in capsys.readouterr().out

def test_card_shows_world_and_body(tmp_path):
    ...  # write_card on a new duck; assert "- World: stones_up_to_4cm, " and "- Body: standard, " in card.md
```

Existing catalogue tests that call `new_duck` or monkeypatch `TRAINING_ROUGHNESS` pass a world instead.
- [ ] **Step 2: Run and see them fail.** `.venv/Scripts/python -m pytest -q -x tests/test_catalogue.py`
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run and see them pass.**
- [ ] **Step 5: Commit**: `Let ducks name their world and body`

---

### Task 7: The tools find the world and body, and can try others

**Files:**
- Modify: `duck/run_settings.py` (one helper), `duck/evaluate.py`, `duck/view_snapshot.py`, `duck/video.py`, `duck/drive.py`, `duck/dashboard.py`, `duck/challenges.py`
- Test: `tests/test_run_settings.py`, `tests/test_evaluate.py`, `tests/test_dashboard.py`, `tests/test_challenges.py`, `tests/test_drive.py`

**Interfaces:**
- Produces:
  - `resolve_settings(folder: Path, world_name: str | None, body_name: str | None) -> tuple[World, Body]` in `duck/run_settings.py`: the folder's own settings, each replaced by a named file when given; with no settings and no world name, raises `ValueError("This folder has no world yet (it is older than worlds). Add --world NAME, for example --world flat_walking.")`. Body falls back to `standard`.
  - `add_settings_arguments(parser)`: adds `--world` and `--body` to a tool's parser.
  - `evaluate_snapshot` as given in Task 6; `record_clip(policy, world, body, ...)`; `Driver(policy, speed, body)` (drive keeps its flat floor: it uses `flat_walking`'s floor with the brain's body); the dashboard films at `world.tallest_stones`; challenges build their worlds from the duck's world and body (`dataclasses.replace`) and turn energy into joules with `world.reward.energy`.
  - Every tool's `main` catches that `ValueError` and prints it, exit code 1, no traceback.

- [ ] **Step 1: Write the failing tests:**

```python
# tests/test_run_settings.py
def test_resolve_from_a_duck_with_overrides():
    world, body = resolve_settings(Path("duck_catalogue/duck_03_2/brain"), "stones_up_to_4cm", None)
    assert world.name == "stones_up_to_4cm" and body.name == "standard"

def test_old_folder_asks_for_a_world(tmp_path):
    with pytest.raises(ValueError, match="--world"):
        resolve_settings(tmp_path, None, None)

# tests/test_evaluate.py
def test_evaluate_command_without_world_prints_a_sentence(tmp_path, capsys, monkeypatch):
    ...  # main() on a run folder with a snapshot but no run_settings.json
    # assert SystemExit code 1 and "--world" in capsys.readouterr().out, no "Traceback"

# tests/test_challenges.py
def test_challenge_energy_uses_the_worlds_weight(): ...  # world with energy 3.0 halves the joules of the same points
```

Existing tool tests get worlds and bodies where they build environments.
- [ ] **Step 2: Run and see them fail.** `.venv/Scripts/python -m pytest -q -x tests/test_run_settings.py tests/test_evaluate.py`
- [ ] **Step 3: Implement** tool by tool.
- [ ] **Step 4: Run the five test files and see them pass.**
- [ ] **Step 5: Commit**: `Let the tools use the duck's world and body or try others`

---

### Task 8: Move the existing ducks over

**Files:**
- Modify: `duck_catalogue/duck_*/duck.json` (six ducks), `runs/**/run_settings.json` (git-ignored, written locally)
- Script: a one-off script in the scratchpad (not kept)

- [ ] **Step 1: Write the names and frozen copies** into each `duck.json`: Duck 1, 2, 2.1 `flat_walking`; Duck 3 `stones_up_to_1cm`; Duck 3.1 `stones_up_to_4cm` with a note in `notes` that the first run used `stones_up_to_3cm`; Duck 3.2 `flat_running`; all `standard`.
- [ ] **Step 2: Write `run_settings.json`** in every run folder under `runs/duck_*/` whose world is known from its duck (Duck 3.1: `from_duck_03` gets `stones_up_to_3cm`, `from_duck_03_1` gets `stones_up_to_4cm`). Leave `runs/early_experiments/` alone.
- [ ] **Step 3: Compare with the baseline.** Rerun every line of Task 0's `baseline_walking.txt`, now without `--roughness` tricks: flat with `--roughness 0`, stones with the duck's own height.

Run: `.venv/Scripts/python -m duck.evaluate duck_catalogue/duck_03_1/brain --roughness 0.03`
Expected: every result equal to the baseline line. Any difference stops the task: find out why before going on.
- [ ] **Step 4: Rewrite the cards.** `.venv/Scripts/python -m duck.catalogue cards`. Expected: each card gains its World and Body lines and nothing else changes (check with `git diff duck_catalogue`).
- [ ] **Step 5: Commit**: `Give every duck its world and body`

---

### Task 9: Documents

**Files:**
- Modify: `docs/guide.md`, `README.md`, `CLAUDE.md`

- [ ] **Step 1:** `docs/guide.md`: a "Worlds and bodies" section (what each is, the file layout with one example of each, rocks and backpack extras, frozen copies, `--world` and `--body` on the tools, how to make a new duck with `new --world --body`); the training command with `--duck`; remove the text about switching `TRAINING_ROUGHNESS` and `TARGET_SPEED_RANGE` per duck.
- [ ] **Step 2:** `README.md` table: the training command with `--duck`. `CLAUDE.md`: layout lines for `duck/worlds.py`, `duck/bodies.py`, `duck/scene.py`, `duck/run_settings.py`, `worlds/`, `bodies/`; the env, exam, train and catalogue lines updated; the reward and floor sections say the values come from the world.
- [ ] **Step 3: Check for dashes.** Search the changed docs for em and en dashes. Expected: none.
- [ ] **Step 4: Commit**: `Explain worlds and bodies in the guide`
