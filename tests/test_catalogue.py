import json

import pytest

from duck import catalogue
from duck.catalogue import film_duck, keep_brain, new_duck, write_index
from duck.snapshots import list_snapshots
from duck.train import DEFAULT_PPO_SETTINGS, train
from tests.test_train import FAST
from tests.worlds_for_tests import BODY, FLAT, STONES
from duck.worlds import load_world, world_from_dict, world_to_dict


# Short films and few gait attempts keep these tests quick
@pytest.fixture(autouse=True)
def quick_measurements(monkeypatch):
    monkeypatch.setattr(catalogue, "FILM_SECONDS", 0.4)
    monkeypatch.setattr(catalogue, "GAIT_SEEDS", (0,))


# A tiny trained run with snapshots at 0 and 512 steps
@pytest.fixture
def small_run(tmp_path):
    run_dir = tmp_path / "runs" / "small"
    train(run_dir, total_steps=512, snapshot_every=512, exam_every=10_000, **FAST)
    return run_dir


# A new duck freezes the current recipe and gets a card
def test_new_duck_freezes_the_recipe(tmp_path):
    duck_dir = new_duck(tmp_path / "duck_catalogue" / "duck_01", "Duck 1", FLAT, BODY)
    story = json.loads((duck_dir / "duck.json").read_text())
    assert story["name"] == "Duck 1"
    assert story["recipe"]["brain"] == "SymmetricPolicy"
    assert story["recipe"]["ppo"] == DEFAULT_PPO_SETTINGS
    assert story["world"] == "flat_walking" and story["body"] == "standard"
    assert story["world_settings"]["reward"]["energy"] == 1.5
    assert "# Duck 1" in (duck_dir / "card.md").read_text()


# Two ducks can not share a folder
def test_new_duck_refuses_a_used_folder(tmp_path):
    new_duck(tmp_path / "duck_01", "Duck 1", FLAT, BODY)
    with pytest.raises(FileExistsError):
        new_duck(tmp_path / "duck_01", "Duck 1 again", FLAT, BODY)


# Keeping a brain copies it, measures it, makes a film and fills the card
def test_keep_brain_fills_the_duck_folder(tmp_path, small_run):
    duck_dir = new_duck(tmp_path / "duck_catalogue" / "duck_01", "Duck 1", FLAT, BODY)
    snapshot = list_snapshots(small_run)[-1]
    assert keep_brain(duck_dir, small_run, snapshot)
    kept = list_snapshots(duck_dir / "brain")
    assert [snap.steps for snap in kept] == [snapshot.steps]
    results = json.loads((duck_dir / "results.json").read_text())
    assert results["steps"] == snapshot.steps and 0 <= results["exam_falls"] <= 24
    assert "exam_lower_score" not in results and results["exam_world"] == "flat_walking"  # flat world: no halves
    assert (duck_dir / "duck_01.mp4").stat().st_size > 0
    assert "catalogue film duck_catalogue/duck_01" in (duck_dir / "card.md").read_text()
    assert "fixed attempts" in (duck_dir / "card.md").read_text()


# A brain that does not beat the kept one is not kept, unless forced
def test_keep_brain_only_keeps_a_better_one(tmp_path, small_run):
    duck_dir = new_duck(tmp_path / "duck_catalogue" / "duck_01", "Duck 1", FLAT, BODY)
    first, second = list_snapshots(small_run)[0], list_snapshots(small_run)[-1]
    keep_brain(duck_dir, small_run, first)
    results = json.loads((duck_dir / "results.json").read_text())
    results["exam_score"] = 1e9  # pretend the kept brain is unbeatable
    (duck_dir / "results.json").write_text(json.dumps(results))
    assert not keep_brain(duck_dir, small_run, second)
    assert [snap.steps for snap in list_snapshots(duck_dir / "brain")] == [first.steps]
    assert keep_brain(duck_dir, small_run, second, force=True)
    assert [snap.steps for snap in list_snapshots(duck_dir / "brain")] == [second.steps]


# The list page has one row per duck, oldest folder first
def test_index_lists_every_duck(tmp_path):
    ducks = tmp_path / "duck_catalogue"
    new_duck(ducks / "duck_02", "Duck 2", FLAT, BODY)
    new_duck(ducks / "duck_01", "Duck 1", FLAT, BODY)
    index = write_index(ducks).read_text()
    assert index.index("[Duck 1]") < index.index("[Duck 2]")
    assert "none yet" in index


# A world's card values: the floor (its tallest stones and every shape number) and the forward reward rule
def test_recipe_records_floor_and_forward_rule():
    recipe = catalogue.card_recipe({"recipe": catalogue.current_recipe(),
                                    "world_settings": world_to_dict(load_world("stones_up_to_3cm"))})
    assert recipe["floor"]["max_roughness"] == 0.03 and recipe["floor"]["rough_exam"] == "spread"
    assert recipe["floor"]["stone_size"] == 0.06 and recipe["floor"]["waves"] == 8
    assert recipe["forward_reward"] == "3.0 x (target - |speed - target|)"


# A recipe frozen before the floor existed says flat and keeps the old cap wording
def test_old_recipe_card_lines_say_flat():
    recipe = catalogue.card_recipe({"recipe": catalogue.current_recipe(), "world_settings": world_to_dict(FLAT)})
    del recipe["floor"], recipe["forward_reward"]
    text = "\n".join(catalogue._recipe_lines(recipe))
    assert "- Floor: flat" in text
    assert "per m/s up to the target speed" in text


# A recipe training on stones lists the floor numbers and the overshoot rule
def test_new_recipe_card_lines_list_the_floor():
    recipe = catalogue.card_recipe({"recipe": catalogue.current_recipe(),
                                    "world_settings": world_to_dict(load_world("stones_up_to_3cm"))})
    text = "\n".join(catalogue._recipe_lines(recipe))
    assert "- Floor: random cobblestones up to 0.03 m" in text
    assert "spread evenly from flat up to the full height" in text
    assert "stones 0.06 m" in text and "gaps 0.07" in text
    assert "3.0 x (target - |speed - target|)" in text


# A recipe training on the flat floor says so, and that the exam is flat only
def test_flat_training_recipe_says_flat():
    recipe = catalogue.card_recipe({"recipe": catalogue.current_recipe(), "world_settings": world_to_dict(FLAT)})
    assert recipe["floor"]["max_roughness"] == 0.0 and recipe["floor"]["rough_exam"] == "none"
    text = "\n".join(catalogue._recipe_lines(recipe))
    assert "- Floor: flat in training" in text and "exam is flat only" in text


OLD_RESULTS = {
    "from_run": "runs/x", "steps": 1000, "measured": "2026-01-01 00:00",
    "exam_score": 986.4, "exam_falls": 0, "walking_test": "10 of 10",
    "gait_at_judging_speed": {"feet_in_air": 0.1, "one_foot_down": 0.8, "both_feet_down": 0.1,
                              "joules_per_metre": 12.3, "speed": 0.46, "falls": 0},
}


# Results without rough keys render as before
def test_old_results_render_as_before():
    text = "\n".join(catalogue._results_lines(OLD_RESULTS, "duck_x"))
    assert "- Exam (12 fixed attempts, no wobble): score 986, falls 0 of 12" in text
    assert "rough" not in text.lower()


# Results with rough keys show both parts, of 24, and the rough lines
def test_new_results_show_both_floors():
    results = {**OLD_RESULTS, "exam_flat_score": 900.0, "exam_rough_score": 500.0, "exam_score": 700.0,
               "exam_falls": 3, "walking_test_rough": "7 of 10",
               "gait_on_rough_floor": {**OLD_RESULTS["gait_at_judging_speed"], "speed": 0.3}}
    text = "\n".join(catalogue._results_lines(results, "duck_x"))
    assert "score 700, falls 3 of 24" in text
    assert "flat 900" in text and "rough 500" in text
    assert "7 of 10" in text and "rough floor (1 cm stones)" in text.lower()


# Results of the stone exam show the lower and taller halves, the attempt count and the tallest stones
def test_stone_exam_results_show_both_halves():
    results = {**OLD_RESULTS, "exam_attempts": 24, "exam_lower_score": 990.0, "exam_taller_score": 870.0,
               "exam_score": 930.0, "exam_falls": 2, "rough_floor_height": 0.04}
    line = catalogue._exam_line(results)
    assert line == ("- Exam (24 fixed attempts, floor heights spread from flat to 4 cm, no wobble): "
                    "score 930, falls 2 of 24 (lower stones 990, taller stones 870)")


# A duck started from another brain gets the extra recipe line, a duck from zero does not
def test_started_run_card_has_the_extra_line():
    recipe = catalogue.card_recipe({"recipe": catalogue.current_recipe(), "world_settings": world_to_dict(FLAT)})
    old_story = {"starts_from": "zero (same seed and untrained brain as Duck 1)"}
    new_story = {"starts_from": "Duck 2.1's best brain (duck_catalogue/duck_02_1/brain, 4.0M steps, exam 986)"}
    old_text = "\n".join(catalogue._recipe_lines(recipe, catalogue._started_from_brain(old_story)))
    new_text = "\n".join(catalogue._recipe_lines(recipe, catalogue._started_from_brain(new_story)))
    assert catalogue.STARTED_FROM_BRAIN_LINE not in old_text
    assert catalogue.STARTED_FROM_BRAIN_LINE in new_text
    lines = new_text.split("\n")
    assert lines[1] == catalogue.STARTED_FROM_BRAIN_LINE and lines[0].startswith("- Brain:")


# The film command makes a lost film again from the kept brain, and says so when there is no brain
def test_film_duck_makes_the_film_again(tmp_path, small_run):
    duck_dir = new_duck(tmp_path / "duck_catalogue" / "duck_01", "Duck 1", FLAT, BODY)
    assert film_duck(duck_dir) is None
    keep_brain(duck_dir, small_run, list_snapshots(small_run)[-1])
    (duck_dir / "duck_01.mp4").unlink()
    assert film_duck(duck_dir) == duck_dir / "duck_01.mp4"
    assert (duck_dir / "duck_01.mp4").stat().st_size > 0


# A new duck freezes its world and body, so editing the world file later does not change it
def test_editing_the_world_file_later_does_not_change_the_duck(tmp_path):
    import json as json_module
    from duck.run_settings import duck_settings
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    (worlds / "w.json").write_text(json_module.dumps(world_to_dict(FLAT) | {"name": "w"}))
    duck_dir = new_duck(tmp_path / "duck_09", "Duck 9", load_world("w", worlds), BODY)
    (worlds / "w.json").write_text(json_module.dumps(world_to_dict(load_world("flat_running")) | {"name": "w"}))
    assert duck_settings(duck_dir).world.target_speed_range == (0.2, 0.7)
    story = json.loads((duck_dir / "duck.json").read_text())
    assert world_from_dict(story["world_settings"]).name == "w"


# Keeping a brain in a run of another world first gives the kept brain this world's exam
def test_keep_reexamines_a_brain_from_another_world(tmp_path, small_run, capsys):
    duck_dir = new_duck(tmp_path / "duck_catalogue" / "duck_01", "Duck 1", FLAT, BODY)
    first, second = list_snapshots(small_run)[0], list_snapshots(small_run)[-1]
    keep_brain(duck_dir, small_run, first)
    results = json.loads((duck_dir / "results.json").read_text())
    results["exam_world"] = "stones_up_to_3cm"
    results["exam_score"] = 1e9  # an unbeatable score from another world does not count
    (duck_dir / "results.json").write_text(json.dumps(results))
    keep_brain(duck_dir, small_run, second)
    assert "took this world's exam first" in capsys.readouterr().out


# The card names the duck's world and body
def test_card_shows_world_and_body(tmp_path):
    duck_dir = new_duck(tmp_path / "duck_09", "Duck 9", STONES, BODY)
    card = (duck_dir / "card.md").read_text()
    assert f"- World: stones_up_to_4cm, {STONES.description}" in card
    assert f"- Body: standard, {BODY.description}" in card
    assert "- Floor: random cobblestones up to 0.04 m" in card


# A run whose brain came from another body says so on the card
def test_run_line_names_the_body_its_brain_came_from():
    line = catalogue._run_lines([{"folder": "runs/x", "dates": "d", "steps": "s", "outcome": "o.",
                                  "brain_from_body": "standard"}])[0]
    assert line.endswith("(its brain came from the standard body)")
