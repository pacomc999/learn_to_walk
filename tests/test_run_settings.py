import json
from pathlib import Path

import pytest

from duck.bodies import body_to_dict
from duck.run_settings import (RunSettings, duck_settings, read_run_settings, settings_for_folder,
                               write_run_settings)
from duck.worlds import world_to_dict
from tests.worlds_for_tests import BODY, FLAT, STONES


# A run's settings are written next to it and read back the same, also from its best/ folder
def test_run_settings_round_trip(tmp_path):
    settings = RunSettings(Path("duck_catalogue/duck_03_1"), STONES, BODY)
    write_run_settings(tmp_path, settings)
    assert read_run_settings(tmp_path) == settings
    assert settings_for_folder(tmp_path / "best") == settings


# A folder with no settings (a run from before worlds) gives None
def test_folder_without_settings_gives_none(tmp_path):
    assert settings_for_folder(tmp_path) is None


# A duck's settings come from the frozen copies in its duck.json, also from its brain/ folder
def test_duck_settings_from_the_frozen_copies(tmp_path):
    duck_dir = tmp_path / "duck_09"
    duck_dir.mkdir()
    story = {"name": "Duck 9", "world": "flat_walking", "body": "standard",
             "world_settings": world_to_dict(FLAT), "body_settings": body_to_dict(BODY)}
    (duck_dir / "duck.json").write_text(json.dumps(story))
    assert duck_settings(duck_dir) == RunSettings(duck_dir, FLAT, BODY)
    assert settings_for_folder(duck_dir / "brain") == RunSettings(duck_dir, FLAT, BODY)


# A duck without frozen copies (from before worlds) is refused with a plain message
def test_duck_without_a_world_is_refused(tmp_path):
    (tmp_path / "duck.json").write_text(json.dumps({"name": "Duck 0"}))
    with pytest.raises(ValueError, match="no world"):
        duck_settings(tmp_path)


# A tool given a run uses the run's world and body, each replaced by a named file when given
def test_resolve_from_a_run_with_overrides(tmp_path):
    from duck.run_settings import resolve_settings
    write_run_settings(tmp_path, RunSettings(None, FLAT, BODY))
    assert resolve_settings(tmp_path / "best", None, None) == (FLAT, BODY)
    world, body = resolve_settings(tmp_path, "stones_up_to_4cm", None)
    assert world == STONES and body == BODY


# A folder from before worlds asks for --world in a plain sentence, and works once it is given
def test_old_folder_asks_for_a_world(tmp_path):
    from duck.run_settings import resolve_settings
    with pytest.raises(ValueError, match="--world"):
        resolve_settings(tmp_path, None, None)
    assert resolve_settings(tmp_path, "flat_walking", None) == (FLAT, BODY)
