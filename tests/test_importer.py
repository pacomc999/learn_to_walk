"""Bringing ducks over from learn_to_walk. Needs a learn_to_walk folder: set LEARN_TO_WALK, or keep it next to
this project (otherwise these tests are skipped)."""
import json
import os
from pathlib import Path

import pytest

from studio import importer, library

FOLDER = Path(os.environ.get("LEARN_TO_WALK", Path(__file__).parent.parent.parent / "learn_to_walk"))
needs_folder = pytest.mark.skipif(not (FOLDER / "duck_catalogue").is_dir(), reason="no learn_to_walk folder")


def test_bodies_become_duck_settings():
    strong = {"motor_force_limits": {"neck": 10, "hips": 40, "knees": 40, "ankles": 30, "toes": 5},
              "motor_stiffness": {"neck": 40, "hips": 200, "knees": 200, "ankles": 200, "toes": 50}}
    values = importer.creature_values(strong)
    assert values["motor_type"] == "position"
    assert values["hip_strength"] == pytest.approx(1000) and values["stiffness"] == pytest.approx(1000)
    force = importer.creature_values({"motor_type": "force", **{"motor_force_limits": strong["motor_force_limits"]}})
    assert force["knee_strength"] == pytest.approx(100)


def test_worlds_become_studio_worlds():
    values = importer.world_values({"floor": {"tallest_stones": 0.04}, "target_speed_range": [0.2, 0.7],
                                    "reward": {"forward": 3, "alive": 0.2, "energy": 1.5, "sideways": 0.5,
                                               "end_stops": 0.1}})
    assert values["terrain"] == "cobblestones" and values["terrain_height"] == 4
    assert values["end_stops"] and values["end_stop_cost"] == 0.1


@needs_folder
def test_duck_3_3_walks_as_well_as_in_learn_to_walk(tmp_path, monkeypatch):
    monkeypatch.setattr(library, "LIBRARY", tmp_path)
    ducks = {duck["key"]: duck for duck in importer.find_ducks(str(FOLDER))}
    training_id = importer.import_duck(ducks["duck_03_3"]["folder"])
    status = json.loads((library.training_folder(training_id) / "status.json").read_text())
    assert status["state"] == "finished"
    assert status["last_exam"]["falls"] == 0          # his card: exam 1188, falls 0 of 12
    assert status["best_quality"] > 85
    # it keeps learn_to_walk's settled step size, so carrying on does not wreck it
    import zipfile
    with zipfile.ZipFile(library.training_folder(training_id) / "brain" / "latest.zip") as archive:
        assert json.loads(archive.read("data"))["learning_rate"] == pytest.approx(1e-5)
