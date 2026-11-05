"""The server's features that need a trained walker: watch, race, report card, live mode, pictures and sharing."""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from studio import train
from studio.learning import clean_learning
from studio.worlds import builtin_worlds


@pytest.fixture(scope="module")
def studio(tmp_path_factory):
    from studio import library, server
    root = tmp_path_factory.mktemp("library")
    old_library, old_clean = library.LIBRARY, train.clean_learning
    library.LIBRARY = root

    def tiny(values):
        cleaned = clean_learning(values)
        cleaned.update(million_steps=0.002, parallel_worlds=1, steps_per_update=256, batch_size=64, epochs=1)
        return cleaned
    train.clean_learning = tiny
    folder = library.training_folder("walker")
    folder.mkdir(parents=True)
    setup = {"id": "walker", "name": "Tiny", "creature": {"animal": "puppy", "values": {}},
             "world": builtin_worlds()[0].to_dict(), "brain": {"layers": [16, 16]}, "learning": {}}
    (folder / "setup.json").write_text(json.dumps(setup))
    train.run(folder, resume=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.StudioHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    library.LIBRARY, train.clean_learning = old_library, old_clean


def call(base, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{base}/api/{path}", data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        content = response.read()
    return content if raw else json.loads(content)


def test_watch_in_another_world_with_the_brain_at_work(studio):
    world = builtin_worlds()[1].to_dict()
    replay = call(studio, "trainings/walker/watch", {"world": world, "speed": 0.4, "seconds": 2})
    assert replay["ground"]["kind"] == "heights"
    assert replay["cells"] and len(replay["cells"][0]) == 2
    assert len(replay["commands"]) == len(replay["frames"])


def test_race_puts_walkers_in_lanes(studio):
    replay = call(studio, "race", {"ids": ["walker", "walker"], "world": builtin_worlds()[0].to_dict(), "seconds": 2})
    assert len(replay["results"]) == 1  # the same walker twice races once
    assert replay["labels"][0]["text"] == "Tiny"


def test_one_report_card_level(studio):
    result = call(studio, "trainings/walker/challenges/ice", {"level": 40})
    assert isinstance(result["passed"], bool) and result["replay"]["frames"]


def test_live_mode_streams_frames_and_takes_commands(studio):
    header = call(studio, "live", {"training": "walker", "speed": 0.4})
    time.sleep(0.6)
    update = call(studio, f"live/{header['id']}?after=-1")
    assert update["frames"] and update["running"]
    call(studio, f"live/{header['id']}", {"speed": 0.8, "push": "left"})
    time.sleep(0.2)
    assert call(studio, f"live/{header['id']}?after=-1")["state"]["asked"] == pytest.approx(0.8)
    call(studio, f"live/{header['id']}", {"stop": True})


def test_thumbnail_and_sharing(studio):
    request = urllib.request.Request(f"{studio}/api/trainings/walker/thumbnail", data=b"\xff\xd8 tiny jpeg", method="POST",
                                     headers={"Content-Type": "image/jpeg"})
    urllib.request.urlopen(request).read()
    shared = call(studio, "trainings/walker/share", raw=True)
    request = urllib.request.Request(f"{studio}/api/trainings/open", data=shared, method="POST",
                                     headers={"Content-Type": "application/zip"})
    opened = json.loads(urllib.request.urlopen(request).read())["id"]
    walkers = {walker["id"]: walker for walker in call(studio, "trainings")}
    assert opened in walkers and walkers[opened]["thumbnail"]
