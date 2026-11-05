"""The library, replays and the web server's API, with a library folder of its own."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    from studio import library, server
    monkeypatch.setattr(library, "LIBRARY", tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.StudioHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def call(base, path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{base}/api/{path}", data=data, method=method or ("POST" if data else "GET"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_catalog_lists_animals_worlds_and_settings(studio):
    catalog = call(studio, "catalog")
    assert [animal["key"] for animal in catalog["animals"]] == ["duck", "puppy", "custom"]
    assert len(catalog["world_templates"]) == 11
    assert catalog["brain_default"]["layers"] == [256, 256]


def test_page_is_served(studio):
    with urllib.request.urlopen(studio + "/") as response:
        assert b"Walk Studio" in response.read()


def test_creature_preview_has_shapes_and_facts(studio):
    preview = call(studio, "preview/creature", {"animal": "puppy", "values": {"overall_size": 120}})
    assert len(preview["shapes"]) > 10
    assert len(preview["frames"][0]) == 7 * len(preview["bodies"])
    assert preview["facts"]["motors"] == 14


def test_drop_test_reports_whether_it_stood(studio):
    replay = call(studio, "preview/drop", {"animal": "duck", "values": {}, "seconds": 1})
    assert replay["stats"]["stood"] is True
    assert len(replay["frames"]) == 26


def test_library_saves_lists_and_deletes(studio):
    saved = call(studio, "creatures", {"name": "Long legs", "animal": "duck", "values": {"leg_length": 150}})
    assert call(studio, "creatures")[0]["id"] == saved["id"]
    again = call(studio, "creatures", {**saved, "name": "Longer legs"})
    assert again["id"] == saved["id"]
    call(studio, f"creatures/{saved['id']}", method="DELETE")
    assert call(studio, "creatures") == []


def test_bad_requests_get_a_plain_message(studio):
    with pytest.raises(urllib.error.HTTPError) as error:
        call(studio, "trainings/../../etc")
    assert error.value.code in (400, 404)


def test_a_walker_can_be_shared_and_opened_again(tmp_path, monkeypatch):
    from studio import library, sharing
    monkeypatch.setattr(library, "LIBRARY", tmp_path)
    folder = library.training_folder("original")
    library.write_json(folder / "setup.json", {"id": "original", "name": "Speedy", "creature": {"animal": "duck"}})
    library.write_json(folder / "status.json", {"state": "training", "steps": 5})
    (folder / "brain").mkdir()
    (folder / "brain" / "best.zip").write_bytes(b"brain")
    (folder / "log.txt").write_text("private")
    data, name = sharing.pack("original")
    assert name == "Speedy.walker"
    opened = sharing.unpack(data)
    copy = library.training_folder(opened)
    assert opened != "original" and (copy / "brain" / "best.zip").read_bytes() == b"brain"
    assert not (copy / "log.txt").exists()
    assert library.read_json(copy / "status.json")["state"] == "stopped"


def test_a_bad_walker_file_is_refused(tmp_path, monkeypatch):
    import io
    import zipfile
    from studio import library, sharing
    monkeypatch.setattr(library, "LIBRARY", tmp_path)
    with pytest.raises(ValueError):
        sharing.unpack(b"not a zip")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("setup.json", '{"name": "sneaky"}')
        archive.writestr("../../evil.txt", "no")
        archive.writestr("brain/../../evil2.txt", "no")
    opened = sharing.unpack(buffer.getvalue())
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()
    assert sorted(p.name for p in library.training_folder(opened).iterdir()) == ["setup.json", "status.json"]


def test_a_creature_without_motors_cannot_start_training(tmp_path, monkeypatch):
    from studio import library, manager
    monkeypatch.setattr(library, "LIBRARY", tmp_path)
    body = {"parts": [{"id": "stick", "parent": "torso", "joint": "fixed", "z": -1}]}
    with pytest.raises(ValueError, match="no motors"):
        manager.create_training({"creature": {"animal": "custom", "values": {"body": body}}, "world": {}})
