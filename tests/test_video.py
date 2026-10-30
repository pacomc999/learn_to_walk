import imageio.v3 as iio
import numpy as np
import pytest
from tests.test_train import FLAT_SETTINGS
from duck.video import caption_text, add_caption, make_video, record_clip
from duck.train import train

def test_captions():
    assert caption_text(1, 25, 0) == "Snapshot 1 of 25, untrained"
    assert caption_text(7, 25, 1_200_000) == "Snapshot 7 of 25, after 1.2 million steps"
    assert caption_text(3, 25, 250_000) == "Snapshot 3 of 25, after 250 thousand steps"
    assert caption_text(25, 25, 30_000_000) == "Snapshot 25 of 25, after 30 million steps"
    assert caption_text(2, 25, 1_007_616) == "Snapshot 2 of 25, after 1 million steps"
    assert caption_text(1, 2, 999_500) == "Snapshot 1 of 2, after 1 million steps"
    assert caption_text(2, 25, 512) == "Snapshot 2 of 25, after 512 steps"

def test_caption_keeps_image_size():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert add_caption(frame, "Snapshot 1 of 1, untrained").shape == (120, 160, 3)

def test_caption_draws_light_text_on_dark_band():
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    captioned = add_caption(frame, "Snapshot 1 of 1, untrained")
    assert (captioned[:60] == frame[:60]).all()   # top half untouched
    assert captioned[-30:].max() > 200             # bright text
    assert captioned[-30:].min() < 100             # darkened band

def test_no_snapshots_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match="No snapshots found"):
        make_video(tmp_path)

def test_video_with_a_single_snapshot(tmp_path):
    train(tmp_path, total_steps=0, snapshot_every=1024, n_envs=2, settings=FLAT_SETTINGS,
          use_subprocess=False, ppo_overrides={"n_steps": 256, "batch_size": 64})
    path = make_video(tmp_path, seconds_per_clip=0.5, fps=30)
    assert path.exists()
    assert len(iio.imread(path)) == 15


# record_clip reports every picture it takes, with the observation and whether the duck has fallen
def test_record_clip_reports_each_frame():
    reports = []
    frames = record_clip(lambda observation: np.zeros(12), seconds=0.5, fps=10, width=80, height=60,
                         on_frame=lambda observation, fallen: reports.append((observation.shape, fallen)))
    assert len(reports) == len(frames) == 5
    assert reports[0] == ((39,), False)


# A stand-in environment that notes how it was built, then stops the recording
class _StopHere(Exception):
    pass


def _capture_env_arguments(monkeypatch, module):
    seen = {}

    def fake_env(**kwargs):
        seen.update(kwargs)
        raise _StopHere

    monkeypatch.setattr(module, "DuckWalkEnv", fake_env)
    return seen


# record_clip is flat by default and builds a rough floor when asked
def test_record_clip_passes_the_roughness(monkeypatch):
    import duck.video
    from duck.floor import MAX_ROUGHNESS
    seen = _capture_env_arguments(monkeypatch, duck.video)
    with pytest.raises(_StopHere):
        record_clip(lambda observation: np.zeros(12), seconds=0.1, fps=10)
    assert seen["roughness"] == 0.0
    with pytest.raises(_StopHere):
        record_clip(lambda observation: np.zeros(12), seconds=0.1, fps=10, roughness=MAX_ROUGHNESS)
    assert seen["roughness"] == MAX_ROUGHNESS
