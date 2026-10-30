from datetime import datetime

import pytest

from duck.exam import (EXAM_LOG_HEADER, ExamResult, FullExamResult, append_exam, best_exam_score, exam_at,
                       read_exams, rough_exam_heights, take_exam, take_full_exam)
from duck.floor import MAX_ROUGHNESS
from duck.train import build_vec_env, train
from tests.test_train import FAST


# A full exam result whose two parts have the given scores
def full_result(flat_score: float, rough_score: float, falls: int = 0) -> FullExamResult:
    flat = ExamResult(score=flat_score, falls=falls, distance=1.0)
    rough = ExamResult(score=rough_score, falls=falls, distance=2.0)
    return FullExamResult(score=(flat_score + rough_score) / 2, falls=2 * falls, distance=1.5,
                          attempts=24, flat=flat, rough=rough)


# The exam is noise-free, so the same brain gets the same result twice
def test_exam_is_repeatable(tmp_path):
    model = train(tmp_path, total_steps=0, **FAST)
    vec_normalize = build_vec_env(1, seed=0, use_subprocess=False)
    first = take_exam(model, vec_normalize, attempts=3)
    second = take_exam(model, vec_normalize, attempts=3)
    vec_normalize.close()
    assert first == second
    assert 0 <= first.falls <= 3


# Taking the exam must not change the sense scaling used for training
def test_exam_leaves_scaling_alone(tmp_path):
    model = train(tmp_path, total_steps=0, **FAST)
    vec_normalize = build_vec_env(1, seed=0, use_subprocess=False)
    count_before = vec_normalize.obs_rms.count
    take_exam(model, vec_normalize, attempts=2)
    assert vec_normalize.obs_rms.count == count_before
    vec_normalize.close()


# The best score is read back from the log, skipping broken rows
def test_best_exam_score_from_log(tmp_path):
    assert best_exam_score(tmp_path) is None
    append_exam(tmp_path, 100, full_result(50.0, 50.0, falls=3), datetime.now())
    append_exam(tmp_path, 200, full_result(80.0, 80.0, falls=1), datetime.now())
    with (tmp_path / "exam.csv").open("a") as log_file:
        log_file.write("broken,row\n")
    assert best_exam_score(tmp_path) == 80.0


# The exam of one step count is read back, or None if that step count had no exam
def test_exam_at_a_step_count(tmp_path):
    assert exam_at(tmp_path, 100) is None
    append_exam(tmp_path, 100, full_result(50.0, 50.0, falls=3), datetime.now())
    with (tmp_path / "exam.csv").open("a") as log_file:
        log_file.write("broken,row\n")
    assert exam_at(tmp_path, 100) == {"steps": 100, "score": 50.0, "falls": 6, "attempts": 24,
                                      "flat_score": 50.0, "rough_score": 50.0}
    assert exam_at(tmp_path, 200) is None


# Training writes exam results and keeps a best brain that opens like a run folder
def test_training_takes_exams_and_keeps_best(tmp_path):
    train(tmp_path, total_steps=1024, snapshot_every=1024, exam_every=512, **FAST)
    lines = (tmp_path / "exam.csv").read_text().strip().splitlines()
    assert lines[0] == ",".join(EXAM_LOG_HEADER)
    exam_steps = [int(line.split(",")[1]) for line in lines[1:]]
    assert exam_steps == sorted(set(exam_steps))  # each step count examined once
    assert exam_steps[0] == 0 and exam_steps[-1] == 1024
    assert len(list((tmp_path / "best" / "snapshots").glob("snapshot_*_norm.pkl"))) == 1  # only the newest best


# A resumed run only saves a new best brain if it beats the best of earlier sessions
def test_resume_keeps_earlier_best(tmp_path):
    train(tmp_path, total_steps=512, snapshot_every=512, exam_every=512, **FAST)
    append_exam(tmp_path, 1, full_result(1e9, 1e9), datetime.now())
    best_before = sorted((tmp_path / "best" / "snapshots").iterdir())
    train(tmp_path, total_steps=1024, snapshot_every=512, exam_every=512, resume=True, **FAST)
    assert sorted((tmp_path / "best" / "snapshots").iterdir()) == best_before


# An exam that always gives the same score, so no exam after the first is a new best
@pytest.fixture
def flat_exam(monkeypatch):
    import duck.train
    monkeypatch.setattr(duck.train, "take_full_exam", lambda model, vec_normalize: full_result(10.0, 10.0, falls=12))


# Training stops by itself after `patience` exams in a row without a new best
def test_training_stops_without_improvement(tmp_path, flat_exam, capsys):
    model = train(tmp_path, total_steps=8192, snapshot_every=8192, exam_every=512, patience=2, **FAST)
    assert model.num_timesteps < 8192
    assert "Stopping early" in capsys.readouterr().out
    assert list((tmp_path / "best" / "snapshots").glob("snapshot_000000000000.zip"))  # the best brain is kept


# Patience 0 means training never stops early
def test_patience_zero_never_stops(tmp_path, flat_exam):
    model = train(tmp_path, total_steps=2048, snapshot_every=2048, exam_every=512, patience=0, **FAST)
    assert model.num_timesteps >= 2048


# The full exam is the plain average of the flat part and the rough part
def test_full_exam_averages_the_two_parts(tmp_path):
    model = train(tmp_path, total_steps=0, **FAST)
    vec_normalize = build_vec_env(1, seed=0, use_subprocess=False)
    result = take_full_exam(model, vec_normalize)
    vec_normalize.close()
    assert result.score == pytest.approx((result.flat.score + result.rough.score) / 2)
    assert result.distance == pytest.approx((result.flat.distance + result.rough.distance) / 2)
    assert result.falls == result.flat.falls + result.rough.falls
    assert result.attempts == 24


# The rough part uses the same floors every time, so the same brain scores the same
def test_rough_part_is_repeatable(tmp_path):
    model = train(tmp_path, total_steps=0, **FAST)
    vec_normalize = build_vec_env(1, seed=0, use_subprocess=False)
    first = take_exam(model, vec_normalize, attempts=3, roughness=MAX_ROUGHNESS)
    second = take_exam(model, vec_normalize, attempts=3, roughness=MAX_ROUGHNESS)
    vec_normalize.close()
    assert first.score == second.score


# An old 5 column log stays 5 columns and is still read, with 12 attempts and no part scores
def test_exam_rows_keep_an_old_log_readable(tmp_path):
    (tmp_path / "exam.csv").write_text("time,timesteps,score,falls,distance\n2026-10-29T00:00:00,0,10.0,12,0.1\n")
    append_exam(tmp_path, 100, full_result(50.0, 50.0), datetime.now())
    rows = read_exams(tmp_path)
    assert [r["score"] for r in rows] == [10.0, 50.0] and rows[1]["attempts"] == 12
    assert rows[1]["flat_score"] is None and rows[1]["rough_score"] is None
    assert len((tmp_path / "exam.csv").read_text().splitlines()[-1].split(",")) == 5


# A new log has all columns, including both part scores
def test_new_log_has_both_parts(tmp_path):
    append_exam(tmp_path, 100, full_result(40.0, 60.0), datetime.now())
    rows = read_exams(tmp_path)
    assert rows[0]["flat_score"] == 40.0 and rows[0]["rough_score"] == 60.0 and rows[0]["attempts"] == 24


# The rough part climbs evenly: 12 heights from a twelfth of the most up to the most, each one higher
def test_rough_exam_heights_climb_to_the_most():
    heights = rough_exam_heights()
    assert len(heights) == 12
    assert heights[0] == pytest.approx(MAX_ROUGHNESS / 12) and heights[-1] == pytest.approx(MAX_ROUGHNESS)
    assert all(low < high for low, high in zip(heights, heights[1:]))


# With one height per attempt, each attempt gets its own floor height
def test_take_exam_uses_one_height_per_attempt(tmp_path, monkeypatch):
    import duck.exam
    seen = []
    monkeypatch.setattr(duck.exam, "_exam_attempt_on",
                        lambda model, vec_normalize, seed, roughness: seen.append((seed, roughness)) or (1.0, False, 0.5))
    take_exam(None, None, roughness=[0.01, 0.02, 0.03], attempts=3)
    assert seen == [(1000, 0.01), (1001, 0.02), (1002, 0.03)]
