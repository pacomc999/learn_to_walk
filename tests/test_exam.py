from datetime import datetime

from duck.exam import (EXAM_ATTEMPTS, ExamResult, append_exam, best_exam_score, take_exam)
from duck.train import build_vec_env, train
from tests.test_train import FAST


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
    append_exam(tmp_path, 100, ExamResult(score=50.0, falls=3, distance=1.0), datetime.now())
    append_exam(tmp_path, 200, ExamResult(score=80.0, falls=1, distance=2.0), datetime.now())
    with (tmp_path / "exam.csv").open("a") as log_file:
        log_file.write("broken,row\n")
    assert best_exam_score(tmp_path) == 80.0


# Training writes exam results and keeps a best brain that opens like a run folder
def test_training_takes_exams_and_keeps_best(tmp_path):
    train(tmp_path, total_steps=1024, snapshot_every=1024, exam_every=512, **FAST)
    lines = (tmp_path / "exam.csv").read_text().strip().splitlines()
    assert lines[0] == "time,timesteps,score,falls,distance"
    exam_steps = [int(line.split(",")[1]) for line in lines[1:]]
    assert exam_steps == sorted(set(exam_steps))  # each step count examined once
    assert exam_steps[0] == 0 and exam_steps[-1] == 1024
    assert list((tmp_path / "best" / "snapshots").glob("snapshot_*_norm.pkl"))


# A resumed run only saves a new best brain if it beats the best of earlier sessions
def test_resume_keeps_earlier_best(tmp_path):
    train(tmp_path, total_steps=512, snapshot_every=512, exam_every=512, **FAST)
    append_exam(tmp_path, 1, ExamResult(score=1e9, falls=0, distance=99.0), datetime.now())
    best_before = sorted((tmp_path / "best" / "snapshots").iterdir())
    train(tmp_path, total_steps=1024, snapshot_every=512, exam_every=512, resume=True, **FAST)
    assert sorted((tmp_path / "best" / "snapshots").iterdir()) == best_before
