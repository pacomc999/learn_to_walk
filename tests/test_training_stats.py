import csv

from duck.training_stats import STAT_KEYS, latest_update_stats
from duck.train import train
from tests.test_train import FAST


# Before the first update there is nothing to log
def test_no_stats_before_an_update():
    assert latest_update_stats({}) is None


# Training writes one row of PPO numbers per brain update
def test_training_writes_stats(tmp_path):
    train(tmp_path, total_steps=1536, snapshot_every=1536, exam_every=10_000, **FAST)
    with (tmp_path / "training_stats.csv").open(newline="") as log_file:
        rows = list(csv.DictReader(log_file))
    assert [int(row["timesteps"]) for row in rows] == [512, 1024, 1536]
    for row in rows:
        for key in STAT_KEYS:
            float(row[key])  # every number is present
    assert float(rows[0]["learning_rate"]) == 3e-4
