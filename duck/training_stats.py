"""A log of PPO's own numbers after each brain update, to see how training behaves.

Written to <run>/training_stats.csv, one row per update:
- approx_kl: how far the brain moved in this update (bigger means a bigger change)
- clip_fraction: share of moves where PPO's safety limit (the clip) kicked in
- explained_variance: how well the judge predicted the scores (1 is perfect, 0 is no better than a guess)
- entropy_loss, value_loss: the curiosity term and the judge's error
- std: the average wobble of the motor commands
- learning_rate: the step size used
"""
import csv
from datetime import datetime
from pathlib import Path

STATS_LOG_NAME = "training_stats.csv"

# The PPO numbers to keep, by their names in Stable-Baselines3's logger
STAT_KEYS = ["approx_kl", "clip_fraction", "explained_variance", "entropy_loss",
             "value_loss", "std", "learning_rate"]
STATS_LOG_HEADER = ["time", "timesteps"] + STAT_KEYS

# Printed if the stats log is open in another program
LOCKED_STATS_LOG_WARNING = "Warning: could not write the training stats log (is it open elsewhere?). Training goes on."


# The PPO numbers of the latest update, or None if no update has happened yet
def latest_update_stats(logger_values: dict) -> dict | None:
    if "train/approx_kl" not in logger_values:
        return None
    return {key: logger_values.get(f"train/{key}") for key in STAT_KEYS}


# Add one row of PPO numbers to the log, writing the header first if the log is new
def append_stats(run_dir: Path, timesteps: int, stats: dict, when: datetime) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / STATS_LOG_NAME
    is_new_log = not log_path.exists()
    try:
        with log_path.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            if is_new_log:
                writer.writerow(STATS_LOG_HEADER)
            writer.writerow([when.isoformat(), timesteps] + [stats[key] for key in STAT_KEYS])
    except OSError:
        print(LOCKED_STATS_LOG_WARNING, flush=True)
