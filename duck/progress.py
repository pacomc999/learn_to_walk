"""Log training progress, check for a stall, and draw the progress chart.

Run from the project root to see the chart and the stall check:
    python -m duck.progress runs/duck_v1
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw straight to a file, no window needed
import matplotlib.pyplot as plt


# Minutes without a new best score before training counts as stalled
STALL_MINUTES = 60

# File names inside a run folder
LOG_NAME = "progress.csv"
CHART_NAME = "progress.png"
LOG_HEADER = ["time", "timesteps", "mean_reward"]


# One line of the progress log: when it was written, the step count and the average score
@dataclass(frozen=True)
class ProgressRow:
    time: datetime
    timesteps: int
    mean_reward: float


# The full path of the progress log in a run folder
def _log_path(run_dir: Path) -> Path:
    return run_dir / LOG_NAME


# Add one row to the log, writing the header first if the log is new
def append_progress(run_dir: Path, timesteps: int, mean_reward: float, when: datetime) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(run_dir)
    is_new_log = not log_path.exists()
    with log_path.open("a", newline="") as log_file:
        writer = csv.writer(log_file)
        if is_new_log:
            writer.writerow(LOG_HEADER)
        writer.writerow([when.isoformat(), timesteps, mean_reward])


# Turn one line of the CSV log into a ProgressRow
def _row_from_csv(csv_line: dict) -> ProgressRow:
    return ProgressRow(
        time=datetime.fromisoformat(csv_line["time"]),
        timesteps=int(csv_line["timesteps"]),
        mean_reward=float(csv_line["mean_reward"]),
    )


# Every row in the log, oldest first; an empty list if there is no log yet
def read_progress(run_dir: Path) -> list[ProgressRow]:
    log_path = _log_path(run_dir)
    if not log_path.exists():
        return []
    with log_path.open(newline="") as log_file:
        return [_row_from_csv(csv_line) for csv_line in csv.DictReader(log_file)]


# Minutes from the best score (the earliest one if tied) to the last row; 0.0 with no rows
def minutes_since_improvement(rows: list[ProgressRow]) -> float:
    if not rows:
        return 0.0
    # max() returns the first of several equal maximums, so ties keep the earliest row
    best_row = max(rows, key=lambda row: row.mean_reward)
    elapsed = rows[-1].time - best_row.time
    return elapsed.total_seconds() / 60


# Hours from the first row's time to each row, for the chart's x axis
def _hours_since_start(rows: list[ProgressRow]) -> list[float]:
    if not rows:
        return []
    start_time = rows[0].time
    return [(row.time - start_time).total_seconds() / 3600 for row in rows]


# Draw the score against hours of training and save it as progress.png
def plot_progress(run_dir: Path) -> Path:
    rows = read_progress(run_dir)
    figure, axes = plt.subplots()
    axes.plot(_hours_since_start(rows), [row.mean_reward for row in rows], marker="o")
    axes.set_title("Duck training progress")
    axes.set_xlabel("Hours of training")
    axes.set_ylabel("Average score per attempt")
    chart_path = run_dir / CHART_NAME
    figure.savefig(chart_path)
    plt.close(figure)
    return chart_path


# Read the run folder from the command line, then save the chart and print the stall check
def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the training progress chart and check for a stall.")
    parser.add_argument("run_dir", type=Path, help="the run folder, for example runs/duck_v1")
    run_dir = parser.parse_args().run_dir

    rows = read_progress(run_dir)
    chart_path = plot_progress(run_dir)
    minutes = minutes_since_improvement(rows)
    print(f"Chart saved to {chart_path}")
    print(f"Minutes since improvement: {minutes:.1f}")
    if minutes >= STALL_MINUTES:
        print("No improvement for an hour. Watch the latest snapshot video, then resume with more curiosity, for example --ent-coef 0.01")


if __name__ == "__main__":
    main()
