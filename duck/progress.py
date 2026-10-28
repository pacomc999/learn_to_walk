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
LOG_HEADER = ["time", "timesteps", "mean_reward", "mean_distance"]

# Printed when the log cannot be written, for example while it is open in Excel
LOCKED_LOG_WARNING = "Could not write progress.csv, is it open in another program? Training continues."


# One line of the progress log: when it was written, the step count,
# the average score and the average forward distance per attempt (metres)
@dataclass(frozen=True)
class ProgressRow:
    time: datetime
    timesteps: int
    mean_reward: float
    mean_distance: float


# The full path of the progress log in a run folder
def _log_path(run_dir: Path) -> Path:
    return run_dir / LOG_NAME


# Add one row to the log, writing the header first if the log is new.
# A locked file (open in another program) only prints a warning, so training goes on
def append_progress(run_dir: Path, timesteps: int, mean_reward: float,
                    mean_distance: float, when: datetime) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(run_dir)
    is_new_log = not log_path.exists()
    try:
        with log_path.open("a", newline="") as log_file:
            writer = csv.writer(log_file)
            if is_new_log:
                writer.writerow(LOG_HEADER)
            writer.writerow([when.isoformat(), timesteps, mean_reward, mean_distance])
    except OSError:  # PermissionError is one kind of OSError
        print(LOCKED_LOG_WARNING, flush=True)


# Turn one line of the CSV log into a ProgressRow
def _row_from_csv(csv_line: dict) -> ProgressRow:
    return ProgressRow(
        time=datetime.fromisoformat(csv_line["time"]),
        timesteps=int(csv_line["timesteps"]),
        mean_reward=float(csv_line["mean_reward"]),
        mean_distance=float(csv_line["mean_distance"]),
    )


# Turn one line into a ProgressRow, or None if the line is broken (for example half written)
def _row_or_none(csv_line: dict) -> ProgressRow | None:
    try:
        return _row_from_csv(csv_line)
    except (KeyError, TypeError, ValueError):
        return None


# Every good row in the log, oldest first; an empty list if there is no log yet
def read_progress(run_dir: Path) -> list[ProgressRow]:
    log_path = _log_path(run_dir)
    if not log_path.exists():
        return []
    with log_path.open(newline="") as log_file:
        all_rows = [_row_or_none(csv_line) for csv_line in csv.DictReader(log_file)]
    return [row for row in all_rows if row is not None]


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


# Draw the score (top panel) and the distance (bottom panel) against hours of training,
# and save them as progress.png
def plot_progress(run_dir: Path) -> Path:
    rows = read_progress(run_dir)
    hours = _hours_since_start(rows)
    figure, (score_axes, distance_axes) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    score_axes.plot(hours, [row.mean_reward for row in rows], marker="o")
    score_axes.set_title("Duck training progress")
    score_axes.set_ylabel("Average score per attempt")
    distance_axes.plot(hours, [row.mean_distance for row in rows], marker="o", color="tab:green")
    distance_axes.set_ylabel("Average distance per attempt (metres)")
    distance_axes.set_xlabel("Hours of training")
    figure.tight_layout()
    chart_path = run_dir / CHART_NAME
    figure.savefig(chart_path)
    plt.close(figure)
    return chart_path


# The sentence that tells how far the duck moved per attempt in the newest row
def latest_distance_message(rows: list[ProgressRow]) -> str:
    if not rows:
        return "No progress logged yet."
    return f"Latest average distance per attempt: {rows[-1].mean_distance:.2f} metres"


# Read the run folder from the command line, then save the chart and print the stall check
def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the training progress chart and check for a stall.")
    parser.add_argument("run_dir", type=Path, help="the run folder, for example runs/duck_v1")
    run_dir = parser.parse_args().run_dir

    rows = read_progress(run_dir)
    chart_path = plot_progress(run_dir)
    minutes = minutes_since_improvement(rows)
    print(f"Chart saved to {chart_path}")
    print(latest_distance_message(rows))
    print(f"Minutes since improvement: {minutes:.1f}")
    if minutes >= STALL_MINUTES:
        print("No improvement for an hour. Watch the latest snapshot video, then resume with more curiosity, for example --ent-coef 0.01")


if __name__ == "__main__":
    main()
