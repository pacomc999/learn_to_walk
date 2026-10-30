# Builds the phone page's status document from a run's exam log and progress log
import csv, json, sys
from datetime import datetime, timedelta
from pathlib import Path

run, duck, summary, total = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
status = sys.argv[5] if len(sys.argv) > 5 else "training"
# The duck to compare with: name and run folder (optional 6th and 7th arguments)
compare_name = sys.argv[6] if len(sys.argv) > 6 else "Duck 2"
compare_run = Path(sys.argv[7]) if len(sys.argv) > 7 else Path("runs/duck_02/from_zero")
compare = [] if compare_name == "none" else [
    {"steps": int(r["timesteps"]), "score": round(float(r["score"]), 1), "falls": int(r["falls"])}
    for r in csv.DictReader(open(compare_run / "exam.csv"))]
if compare_name == "none":
    compare_name = ""
# Each exam, with its attempt count and its two part scores when the log has them:
# lower and taller stones (stone exam, from Duck 3.1's second run), or flat and rough (older exams)
def part(row, key): return round(float(row[key]), 1) if row.get(key) not in (None, "") else None
rows = list(csv.DictReader(open(run / "exam.csv")))
part_names = ["lower", "taller"] if any(part(r, "lower_score") is not None for r in rows) else ["flat", "rough"]
exams = [{"steps": int(r["timesteps"]), "score": round(float(r["score"]), 1), "falls": int(r["falls"]),
          "attempts": int(r.get("attempts") or 12), "parts": [part(r, f"{name}_score") for name in part_names]}
         for r in rows]
progress = list(csv.DictReader(open(run / "progress.csv")))
last, earlier = progress[-1], progress[max(0, len(progress) - 30)]
steps = int(last["timesteps"])
t1, t0 = datetime.fromisoformat(last["time"]), datetime.fromisoformat(earlier["time"])
speed = (steps - int(earlier["timesteps"])) / max((t1 - t0).total_seconds(), 1)
eta = (t1 + timedelta(seconds=(total - steps) / max(speed, 1))).strftime("%H:%M") if status == "training" else ""
doc = {"duck": duck, "summary": summary, "status": status, "steps": steps if status == "training" else max(steps, exams[-1]["steps"]),
       "total": total, "eta": eta, "updated": datetime.now().strftime("%d %b %H:%M"), "exams": exams,
       "part_names": [name.capitalize() for name in part_names],
       "compare": {"name": compare_name, "exams": compare}}
out = Path(sys.argv[0]).parent / "phone_status.json"  # next to this script
out.write_text(json.dumps(doc))
print(f"{steps:,} steps, {len(exams)} exams, eta {eta}")
