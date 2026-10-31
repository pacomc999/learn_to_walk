# Phone progress page

A page for following training on the phone: https://claude.ai/artifact/TPM1tzECd8biqf2M5ytR3g (private, Francisco's claude.ai account). It shows the duck being trained, a progress bar, the latest and best exam, a film of the best brain so far, a chart of every exam and a table, from one shared document (`training/current`) in the page's database.

The page does not read the laptop by itself. During training Claude refreshes it after every exam:

1. `.venv/Scripts/python docs/phone_page/phone_status.py RUN_FOLDER "DUCK NAME" "ONE LINE SUMMARY" TOTAL_STEPS [training|finished] [COMPARE_NAME COMPARE_RUN]` reads the run's `exam.csv` and `progress.csv` and writes `phone_status.json` next to the script. Use `none` as COMPARE_NAME for no comparison line on the chart.
2. Claude writes that file to the page's database with its ArtifactData tool (`set` on collection `training`, document `current`, pinned to the version it last saw).
3. The film: when the run's `dashboard_latest_steps.txt` changes, Claude uploads `dashboard_latest.mp4` to the page (Artifact tool, `asset: true`; the page declares the `assets` capability), notes its `url`, `id` and `steps` in `phone_video.json` next to the script (not in git), deletes the previous upload, and runs step 1 again: the script copies `phone_video.json` into the document as `video`, and the page shows it under the three numbers.

While a run trains, Claude can repeat these steps every 5 minutes with a session job (CronCreate); the job ends when Claude Code closes.

`duck_training.html` is the page's source, kept here in case it needs changing; publish it again to the same artifact URL.
