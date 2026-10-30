# Phone progress page

A page for following training on the phone: https://claude.ai/artifact/TPM1tzECd8biqf2M5ytR3g (private, Francisco's claude.ai account). It shows the duck being trained, a progress bar, the latest and best exam, a chart of every exam and a table, from one shared document (`training/current`) in the page's database.

The page does not read the laptop by itself. During training Claude refreshes it after every exam:

1. `.venv/Scripts/python docs/phone_page/phone_status.py RUN_FOLDER "DUCK NAME" "ONE LINE SUMMARY" TOTAL_STEPS [training|finished] [COMPARE_NAME COMPARE_RUN]` reads the run's `exam.csv` and `progress.csv` and writes `phone_status.json` next to the script. Use `none` as COMPARE_NAME for no comparison line on the chart.
2. Claude writes that file to the page's database with its ArtifactData tool (`set` on collection `training`, document `current`, pinned to the version it last saw).

`duck_training.html` is the page's source, kept here in case it needs changing; publish it again to the same artifact URL.
