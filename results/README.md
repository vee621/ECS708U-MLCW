# Experiment register

Running `python scripts/smoke_run.py` writes `results/cpu-smoke.json` with
hardware information, partition sizes, cache hit/miss state, actual measured
extraction/training/inference times (in seconds), and **validation-only**
accuracy. Do not fill in timings before a real run. Save a distinct JSON
report for each future experiment ID and report the actual hardware used.
No test metrics belong here until the final evaluation.
