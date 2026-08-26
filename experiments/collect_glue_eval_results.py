#!/usr/bin/env python3
"""Collect GLUE eval_results.json files beneath one run root."""

import json
from pathlib import Path
import sys


root = Path(sys.argv[1])
records = []
for path in sorted(root.glob("**/eval_results.json")):
    run_name = path.parent.name
    if "_glue_" not in run_name or "ftseed" not in run_name:
        continue
    try:
        metrics = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    states = []
    for state_path in path.parent.glob("checkpoint-*/trainer_state.json"):
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            continue
        states.append((int(state.get("global_step", 0)), state_path, state))
    states.sort(key=lambda item: item[0])
    latest_state = states[-1][2] if states else {}
    records.append(
        {
            "run_name": run_name,
            "path": str(path),
            "metrics": metrics,
            "log_history": latest_state.get("log_history", []),
        }
    )

json.dump(records, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
