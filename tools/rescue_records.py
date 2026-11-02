"""Save records from runs that beat the table but were interrupted before the sweep could save them.

  python tools/rescue_records.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import ROOT, save_record  # noqa: E402


def main():
    for run_dir in sorted((ROOT / "runs").glob("*_*d_*")):
        sol = run_dir / "best" / "solution.json"
        if not run_dir.is_dir() or not sol.exists():
            continue
        info = json.loads(sol.read_text())
        if info.get("gap_to_record") is None or info["gap_to_record"] >= -1e-9 * max(1.0, abs(info["best_known"])):
            continue
        m = re.search(r"(\d+d_\d+)$", run_dir.name)
        if not m:
            continue
        t = m.group(1)
        run_json = run_dir / "run.json"
        images = json.loads(run_json.read_text()).get("use_images", True) if run_json.exists() else True
        out = save_record(t, run_dir, info, "pictures on" if images else "pictures off")
        if out:
            print(f"=== {t}: saved from {run_dir.name}\n{out}", flush=True)


if __name__ == "__main__":
    main()
