"""Run the discovery engine over a list of spherical-code targets; snapshot and verify every record found."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

TARGETS = ["4d_247", "4d_150", "4d_216", "4d_153", "5d_159", "4d_163", "4d_250", "4d_272", "4d_292", "5d_173"]


def main():
    iterations = int(os.environ.get("SWEEP_ITERATIONS", "64"))
    budget = os.environ.get("SWEEP_BUDGET", "1.5")
    targets = sys.argv[1:] or TARGETS
    for t in targets:
        name = f"sweep_{t}"
        run_dir = ROOT / "runs" / name
        print(f"=== {t}: starting", flush=True)
        with open(ROOT / "runs" / f"{name}.log", "w", encoding="utf-8") as log:
            subprocess.run([PY, "-u", "-m", "discovery.run", "--problem", f"spherical_code_{t}", "--iterations", str(iterations),
                            "--workers", "8", "--budget", budget, "--lead-every", "16", "--name", name],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        sol = run_dir / "best" / "solution.json"
        if not sol.exists():
            print(f"=== {t}: no solution", flush=True)
            continue
        info = json.loads(sol.read_text())
        gap = info["gap_to_record"]
        cost = json.loads((run_dir / "cost.json").read_text())["total"]
        print(f"=== {t}: best={info['score']:.13f} record={info['best_known']:.12f} gap={gap:+.3e} cost=${cost:.2f}", flush=True)
        if gap < -1e-9:
            d, n = t.split("_")
            rec = ROOT / "records" / f"spherical_{t}"
            rec.mkdir(parents=True, exist_ok=True)
            for f in (run_dir / "best").iterdir():
                shutil.copy(f, rec / f.name)
            x = np.load(rec / "solution.npy")
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            csv = rec / f"code_{d[:-1]}_{n}.csv"
            np.savetxt(csv, x, delimiter=",", fmt="%.17g")
            out = subprocess.run([PY, str(ROOT / "tools" / "verify_code.py"), str(csv), str(info["best_known"])],
                                 capture_output=True, text=True).stdout
            (rec / "verification.txt").write_text(out)
            print(f"=== {t}: RECORD saved\n{out}", flush=True)


if __name__ == "__main__":
    main()
