"""Run the discovery engine over spherical-code targets; snapshot and verify every record found.

Examples:
  python tools/sweep.py --targets 4d_247 4d_150
  python tools/sweep.py --dims 6 7 --max-n 140 --workers 6
  python tools/sweep.py --targets 4d_150 4d_153 --no-images --prefix ablation
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def table_targets(dims, max_n, skip):
    table = json.loads((ROOT / "data" / "spherical_table.json").read_text())
    rows = [e for e in table if e["needs_opt"] and e["d"] in dims and e["n"] <= max_n]
    rows.sort(key=lambda e: (e["n"] * e["d"], e["d"]))
    return [f"{e['d']}d_{e['n']}" for e in rows if f"{e['d']}d_{e['n']}" not in skip]


def save_record(t, run_dir, info, tag):
    d, n = t.split("_")
    rec = ROOT / "records" / f"spherical_{t}"
    old = rec / "solution.json"
    if old.exists() and json.loads(old.read_text())["score"] <= info["score"]:
        return None
    rec.mkdir(parents=True, exist_ok=True)
    for f in (run_dir / "best").iterdir():
        shutil.copy(f, rec / f.name)
    x = np.load(rec / "solution.npy")
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    csv = rec / f"code_{d[:-1]}_{n}.csv"
    np.savetxt(csv, x, delimiter=",", fmt="%.17g")
    out = subprocess.run([PY, str(ROOT / "tools" / "verify_code.py"), str(csv), str(info["best_known"])],
                         capture_output=True, text=True).stdout
    (rec / "verification.txt").write_text(out + f"found by run: {run_dir.name} ({tag})\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=[])
    ap.add_argument("--dims", nargs="*", type=int, default=[])
    ap.add_argument("--max-n", type=int, default=10_000)
    ap.add_argument("--iterations", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--prefix", default="sweep")
    args = ap.parse_args()

    done = {p.name.split("_", 1)[1] for p in (ROOT / "runs").glob(f"{args.prefix}_*") if (p / "best" / "solution.json").exists()}
    targets = [t for t in args.targets if t not in done]
    if args.dims:
        targets += table_targets(set(args.dims), args.max_n, done | set(targets))
    print(f"{len(targets)} targets: {' '.join(targets)}", flush=True)
    tag = "pictures off" if args.no_images else "pictures on"
    for t in targets:
        name = f"{args.prefix}_{t}"
        run_dir = ROOT / "runs" / name
        print(f"=== {t}: starting ({tag})", flush=True)
        cmd = [PY, "-u", "-m", "discovery.run", "--problem", f"spherical_code_{t}", "--iterations", str(args.iterations),
               "--workers", str(args.workers), "--budget", str(args.budget), "--lead-every", "16", "--name", name]
        if args.no_images:
            cmd.append("--no-images")
        with open(ROOT / "runs" / f"{name}.log", "w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        sol = run_dir / "best" / "solution.json"
        if not sol.exists():
            print(f"=== {t}: no solution", flush=True)
            continue
        info = json.loads(sol.read_text())
        gap = info["gap_to_record"]
        cost = json.loads((run_dir / "cost.json").read_text())["total"]
        print(f"=== {t}: best={info['score']:.13f} record={info['best_known']:.12f} gap={gap:+.3e} cost=${cost:.2f}", flush=True)
        if gap < -1e-9 * max(1.0, abs(info["best_known"])):
            out = save_record(t, run_dir, info, tag)
            print(f"=== {t}: RECORD ({tag})" + (f" saved\n{out}" if out else " (an earlier saved record is better)"), flush=True)


if __name__ == "__main__":
    main()
