import json
from pathlib import Path

from .config import RUNS_DIR

PROGRAM_FIELDS = ("id", "parent_id", "generation", "idea", "valid", "score", "image", "model", "direction",
                  "metrics", "llm_cost", "run_seconds", "code")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def run_summary(run_dir: Path) -> dict | None:
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    events = read_jsonl(run_dir / "events.jsonl")
    cost_path = run_dir / "cost.json"
    cost = json.loads(cost_path.read_text())["total"] if cost_path.exists() else 0.0
    bests = [e for e in events if e["type"] == "new_best"]
    candidates = [e for e in events if e["type"] == "candidate"]
    best = bests[-1] if bests else None
    record_event = next((e for e in bests if e.get("beats_record") and e.get("gap_to_record", 0) < -1e-9 * max(1, abs(meta.get("best_known") or 1))), None)
    return {
        "name": run_dir.name,
        "title": meta.get("title"),
        "problem": meta.get("problem"),
        "best_known": meta.get("best_known"),
        "best_known_source": meta.get("best_known_source"),
        "maximize": meta.get("maximize"),
        "worker_model": meta.get("worker_model"),
        "use_images": meta.get("use_images", True),
        "lead_model": meta.get("lead_model"),
        "started": meta.get("started"),
        "programs": len(candidates),
        "cost": cost,
        "best_score": best["score"] if best else None,
        "gap_to_record": best.get("gap_to_record") if best else None,
        "record_program": record_event["id"] if record_event else None,
        "finished": any(e["type"] in ("finished", "budget_stop") for e in events),
    }


def index(runs_dir: Path = RUNS_DIR, names: list[str] | None = None) -> dict:
    runs = []
    for d in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if d.is_dir() and (names is None or d.name in names):
            s = run_summary(d)
            if s:
                runs.append(s)
    return {"runs": runs}


RECORDS_DIR = RUNS_DIR.parent / "records"


def records() -> list[dict]:
    """Verified records with coordinates, for the 3D viewer and the proof section."""
    out = []
    for rec in sorted(RECORDS_DIR.glob("spherical_*")):
        sol = rec / "solution.json"
        csv = next(rec.glob("code_*.csv"), None)
        if not sol.exists() or csv is None:
            continue
        info = json.loads(sol.read_text())
        d, n = rec.name.split("_")[1:]
        pts = [[round(float(v), 9) for v in line.split(",")] for line in csv.read_text().splitlines() if line.strip()]
        out.append({"key": f"{d}_{n}", "d": int(d[:-1]), "n": int(n), "score": info["score"], "best_known": info["best_known"],
                    "csv": f"records/{rec.name}/{csv.name}", "verification": f"records/{rec.name}/verification.txt", "points": pts})
    return out


def model_stats(runs: list[str] | None = None) -> dict:
    spark = wonder = pictures = 0
    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or (runs is not None and d.name not in runs):
            continue
        for e in read_jsonl(d / "events.jsonl"):
            if e["type"] == "candidate" and e.get("parent") is not None:
                spark += 1
                if e.get("picture") and e["picture"].lower() != "none":
                    pictures += 1
            elif e["type"] == "strategy":
                wonder += 1
    return {"spark_programs": spark, "wonder_memos": wonder, "pictures_studied": pictures}


def _outcome(run_dir: Path, require_finished: bool = True) -> dict | None:
    s = run_summary(run_dir)
    if not s or (require_finished and not s["finished"]):
        return None
    first, count = None, 0
    tol = 1e-9 * max(1.0, abs(s["best_known"] or 1))
    for e in read_jsonl(run_dir / "events.jsonl"):
        if e["type"] == "candidate":
            count += 1
        elif e["type"] == "new_best" and first is None and (e.get("gap_to_record") or 0) < -tol:
            first = count
    gap = s["gap_to_record"] or 0.0
    return {"run": run_dir.name, "programs": s["programs"], "gap": gap, "beat": gap < -tol, "first_beat_at": first}


def ablation() -> list[dict]:
    """Same targets with and without the solution pictures."""
    rows = []
    for off_dir in sorted(RUNS_DIR.glob("ablation_*")):
        t = off_dir.name.split("_", 1)[1]
        on_dir = RUNS_DIR / ("sc_4d_168_v1" if t == "4d_168" else f"sweep_{t}")
        on, off = _outcome(on_dir, require_finished=False), _outcome(off_dir)
        if on and off:
            d, n = t.split("_")
            rows.append({"target": t, "label": f"{n} points, {d[:-1]}D", "with_pictures": on, "without_pictures": off})
    return rows


def programs(run_dir: Path) -> list[dict]:
    return [{k: p.get(k) for k in PROGRAM_FIELDS} for p in read_jsonl(run_dir / "programs.jsonl")]
