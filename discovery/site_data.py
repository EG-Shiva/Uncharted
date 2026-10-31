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


def programs(run_dir: Path) -> list[dict]:
    return [{k: p.get(k) for k in PROGRAM_FIELDS} for p in read_jsonl(run_dir / "programs.jsonl")]
