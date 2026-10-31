import json
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import config, prompts
from .database import Program, ProgramDatabase
from .llm import LLM, BudgetExceeded, CostLedger
from .problems import Problem
from .sandbox import run_candidate


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def emit(self, kind: str, **payload):
        line = json.dumps({"t": time.time(), "type": kind, **payload})
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {kind}: " + ", ".join(f"{k}={v}" for k, v in payload.items() if k not in ("code", "text")))


class DiscoveryEngine:
    def __init__(self, problem: Problem, run_dir: Path, budget_usd: float, workers: int = 4,
                 worker_model: str = config.WORKER_MODEL, lead_model: str | None = config.LEAD_MODEL,
                 lead_every: int = 20, worker_effort: str = "medium", use_images: bool = True, seed: int = 0):
        self.problem = problem
        self.run_dir = run_dir
        (run_dir / "images").mkdir(parents=True, exist_ok=True)
        (run_dir / "best").mkdir(exist_ok=True)
        self.db = ProgramDatabase(run_dir / "programs.jsonl", maximize=problem.maximize)
        self.ledger = CostLedger(run_dir / "cost.json", budget_usd)
        self.llm = LLM(self.ledger)
        self.events = EventLog(run_dir / "events.jsonl")
        self.workers = workers
        self.worker_model = worker_model
        self.lead_model = lead_model
        self.lead_every = lead_every
        self.worker_effort = worker_effort
        self.use_images = use_images
        self.rng = random.Random(seed)
        self.strategy = ""
        self.directions: list[str] = []
        self.best_history: list[str] = []
        self._strategy_lock = threading.Lock()
        (run_dir / "run.json").write_text(json.dumps({
            "problem": problem.name, "title": problem.title, "best_known": problem.best_known,
            "best_known_source": problem.best_known_source, "maximize": problem.maximize,
            "worker_model": worker_model, "lead_model": lead_model, "budget": budget_usd,
            "workers": workers, "use_images": use_images, "started": time.time(),
        }, indent=2))

    def _evaluate_code(self, code: str):
        run = run_candidate(code, self.problem.timeout_seconds)
        if not run.ok:
            return run, None
        return run, self.problem.evaluate(run.output)

    def _record(self, parent: Program | None, idea: str, code: str, model: str, cost: float, direction: str = "",
                picture_note: str = "") -> Program:
        run, ev = self._evaluate_code(code)
        pid = self.db.reserve_id()
        image_rel = ""
        if ev is not None:
            image_rel = f"images/{pid:05d}.png"
            self.problem.render(run.output, ev, self.run_dir / image_rel)
        prog = Program(
            id=pid, parent_id=parent.id if parent else None, generation=(parent.generation + 1) if parent else 0,
            idea=idea, code=code, valid=bool(ev and ev.valid), score=ev.score if ev else 0.0,
            message=(ev.message if ev else run.error), metrics=(ev.metrics if ev else {}), image=image_rel,
            model=model, llm_cost=cost, run_seconds=run.seconds, direction=direction,
        )
        prev_best = self.db.best()
        self.db.add(prog)
        self.events.emit("candidate", id=prog.id, parent=prog.parent_id, valid=prog.valid, score=round(prog.score, 10),
                         idea=idea[:160], picture=picture_note[:200], message=prog.message[:160], cost_total=round(self.ledger.total, 4))
        if prog.valid and (prev_best is None or self.problem.better(prog.score, prev_best.score)):
            self._new_best(prog, run.output)
        return prog

    def _new_best(self, prog: Program, output: np.ndarray):
        gap = self.problem.gap_to_record(prog.score)
        self.best_history.append(f"after {len(self.db.programs)} programs: {prog.score:.10f}")
        best_dir = self.run_dir / "best"
        (best_dir / "program.py").write_text(prog.code, encoding="utf-8")
        np.save(best_dir / "solution.npy", output)
        (best_dir / "solution.json").write_text(json.dumps({
            "program_id": prog.id, "score": prog.score, "metrics": prog.metrics, "solution": output.tolist(),
            "best_known": self.problem.best_known, "gap_to_record": gap,
        }, indent=2))
        if prog.image:
            (best_dir / "image.png").write_bytes((self.run_dir / prog.image).read_bytes())
        beats = gap is not None and gap < 0
        self.events.emit("new_best", id=prog.id, score=round(prog.score, 10),
                         gap_to_record=None if gap is None else float(f"{gap:.3e}"), beats_record=beats, idea=prog.idea[:160])

    def _attempt(self, it: int) -> Program | None:
        parent = self.db.select_parent(self.rng)
        insp = self.db.inspirations(parent.id)
        fails = self.db.recent_failures()
        best = self.db.best()
        with self._strategy_lock:
            strategy = self.strategy
            direction = self.rng.choice(self.directions) if self.directions and self.rng.random() < 0.7 else ""
        img = (self.run_dir / parent.image) if (self.use_images and parent.image) else None
        text = prompts.worker_prompt(self.problem, parent, insp, fails, best.score if best else None, strategy, direction, img is not None)
        res = self.llm.ask(self.worker_model, text, images=[img] if img else None, effort=self.worker_effort,
                           instructions=prompts.WORKER_INSTRUCTIONS)
        idea, picture, code = prompts.parse_worker(res.text)
        if not code:
            self.events.emit("no_code", parent=parent.id, cost_total=round(self.ledger.total, 4))
            return None
        return self._record(parent, idea or f"mutation {it}", code, res.model, res.cost, direction, picture)

    def _lead(self):
        best = self.db.best()
        top = self.db.ranked()[:4]
        if not top or not self.lead_model:
            return
        img = (self.run_dir / best.image) if best and best.image else None
        text = prompts.lead_prompt(self.problem, top, self.db.recent_failures(8), len(self.db.programs), self.best_history)
        res = self.llm.ask(self.lead_model, text, images=[img] if img else None, effort="high", instructions=prompts.LEAD_INSTRUCTIONS)
        diagnosis, directions = prompts.parse_directions(res.text)
        if directions:
            with self._strategy_lock:
                self.strategy = diagnosis
                self.directions = directions
            (self.run_dir / "strategy.md").write_text(res.text, encoding="utf-8")
            self.events.emit("strategy", model=res.model, diagnosis=diagnosis[:400], directions=len(directions),
                             cost=round(res.cost, 4), cost_total=round(self.ledger.total, 4))

    def run(self, iterations: int):
        if not self.db.programs:
            self._record(None, "seed program", self.problem.seed_code(), "seed", 0.0)
        completed = 0
        submitted = 0
        since_lead = 0
        lead_future = None
        stop = False
        with ThreadPoolExecutor(max_workers=self.workers) as pool, ThreadPoolExecutor(max_workers=1) as lead_pool:
            futures = set()
            while (completed < iterations and not stop) or futures:
                while not stop and submitted < iterations and len(futures) < self.workers:
                    futures.add(pool.submit(self._attempt, submitted))
                    submitted += 1
                done, futures = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for f in done:
                    completed += 1
                    since_lead += 1
                    try:
                        f.result()
                    except BudgetExceeded as e:
                        self.events.emit("budget_stop", message=str(e))
                        stop = True
                    except Exception as e:
                        self.events.emit("error", message=f"{type(e).__name__}: {str(e)[:300]}")
                if (self.lead_model and since_lead >= self.lead_every and not stop
                        and (lead_future is None or lead_future.done())):
                    since_lead = 0
                    lead_future = lead_pool.submit(self._lead)
        best = self.db.best()
        self.events.emit("finished", programs=len(self.db.programs), best=round(best.score, 10) if best else None,
                         cost_total=round(self.ledger.total, 4))
        return best
