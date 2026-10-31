import argparse
import time

from . import config
from .engine import DiscoveryEngine
from .problems import get_problem


def main():
    ap = argparse.ArgumentParser(description="Run the discovery engine on an open problem.")
    ap.add_argument("--problem", default="circle_packing_26")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=2.0, help="Hard spending cap in USD for this run")
    ap.add_argument("--worker-model", default=config.WORKER_MODEL)
    ap.add_argument("--lead-model", default=config.LEAD_MODEL, help="Use 'none' to disable the lead researcher")
    ap.add_argument("--lead-every", type=int, default=20)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--no-images", action="store_true", help="Ablation: workers do not see pictures of solutions")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    problem = get_problem(args.problem)
    run_name = args.name or f"{problem.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = config.RUNS_DIR / run_name
    engine = DiscoveryEngine(
        problem, run_dir, budget_usd=args.budget, workers=args.workers, worker_model=args.worker_model,
        lead_model=None if args.lead_model == "none" else args.lead_model, lead_every=args.lead_every,
        worker_effort=args.effort, use_images=not args.no_images,
    )
    print(f"Run directory: {run_dir}")
    best = engine.run(args.iterations)
    if best:
        gap = problem.gap_to_record(best.score)
        print(f"\nBest score: {best.score:.10f} (program {best.id})")
        if gap is not None:
            print(f"Gap to public record: {gap:+.3e}  ->  {'NEW RECORD' if gap < 0 else 'not yet'}")
    print(f"Spent: ${engine.ledger.total:.4f}")


if __name__ == "__main__":
    main()
