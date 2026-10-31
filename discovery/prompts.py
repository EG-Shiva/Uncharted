import re

from .database import Program
from .problems import Problem

WORKER_INSTRUCTIONS = """You are a world-class research mathematician and optimisation engineer working inside an automated discovery lab.
Your job is to write a complete Python program that beats the current best solution to an open problem.
Every program you write is executed and checked by an exact verifier, so correctness matters more than cleverness.
Be bold: try genuinely different constructions, better optimisers, smarter initialisations, and careful final polishing."""

LEAD_INSTRUCTIONS = """You are the principal investigator of an automated mathematical discovery lab.
Worker models write programs; a verifier scores them. You read the evidence and set research strategy."""


def _clip(code: str, limit: int) -> str:
    return code if len(code) <= limit else code[:limit] + "\n# ... (truncated)"


def worker_prompt(problem: Problem, parent: Program, inspirations: list[Program], failures: list[Program],
                  best_score: float | None, strategy: str, direction: str, has_image: bool) -> str:
    record = f"{problem.best_known:.10f} ({problem.best_known_source})" if problem.best_known is not None else "unknown"
    parts = [
        f"# Problem\n{problem.description()}",
        f"# Scoreboard\nPublic best-known value: {record}\nOur lab's best so far: {best_score if best_score is not None else 'none yet'}",
    ]
    if strategy:
        parts.append(f"# Research strategy from the principal investigator\n{strategy}")
    if direction:
        parts.append(f"# Your assigned direction for this attempt\n{direction}")

    parent_status = f"score {parent.score:.10f}" if parent.valid else f"INVALID: {parent.message}"
    parts.append(f"# Parent program (id {parent.id}, {parent_status})\nIdea: {parent.idea or 'seed'}\n```python\n{_clip(parent.code, 12000)}\n```")
    if has_image:
        parts.append("# Picture of the parent's solution\nThe attached image is the parent's solution drawn to scale. "
                     "Study it before coding: where is space wasted, which elements are cramped or could grow, what symmetry or structure is missing?")

    for p in inspirations:
        parts.append(f"# Another strong program (id {p.id}, score {p.score:.10f})\nIdea: {p.idea}\n```python\n{_clip(p.code, 5000)}\n```")
    if failures:
        lines = "\n".join(f"- idea: {f.idea[:150]} -> {f.message[:200]}" for f in failures)
        parts.append(f"# Recent failed attempts (avoid repeating these mistakes)\n{lines}")

    parts.append("""# Your response format
IDEA: <one sentence describing the new approach>
PICTURE: <one or two sentences on what you noticed in the image, or 'none'>
Then exactly one ```python code block containing the COMPLETE program with a `solve()` function.
The program must be self-contained, deterministic (fix random seeds), and finish within the time limit.
Make sure the returned solution satisfies every constraint strictly; shrink by a tiny margin if needed.""")
    return "\n\n".join(parts)


def lead_prompt(problem: Problem, top: list[Program], failures: list[Program], total: int, best_history: list[str]) -> str:
    record = f"{problem.best_known:.10f}" if problem.best_known is not None else "unknown"
    parts = [
        f"# Problem\n{problem.description()}",
        f"# Status\nPublic best-known: {record}. Programs evaluated so far: {total}.\nProgress of our best score over time:\n" + "\n".join(best_history[-15:]),
    ]
    for p in top:
        parts.append(f"## Top program id {p.id}: score {p.score:.10f}\nIdea: {p.idea}\n```python\n{_clip(p.code, 6000)}\n```")
    if failures:
        parts.append("## Recent failures\n" + "\n".join(f"- {f.idea[:150]} -> {f.message[:150]}" for f in failures))
    parts.append("""The attached image shows our current best solution.
Diagnose why progress is or isn't happening and what structural property the best solution is missing.
Then propose exactly 4 distinct, concrete research directions for the worker models.
Format:
DIAGNOSIS: <3-5 sentences>
DIRECTION 1: <one paragraph, specific enough to implement>
DIRECTION 2: ...
DIRECTION 3: ...
DIRECTION 4: ...""")
    return "\n\n".join(parts)


def parse_worker(text: str) -> tuple[str, str, str | None]:
    idea = re.search(r"IDEA:\s*(.+)", text)
    picture = re.search(r"PICTURE:\s*(.+)", text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    code = max(blocks, key=len) if blocks else None
    return (idea.group(1).strip() if idea else "", picture.group(1).strip() if picture else "", code)


def parse_directions(text: str) -> tuple[str, list[str]]:
    diag = re.search(r"DIAGNOSIS:\s*(.+?)(?=DIRECTION 1:|$)", text, flags=re.S)
    dirs = re.findall(r"DIRECTION \d+:\s*(.+?)(?=DIRECTION \d+:|$)", text, flags=re.S)
    return (diag.group(1).strip() if diag else ""), [d.strip() for d in dirs if d.strip()]
