import base64
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from . import config


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost: float
    seconds: float


class CostLedger:
    """Thread-safe running total of API spend, persisted so a run can be resumed."""

    def __init__(self, path: Path, budget_usd: float):
        self.path = path
        self.budget = budget_usd
        self._lock = threading.Lock()
        self.total = 0.0
        self.by_model: dict[str, dict] = {}
        if path.exists():
            data = json.loads(path.read_text())
            self.total = data.get("total", 0.0)
            self.by_model = data.get("by_model", {})

    def check(self):
        if self.total >= self.budget:
            raise BudgetExceeded(f"Budget ${self.budget:.2f} reached (spent ${self.total:.4f})")

    def add(self, r: LLMResult):
        with self._lock:
            self.total += r.cost
            m = self.by_model.setdefault(r.model, {"calls": 0, "input": 0, "cached": 0, "output": 0, "cost": 0.0})
            m["calls"] += 1
            m["input"] += r.input_tokens
            m["cached"] += r.cached_tokens
            m["output"] += r.output_tokens
            m["cost"] += r.cost
            self.path.write_text(json.dumps({"total": self.total, "budget": self.budget, "by_model": self.by_model}, indent=2))


def cost_of(model: str, input_tokens: int, cached: int, output_tokens: int) -> float:
    p_in, p_cached, p_out = config.PRICES.get(model, config.PRICES["muse-spark-1.3"])
    return ((input_tokens - cached) * p_in + cached * p_cached + output_tokens * p_out) / 1_000_000


def image_part(png_path: Path) -> dict:
    b64 = base64.b64encode(png_path.read_bytes()).decode()
    return {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}


class LLM:
    def __init__(self, ledger: CostLedger):
        self.client = OpenAI(base_url=config.API_BASE_URL, api_key=config.api_key(), timeout=600)
        self.ledger = ledger

    def ask(self, model: str, text: str, images: list[Path] | None = None, effort: str = "medium",
            instructions: str | None = None, retries: int = 4) -> LLMResult:
        self.ledger.check()
        content = [{"type": "input_text", "text": text}]
        for img in images or []:
            if img and Path(img).exists():
                content.append(image_part(Path(img)))
        kwargs = {"model": model, "input": [{"role": "user", "content": content}], "reasoning": {"effort": effort}}
        if instructions:
            kwargs["instructions"] = instructions

        delay = 5.0
        for attempt in range(retries + 1):
            start = time.time()
            try:
                resp = self.client.responses.create(**kwargs)
                break
            except Exception as e:
                status = getattr(e, "status_code", None)
                if attempt == retries or status in (400, 401, 403, 404):
                    raise
                time.sleep(delay)
                delay *= 2

        u = resp.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        result = LLMResult(
            text=resp.output_text or "",
            model=model,
            input_tokens=u.input_tokens,
            cached_tokens=cached,
            output_tokens=u.output_tokens,
            cost=cost_of(model, u.input_tokens, cached, u.output_tokens),
            seconds=time.time() - start,
        )
        self.ledger.add(result)
        return result
