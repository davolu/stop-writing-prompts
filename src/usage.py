"""Real token usage and cost, read from each API response's usage fields.

DSPy keeps a history per LM. Calls answered from DSPy's local cache have an
empty usage dict and cost nothing, so we only count entries with usage.
"""
import json

from .config import OUTPUTS, PRICES


def summarize(lms, since_last=False) -> dict:
    """Sum usage per model. since_last=True counts only calls made since the last record()."""
    out = {}
    for lm in lms:
        model = lm.model.split("/", 1)[1]
        calls = inp = outp = reasoning = cached = 0
        start = getattr(lm, "_usage_mark", 0) if since_last else 0
        for h in lm.history[start:]:
            u = h.get("usage") or {}
            if not u:
                continue
            calls += 1
            inp += u.get("prompt_tokens", 0)
            outp += u.get("completion_tokens", 0)
            reasoning += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            cached += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        pin, pout = PRICES[model]
        row = out.setdefault(model, dict(api_calls=0, input_tokens=0, output_tokens=0,
                                         reasoning_tokens=0, cached_input_tokens=0))
        row["api_calls"] += calls; row["input_tokens"] += inp; row["output_tokens"] += outp
        row["reasoning_tokens"] += reasoning; row["cached_input_tokens"] += cached
    for model, row in out.items():
        pin, pout = PRICES[model]
        row["cost_usd"] = round(row["input_tokens"] / 1e6 * pin + row["output_tokens"] / 1e6 * pout, 4)
    return out


def record(step: str, lms) -> dict:
    """Save usage for the calls made since the previous record() (one step) to outputs/usage.json and return it."""
    path = OUTPUTS / "usage.json"
    all_steps = json.loads(path.read_text()) if path.exists() else {}
    all_steps[step] = summarize(lms, since_last=True)
    for lm in lms:
        lm._usage_mark = len(lm.history)
    path.write_text(json.dumps(all_steps, indent=2))
    return all_steps[step]


def total() -> dict:
    path = OUTPUTS / "usage.json"
    steps = json.loads(path.read_text()) if path.exists() else {}
    t = {}
    for step, models in steps.items():
        for m, row in models.items():
            r = t.setdefault(m, dict(api_calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0))
            for k in ("api_calls", "input_tokens", "output_tokens"):
                r[k] += row[k]
            r["cost_usd"] = round(r["cost_usd"] + row["cost_usd"], 4)
    return t


def print_usage(u: dict):
    for m, r in u.items():
        print(f"{m:18} calls={r['api_calls']:4}  in={r['input_tokens']:>8,}  out={r['output_tokens']:>7,}  ${r['cost_usd']:.4f}")
