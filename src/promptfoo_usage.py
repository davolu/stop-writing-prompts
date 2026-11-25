"""Add Promptfoo's token usage (from the API usage fields it records) to outputs/usage.json."""
import json

from .config import OUTPUTS, PRICES

TASK, GRADER = "claude-haiku-5-5", "claude-sonnet-5-5"


def record(results_path=OUTPUTS / "promptfoo_results.json", step="5_promptfoo"):
    tu = json.loads(results_path.read_text())["results"]["stats"]["tokenUsage"]
    rows = {}
    for model, src in ((TASK, tu), (GRADER, tu["assertions"])):
        pin, pout = PRICES[model]
        rows[model] = dict(api_calls=src["numRequests"], input_tokens=src["prompt"], output_tokens=src["completion"],
                           reasoning_tokens=src.get("completionDetails", {}).get("reasoning", 0),
                           cached_input_tokens=src.get("cached", 0),
                           cost_usd=round(src["prompt"] / 1e6 * pin + src["completion"] / 1e6 * pout, 4))
    path = OUTPUTS / "usage.json"
    steps = json.loads(path.read_text()) if path.exists() else {}
    steps[step] = rows
    path.write_text(json.dumps(steps, indent=2))
    return rows


if __name__ == "__main__":
    print(json.dumps(record(), indent=2))
