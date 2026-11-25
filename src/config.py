"""Models, paths and official list prices.

Prices: USD per million tokens, standard Global, from
https://platform.claude.com/docs/en/about-claude/pricing (checked 25 Nov 2026).
Re-check before you run; prices change.
"""
import os
from pathlib import Path

import dspy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

TASK_MODEL = "claude-haiku-5-5"    # the support bot: many cheap calls
JUDGE_MODEL = "claude-sonnet-5-5"  # judge + GEPA reflection: fewer, smarter calls

# (input, output) USD per 1M tokens. Haiku 5.5 row is for prompts up to 100K tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}


def key_is_set() -> bool:
    """True if ANTHROPIC_API_KEY is present. Never prints the key."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def make_lms():
    task_lm = dspy.LM(f"anthropic/{TASK_MODEL}", max_tokens=1000)
    judge_lm = dspy.LM(f"anthropic/{JUDGE_MODEL}", max_tokens=8000)
    return task_lm, judge_lm


def read(name: str) -> str:
    return (DATA / name).read_text().strip()
