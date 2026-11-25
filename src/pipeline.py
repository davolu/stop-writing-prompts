"""Run steps 4 to 6 end to end: baseline score, GEPA, test score.

    python -m src.pipeline
"""
import json
import time

import dspy

from . import usage
from .config import OUTPUTS, key_is_set, make_lms
from .data import examples
from .evaluate import run
from .judge import set_judge_lm
from .optimize import export, optimize
from .signatures import make_bot


def main():
    assert key_is_set(), "Set ANTHROPIC_API_KEY first."
    task_lm, judge_lm = make_lms()
    dspy.configure(lm=task_lm)
    set_judge_lm(judge_lm)
    train, val, test = examples("train"), examples("val"), examples("test")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    base = run(make_bot(), test)
    print(f"Baseline test score: {base['mean']:.3f}")
    (OUTPUTS / "baseline_test.json").write_text(json.dumps(base, indent=2))
    print("usage after baseline:"); usage.print_usage(usage.record("2_baseline_test", [task_lm, judge_lm]))

    t0 = time.time()
    compiled = optimize(train, val, reflection_lm=judge_lm)
    print(f"GEPA finished in {time.time() - t0:.0f}s")
    print("usage after GEPA:"); usage.print_usage(usage.record("3_gepa", [task_lm, judge_lm]))
    instructions = export(compiled)
    print("\n--- Optimized system prompt ---\n" + instructions + "\n-------------------------------")

    opt = run(compiled, test)
    print(f"Baseline test score:  {base['mean']:.3f}")
    print(f"Optimized test score: {opt['mean']:.3f}")
    (OUTPUTS / "optimized_test.json").write_text(json.dumps(opt, indent=2))
    usage.print_usage(usage.record("4_full_pipeline", [task_lm, judge_lm]))


if __name__ == "__main__":
    main()
