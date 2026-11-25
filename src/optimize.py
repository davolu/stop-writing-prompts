"""Step 5: let GEPA write the system prompt."""
import json

import dspy

from .config import OUTPUTS, PROMPTS
from .judge import judge_metric
from .signatures import BASELINE, make_bot


def optimize(trainset, valset, reflection_lm, auto="light", seed=0, threads=6):
    gepa = dspy.GEPA(
        metric=judge_metric,
        reflection_lm=reflection_lm,   # stronger model writes new prompts
        auto=auto,                     # budget preset: light / medium / heavy
        track_stats=True,
        seed=seed,
        num_threads=threads,
        log_dir=str(OUTPUTS / "gepa_logs"),
    )
    student = make_bot(BASELINE)
    compiled = gepa.compile(student, trainset=trainset, valset=valset)
    compiled.save(str(OUTPUTS / "optimized_program.json"))
    return compiled


def export(compiled) -> str:
    instructions = compiled.signature.instructions
    PROMPTS.mkdir(exist_ok=True)
    (PROMPTS / "baseline.txt").write_text(BASELINE + "\n")
    (PROMPTS / "optimized.txt").write_text(instructions.strip() + "\n")
    r = compiled.detailed_results
    (OUTPUTS / "gepa_summary.json").write_text(json.dumps(dict(
        candidates=len(r.candidates), best_idx=r.best_idx,
        val_aggregate_scores=[round(s, 3) for s in r.val_aggregate_scores],
        parents=r.parents, discovery_eval_counts=r.discovery_eval_counts,
        total_metric_calls=getattr(r, "total_metric_calls", None)), indent=2, default=str))
    return instructions
