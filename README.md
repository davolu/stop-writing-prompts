# Stop Writing Prompts: Let Your Best Chats Write Them for You

Companion code for the Agentic AI Hub video of the same name.

You don't write the system prompt. You write a short **rubric**, collect **example chats you'd actually send**,
and let an optimizer write and test the prompt. This repo does that with
[DSPy](https://dspy.ai)'s **GEPA** optimizer, an **LLM judge**, and a
[Promptfoo](https://www.promptfoo.dev) regression test.

The demo is a support bot for **Maple & Co.**, a fictional home-goods shop. All tickets and replies are synthetic.

## The pipeline

```
rubric  ->  60 gold chats  ->  train / val / test  ->  GEPA + LLM judge  ->  Promptfoo: baseline vs optimized
```

1. **Rubric** (`data/rubric.md`): empathetic, friendly, straight to the point (under 90 words), faithful to the policy (`data/policy.md`).
2. **Gold chats** (`data/gold_chats.json`): 60 tickets. A model drafted the tickets and replies
   (`src/generate_scenarios.py`, `src/draft_gold.py`, notes in `data/drafting_notes.md`), then we edited
   replies by hand until we'd send them (`data/edits.json`, 7 edits). The optimizer never sees the drafting notes.
3. **Split** (`data/splits.json`): 36 train, 12 validation, 12 test, seed 7. The test set is never shown to the optimizer.
4. **Optimize** (`src/optimize.py`): `dspy.GEPA(metric=judge_metric, reflection_lm=..., auto="light")`.
   The judge (`src/judge.py`) returns `dspy.Prediction(score, feedback)`, so GEPA can read *why* a reply missed.
5. **Regression test** (`promptfoo/`): the same model with the baseline prompt and the optimized prompt, on the 12 test tickets.

## Our run (25 Nov 2026)

Models: `claude-haiku-5-5` runs the bot; `claude-sonnet-5-5` is the judge and GEPA's reflection model.

| | Baseline prompt | Optimized prompt |
|---|---|---|
| DSPy judge, mean score on 12 test tickets | 0.646 | 0.854 |
| Promptfoo, tests passed (under 90 words + rubric >= 0.75) | 1 / 12 | 11 / 12 |

- GEPA tried 20 candidate prompts with `auto="light"` (about 428 metric calls). Best validation score: 0.969.
- 11 of 12 test tickets scored higher with the optimized prompt. One (`LATE-04`) scored lower: the reply
  escalated to a team lead when the customer hadn't asked. Small test sets move around; read the rows, not just the mean.
- The optimized prompt is long (see `prompts/optimized.txt`). It also copied policy facts into the prompt,
  so if your policy changes, re-run the optimizer.
- Every number above comes from `outputs/`. Your run will differ.

### What it cost

Computed from the token counts in each API response's `usage` field and the
[Anthropic list prices](https://platform.claude.com/docs/en/about-claude/pricing) on 25 Nov 2026
(Haiku 5.5: $0.10 in / $0.50 out per million tokens, prompts up to 100K; Sonnet 5.5: $2 in / $10 out).

| Step | Cost (USD) |
|---|---|
| Setup checks | 0.02 |
| Generate 60 tickets + draft gold replies | 0.28 |
| Baseline on test set | 0.11 |
| GEPA optimization (`auto="light"`) | 4.58 |
| Optimized prompt on test set | 0.10 |
| Promptfoo eval (24 replies + 24 rubric grades) | 0.30 |
| **Total** | **5.39** |

Per-step token counts are in `outputs/usage.json`. Most of the cost is the Sonnet judge inside GEPA.
DSPy caches identical calls locally, and cached calls cost nothing, so re-running the notebook is nearly free.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # put your key in .env, or export ANTHROPIC_API_KEY
test -n "$ANTHROPIC_API_KEY" && echo set     # checks the key without printing it

python -m src.pipeline          # baseline score, GEPA, test score  (paid: about $5 in our run)
python -m src.export_promptfoo  # write promptfoo/ prompts and tests
cd promptfoo && npx promptfoo@latest eval
```

Or open `notebook/stop_writing_prompts.ipynb` in Colab and add `ANTHROPIC_API_KEY` to Colab Secrets.
The notebook loads our saved optimized program by default (`RUN_GEPA = False`) so you can follow along for free;
set it to `True` to run the optimizer yourself.

To rebuild the dataset from scratch: `generate_scenarios.generate()`, `draft_gold.draft()`, then edit
`data/edits.json` and run `finalize_gold.finalize()` and `data.make_split()`.

## Notes and gotchas

- **Thinking tokens.** Current Claude models think adaptively, and thinking is billed as output.
  In our run, most of Haiku's output tokens were thinking, and some replies hit `max_tokens=1000` and were cut off.
  The optimizer noticed and added a "COMPLETENESS" rule. Give the bot more room (`max_tokens`) than you think it needs.
- **Feedback labels.** The judge's feedback starts with `[E F P T, N words]` (empathetic, friendly, to the point, faithful).
  GEPA's reflection model guessed different names for those letters. Spell labels out if you reuse this.
- **GEPA needs `gepa`**, which supports Python below 3.15.
- **Keys.** Never commit `.env`. Never print your key.

## Files

```
data/       policy, rubric, drafting notes, tickets, gold chats, hand edits, split
src/        config, signatures, judge, data, evaluate, optimize, pipeline, Promptfoo export
prompts/    baseline.txt, optimized.txt (written by the optimizer)
promptfoo/  promptfooconfig.yaml, prompt files, tests
notebook/   the Colab walkthrough used in the video
outputs/    real results, logs and token usage from our run
```

## Docs

- DSPy GEPA: https://dspy.ai/current/api/optimizers/GEPA/overview/
- Choosing a DSPy optimizer: https://dspy.ai/current/diving-deeper/choosing-an-optimizer/
- Promptfoo: https://www.promptfoo.dev/docs/getting-started/
- Anthropic models and pricing: https://platform.claude.com/docs/en/models/overview

MIT License.
