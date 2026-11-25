"""Build notebook/stop_writing_prompts.ipynb from the cells below.

    python notebook/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebook/stop_writing_prompts.ipynb
"""
from pathlib import Path

import nbformat as nbf

C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Stop Writing Prompts: Let Your Best Chats Write Them for You

We don't write the system prompt. We give an optimizer a short **rubric** and **60 labelled example chats**,
and DSPy's **GEPA** optimizer writes and tests the prompt, graded by an **LLM judge**.
Then **Promptfoo** checks the new prompt against the old one on tickets the optimizer never saw.

Demo: *Maple & Co.*, a fictional home-goods shop. All tickets are synthetic.
Models: Anthropic's Claude API (`claude-haiku-5-5` runs the bot, `claude-sonnet-5-5` judges and writes new prompts).
This uses a paid API. Check the pricing page before you run.
""")
code("""
# Setup. In Colab this clones the repo and installs pinned packages.
import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !git clone -q https://github.com/davolu/stop-writing-prompts.git
    %cd stop-writing-prompts
    %pip -q install -r requirements.txt
    from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")  # from Colab Secrets
else:
    os.chdir(os.path.dirname(os.getcwd()) if os.getcwd().endswith("notebook") else os.getcwd())
sys.path.insert(0, os.getcwd())
""")
code("""
# Check the key is set. This never prints the key.
!test -n "$ANTHROPIC_API_KEY" && echo set
""")
md("## 1. The rubric: what a good reply looks like")
code("""
from src.config import read
print(read("rubric.md"))
""")
md("## 2. Gold chats: example replies we'd actually send")
code("""
import json
from src.data import load_gold
gold = load_gold()
print(len(gold), "gold chats,", sum(g["edited"] for g in gold), "edited by hand")
ex = next(g for g in gold if g["id"] == "T-late")
print("\\nTICKET:", ex["ticket"])
print("\\nGOLD REPLY:\\n" + ex["gold_reply"])
""")
code("""
# One of our hand edits: the draft was too long and offered an exception nobody asked for.
drafts = {d["id"]: d for d in json.load(open("data/gold_drafts.json"))}
edits = json.load(open("data/edits.json"))
print("WHY:", edits["T-late"]["why"])
print("\\nDRAFT (%d words):\\n%s" % (len(drafts["T-late"]["gold_reply"].split()), drafts["T-late"]["gold_reply"]))
""")
md("## 3. Train, validation and test")
code("""
from src.data import examples
split = json.load(open("data/splits.json"))
print({k: len(v) for k, v in split.items() if k != "seed"}, "seed =", split["seed"])
print("cracked vase is in test:", "T-vase" in split["test"])
train, val, test = examples("train"), examples("val"), examples("test")
""")
md("## 4. Models, the baseline prompt and the judge")
code("""
import dspy
from src.config import make_lms
from src.judge import set_judge_lm, grade
from src.signatures import BASELINE, make_bot

task_lm, judge_lm = make_lms()      # haiku-5-5 runs the bot, sonnet-5-5 judges
dspy.configure(lm=task_lm)
set_judge_lm(judge_lm)

baseline_bot = make_bot(BASELINE)
print("BASELINE PROMPT:", BASELINE)
""")
code("""
vase = next(e for e in test if e.id == "T-vase")
reply = baseline_bot(policy=vase.policy, ticket=vase.ticket).reply
print(f"BASELINE REPLY ({len(reply.split())} words):\\n{reply}")
g = grade(vase.ticket, vase.reply, reply)
print(f"\\nJUDGE score={g.score:.3f}\\n{g.feedback}")
""")
code("""
from src.evaluate import run
base = run(baseline_bot, test)
print(f"Baseline test score: {base['mean']:.3f}")
""")
md("## 5. Let GEPA write the prompt")
code("""
from src.optimize import optimize, export
RUN_GEPA = False   # True re-runs the optimizer (paid). False loads the program we saved from our run.

if RUN_GEPA:
    optimized_bot = optimize(train, val, reflection_lm=judge_lm)   # auto="light"
else:
    optimized_bot = make_bot()
    optimized_bot.load("outputs/optimized_program.json")
print(open("prompts/optimized.txt").read())
""")
code("""
s = json.load(open("outputs/gepa_summary.json"))
print("candidates tried:", s["candidates"], " winner:", s["best_idx"])
print("validation scores:", s["val_aggregate_scores"])
""")
md("## 6. Score the new prompt on the held-out test set")
code("""
opt = run(optimized_bot, test)
print(f"Baseline test score:  {base['mean']:.3f}")
print(f"Optimized test score: {opt['mean']:.3f}")
""")
code("""
reply = optimized_bot(policy=vase.policy, ticket=vase.ticket).reply
print(f"OPTIMIZED REPLY ({len(reply.split())} words):\\n{reply}")
print("\\nJUDGE:", grade(vase.ticket, vase.reply, reply).feedback)
""")
md("## 7. Regression test with Promptfoo")
code("""
from src.export_promptfoo import export as export_pf
print("tests:", export_pf())
RUN_PROMPTFOO = False   # True re-runs the eval (paid). False reads the results we saved from our run.
if RUN_PROMPTFOO:
    !cd promptfoo && npx -y promptfoo@latest eval --no-progress-bar -o ../outputs/promptfoo_results.json
    from src.promptfoo_usage import record as record_pf; record_pf()
pf = json.load(open("outputs/promptfoo_results.json"))["results"]
for p in pf["prompts"]:
    m = p["metrics"]
    print(f'{p["label"]:10} passed {m["testPassCount"]:2}/12   under_90_words {m["namedScores"]["under_90_words"]:.0f}/12   rubric avg {m["namedScores"]["rubric"]/12:.2f}')
""")
md("## 8. What it cost (from the API usage fields)")
code("""
from src import usage
usage.record("6_notebook_run", [task_lm, judge_lm])   # calls this notebook made (cached calls are free)
t = usage.total()
usage.print_usage(t)
print(f"TOTAL ${sum(r['cost_usd'] for r in t.values()):.2f}")
""")
nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "colab": {"provenance": []}}
out = Path(__file__).with_name("stop_writing_prompts.ipynb")
nbf.write(nb, out); print("wrote", out)
