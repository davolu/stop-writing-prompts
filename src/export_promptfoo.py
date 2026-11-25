"""Step 7: write Promptfoo prompt files and tests from the frozen test split."""
import json

from .config import DATA, PROMPTS, ROOT, read
from .data import examples

PF = ROOT / "promptfoo"
USER = "Store policy:\n{{policy}}\n\nCustomer message:\n{{ticket}}"


def chat(system: str) -> str:
    assert "{{" not in system and "{%" not in system, "system prompt would be read as a template"
    return json.dumps([{"role": "system", "content": system.strip()},
                       {"role": "user", "content": USER}], indent=2)


def export():
    PF.mkdir(exist_ok=True)
    (PF / "baseline.json").write_text(chat((PROMPTS / "baseline.txt").read_text()))
    (PF / "optimized.json").write_text(chat((PROMPTS / "optimized.txt").read_text()))
    lines = []
    for ex in examples("test"):
        lines += [f"- description: {json.dumps(ex.id)}",
                  "  vars:",
                  f"    ticket: {json.dumps(ex.ticket)}",
                  f"    gold: {json.dumps(ex.reply)}"]
    (PF / "tests.yaml").write_text("\n".join(lines) + "\n")
    (PF / "policy.txt").write_text(read("policy.md") + "\n")
    (PF / "rubric.txt").write_text(read("rubric.md") + "\n")
    return len(lines) // 4


if __name__ == "__main__":
    print("tests:", export())
