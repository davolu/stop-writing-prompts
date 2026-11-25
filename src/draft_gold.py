"""Step 2: draft a gold reply for every ticket. We then edit them by hand."""
import json
from concurrent.futures import ThreadPoolExecutor

import dspy

from .config import DATA, read


class DraftGold(dspy.Signature):
    """You are the Maple & Co. support lead. Write the reply you would send, following your house style exactly."""

    house_style: str = dspy.InputField()
    policy: str = dspy.InputField()
    ticket: str = dspy.InputField()
    reply: str = dspy.OutputField()


def draft() -> list[dict]:
    rows = json.loads((DATA / "scenarios.json").read_text())
    drafter = dspy.Predict(DraftGold)
    style, policy = read("drafting_notes.md"), read("policy.md")

    def one(r):
        return dict(**r, gold_reply=drafter(house_style=style, policy=policy, ticket=r["ticket"]).reply.strip())

    with ThreadPoolExecutor(6) as pool:
        out = list(pool.map(one, rows))
    (DATA / "gold_drafts.json").write_text(json.dumps(out, indent=2))
    return out
