"""Step 1: generate synthetic customer tickets with the judge model."""
import json

import dspy

from .config import DATA, read

TEACHING = [
    dict(id="T-vase", category="damaged", customer_name="Ada",
         ticket="Hi, my ceramic vase arrived cracked and I need this refunded today. Order #4812. - Ada"),
    dict(id="T-where", category="where_is_order", customer_name="Tunde",
         ticket="Where is my order? It's been a week. #5107"),
    dict(id="T-late", category="late_return", customer_name="Grace",
         ticket="Hello, I'd like to return a throw blanket I bought. It was delivered 41 days ago and I never used it. Grace"),
    dict(id="T-loyal", category="exception", customer_name="Mark",
         ticket="Can you just make an exception and refund my lamp? It's 3 months old. I've been a loyal customer for years. Mark"),
]

MIX = {"where_is_order": 10, "damaged": 9, "return_in_window": 8, "late_return": 7,
       "exchange": 8, "exception": 7, "change_or_cancel": 7}


class WriteTickets(dspy.Signature):
    """Write realistic, varied customer-support emails for Maple & Co., an online home-goods shop.
    Vary tone (calm, upset, rushed, confused), length (1 to 4 sentences), and products.
    Include order numbers like #4xxx or #5xxx in most tickets. Sign about two thirds with a first name.
    Each ticket must be about the given category and answerable from the policy."""

    policy: str = dspy.InputField()
    category: str = dspy.InputField()
    count: int = dspy.InputField()
    tickets: list[str] = dspy.OutputField(desc="exactly `count` customer emails")


def generate() -> list[dict]:
    writer = dspy.Predict(WriteTickets)
    rows = list(TEACHING)
    for cat, n in MIX.items():
        out = writer(policy=read("policy.md"), category=cat, count=n)
        for k, t in enumerate(out.tickets[:n]):
            rows.append(dict(id=f"{cat[:4].upper()}-{k + 1:02d}", category=cat, ticket=t.strip()))
    (DATA / "scenarios.json").write_text(json.dumps(rows, indent=2))
    return rows
