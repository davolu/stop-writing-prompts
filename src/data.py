"""Load gold chats and the frozen train / validation / test split."""
import json
import random

import dspy

from .config import DATA, read


def load_gold() -> list[dict]:
    return json.loads((DATA / "gold_chats.json").read_text())


def make_split(seed: int = 7, n_val: int = 12, n_test: int = 12, force_test=("T-vase",)) -> dict:
    """Shuffle once with a fixed seed. Teaching tickets in force_test always land in test."""
    ids = [c["id"] for c in load_gold()]
    forced = [i for i in ids if i in force_test]
    rest = [i for i in ids if i not in force_test]
    random.Random(seed).shuffle(rest)
    test = forced + rest[: n_test - len(forced)]
    val = rest[n_test - len(forced): n_test - len(forced) + n_val]
    train = rest[n_test - len(forced) + n_val:]
    split = {"seed": seed, "train": train, "val": val, "test": test}
    (DATA / "splits.json").write_text(json.dumps(split, indent=2))
    return split


def examples(part: str) -> list[dspy.Example]:
    split = json.loads((DATA / "splits.json").read_text())
    by_id = {c["id"]: c for c in load_gold()}
    policy = read("policy.md")
    return [dspy.Example(id=i, category=by_id[i]["category"], policy=policy,
                         ticket=by_id[i]["ticket"], reply=by_id[i]["gold_reply"]).with_inputs("policy", "ticket")
            for i in split[part]]
