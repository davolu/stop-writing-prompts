"""Step 3: apply our hand edits to the drafts and save data/gold_chats.json."""
import json
import re

from .config import DATA


def normalize(reply: str) -> str:
    reply = reply.strip()
    reply = re.sub(r"\s*Warmly,\s*Maple & Co\. Support\s*$", "\n\nWarmly,\nMaple & Co. Support", reply)
    reply = re.sub(r"^(Hi [^,\n]+,|Hello,)\s*", r"\1\n\n", reply)
    return reply


def finalize() -> list[dict]:
    drafts = json.loads((DATA / "gold_drafts.json").read_text())
    edits = json.loads((DATA / "edits.json").read_text())
    out = []
    for d in drafts:
        reply = edits[d["id"]]["reply"] if d["id"] in edits else d["gold_reply"]
        out.append(dict(id=d["id"], category=d["category"], ticket=d["ticket"],
                        gold_reply=normalize(reply), edited=d["id"] in edits))
    (DATA / "gold_chats.json").write_text(json.dumps(out, indent=2))
    return out
