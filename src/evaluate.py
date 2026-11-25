"""Score a bot on a set of examples with the judge, in parallel."""
from concurrent.futures import ThreadPoolExecutor

from .judge import grade


def run(bot, exs, threads: int = 6) -> dict:
    def one(ex):
        reply = bot(policy=ex.policy, ticket=ex.ticket).reply
        g = grade(ex.ticket, ex.reply, reply)
        return dict(id=ex.id, category=ex.category, reply=reply, score=g.score, parts=g.parts, feedback=g.feedback)
    with ThreadPoolExecutor(threads) as pool:
        rows = list(pool.map(one, exs))
    mean = sum(r["score"] for r in rows) / len(rows)
    return dict(mean=round(mean, 3), rows=rows)
