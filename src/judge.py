"""LLM judge. Returns a score from 0 to 1 and written feedback, the shape GEPA reads."""
import dspy

from .config import read


class JudgeReply(dspy.Signature):
    """You grade one customer-support reply for Maple & Co.
    Use the rubric and the store policy. The gold reply shows the style the support lead wants;
    the reply does not need the same words, but it should be at least as good.
    Score each criterion 0, 1 or 2. Then give one or two sentences of specific feedback
    that says what to change in the reply. Quote the problem words if there are any."""

    rubric: str = dspy.InputField()
    policy: str = dspy.InputField()
    ticket: str = dspy.InputField()
    gold_reply: str = dspy.InputField()
    reply: str = dspy.InputField()
    empathetic: int = dspy.OutputField(desc="0, 1 or 2")
    friendly: int = dspy.OutputField(desc="0, 1 or 2")
    to_the_point: int = dspy.OutputField(desc="0, 1 or 2")
    faithful: int = dspy.OutputField(desc="0, 1 or 2")
    feedback: str = dspy.OutputField()


_judge = dspy.Predict(JudgeReply)
JUDGE_LM = None  # set by set_judge_lm()


def set_judge_lm(lm):
    global JUDGE_LM
    JUDGE_LM = lm


def grade(ticket: str, gold_reply: str, reply: str) -> dspy.Prediction:
    with dspy.context(lm=JUDGE_LM):
        g = _judge(rubric=read("rubric.md"), policy=read("policy.md"),
                   ticket=ticket, gold_reply=gold_reply, reply=reply)
    parts = [max(0, min(2, int(x))) for x in (g.empathetic, g.friendly, g.to_the_point, g.faithful)]
    score = sum(parts) / 8
    if parts[3] == 0:          # invented something: fail
        score = min(score, 0.25)
    words = len(reply.split())
    note = f"[E{parts[0]} F{parts[1]} P{parts[2]} T{parts[3]}, {words} words] {g.feedback}"
    return dspy.Prediction(score=score, feedback=note, parts=parts)


def judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric: dspy.Prediction(score, feedback)."""
    r = grade(gold.ticket, gold.reply, pred.reply)
    return dspy.Prediction(score=r.score, feedback=r.feedback)
