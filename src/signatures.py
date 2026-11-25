"""The support bot: one DSPy signature, one predictor."""
import dspy

BASELINE = "You are a helpful support agent for Maple & Co. Be nice and solve the customer's problem."


class SupportReply(dspy.Signature):
    """Placeholder. Replaced by with_instructions()."""

    policy: str = dspy.InputField(desc="Maple & Co. store policy")
    ticket: str = dspy.InputField(desc="the customer's message")
    reply: str = dspy.OutputField(desc="the reply to send to the customer")


def make_bot(instructions: str = BASELINE) -> dspy.Module:
    return dspy.Predict(SupportReply.with_instructions(instructions))
