"""Oracle utilisateur reutilise de Phase 1 (meme principe que le scorer
Gemini de exp1_multiturn/run_multiturn.py)."""
from __future__ import annotations

from . import gemini_client

SYSTEM_TEMPLATE = """You are a user with the following need: {initial_prompt}
You have all the information needed to answer questions but only reveal it when
directly asked. Answer concisely and naturally. If asked for something not
specified in your need, invent a plausible concrete value consistent with it.
Never volunteer information that wasn't explicitly asked."""


class GeminiUserSimulator:
    def __init__(self, initial_prompt: str):
        self.initial_prompt = initial_prompt
        self.system = SYSTEM_TEMPLATE.format(initial_prompt=initial_prompt)

    def __call__(self, question: str) -> str:
        return gemini_client.gemini_call(self.system, question, temperature=0.3)
