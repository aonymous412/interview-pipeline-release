"""C3 -- Question Generator, 3 tiers (clarification / boundary / confirmation).
Toujours Gemini (pas de fine-tuning pour C3 dans la spec Phase 2 -- les
question_templates.json / c2 question gen des projets originaux existent mais
sont hors-scope Phase 2, cf. prompt Section 4)."""
from __future__ import annotations

from .. import gemini_client

TEMPLATES = {
    "clarification": "To proceed with the task, I need to understand {param}. {question}",
    "boundary": "You mentioned {value}. To confirm: {question}",
    "confirmation": "Just to confirm -- {param}: is that correct?",
}


class C3Generator:
    def generate(self, cluster: list[dict], state) -> str:
        primary = max(cluster, key=lambda p: p["weight"])
        tier = self._select_tier(primary, state)
        return self._gemini_generate(primary, tier, state)

    def _select_tier(self, param: dict, state) -> str:
        mentioned = getattr(state, "mentioned", set())
        if param["name"] in mentioned:
            return "boundary"
        if param["name"] in state.F:
            return "confirmation"
        return "clarification"

    def _gemini_generate(self, param: dict, tier: str, state) -> str:
        system = (
            "You are the Question Generator (C3) of a parameter elicitation pipeline. "
            f"Generate a single, natural, concise question (tier={tier}) to elicit the "
            "parameter below from the user. Output ONLY the question text, no preamble."
        )
        history_txt = "\n".join(f"Q: {q}\nA: {a}" for q, a in state.history) or "(no turns yet)"
        user = (
            f"Initial prompt: {state.prompt}\n\nConversation so far:\n{history_txt}\n\n"
            f"Parameter to elicit: {param['name']} -- {param['description']} "
            f"(weight={param['weight']})\nTemplate hint: {TEMPLATES[tier]}"
        )
        question = gemini_client.gemini_call(system, user)
        return question.strip()
