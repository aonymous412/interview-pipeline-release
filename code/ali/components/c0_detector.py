"""
C0 -- Domain Detector + Pre-Extractor.

Zero-shot, aucun fine-tuning requis (Gemini uniquement).

Pour FinAgent specifiquement, C0 doit d'abord detecter le DOMAINE NATIF
(trading / payroll / content / procurement / generic) via les `keywords` de
chaque DomainTemplate (voir checkpoints/chain_pilot/taxonomy_extracted_chain_pilot.py)
avant d'extraire les parametres -- c'est necessaire pour que C1 natif (entraine
par domaine) recoive le bon contexte, et pour que taxonomy_mapping puisse
calculer les uncoverable_params du bon domaine natif.
"""
from __future__ import annotations

from .. import gemini_client

SYSTEM_PROMPT = """You are a parameter extractor. Given a user prompt and a domain
taxonomy, identify which parameters already have concrete values in the prompt.
For each parameter found, output: {"param_name": "value_extracted"}.
Output ONLY the JSON dict, nothing else."""


class C0Detector:
    def extract(self, prompt: str, taxonomy: dict) -> dict:
        params_list = [
            f"{name}: {p['description']}" for name, p in taxonomy["parameters"].items()
        ]
        user_prompt = (
            f"User prompt: {prompt}\n\nParameters to look for:\n"
            + "\n".join(f"- {p}" for p in params_list)
        )
        response = gemini_client.gemini_call(SYSTEM_PROMPT, user_prompt)
        return gemini_client.parse_json_safe(response)

    def detect_finagent_native_domain(self, prompt: str) -> str:
        """Detection par mots-cles (identique a la logique originale
        FinAgent -- pas besoin de LLM, c'est deterministe)."""
        from ..taxonomy_mapping import FINAGENT_NATIVE_DOMAINS

        # Import local des keywords (copie depuis taxonomy_extracted_chain_pilot.py
        # pour eviter une dependance sur le module externe non empaquete).
        KEYWORDS = {
            "trading": ["trade", "trading", "trader", "buy", "sell", "exchange",
                        "crypto", "stock", "forex", "eth", "btc", "shares",
                        "securities", "assets", "portfolio", "investment",
                        "market", "ticker"],
            "payroll": ["payroll", "salary", "pay", "payment", "wage",
                        "compensation", "employee", "worker", "staff",
                        "personnel", "hr", "benefits", "payday", "biweekly",
                        "monthly pay"],
            "content": ["content", "moderation", "moderate", "filter", "block",
                        "censor", "topic", "subject", "mention", "talk about",
                        "discuss", "generate", "writing", "teaching",
                        "training", "llm", "ai model", "chatbot",
                        "geopolitical", "political", "religious", "nsfw",
                        "inappropriate"],
            "procurement": ["buy", "purchase", "order", "procurement",
                            "supplier", "vendor", "inventory", "stock",
                            "goods", "products", "materials", "supplies",
                            "vegetables", "food", "restaurant", "grocery",
                            "wholesale"],
        }
        text = prompt.lower()
        scores = {dom: sum(1 for kw in kws if kw in text) for dom, kws in KEYWORDS.items()}
        best_dom = max(scores, key=scores.get)
        if scores[best_dom] == 0:
            return "generic"
        return best_dom
