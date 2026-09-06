"""
C4 -- Answer Extractor. Fine-tuned (Telos, FinAgent) ou Gemini.

Formats reels (checkpoints/{telos,chain_pilot}/inference_*.py) :

  Telos C4 (GPT-2 + LoRA, "extractor_llm") :
    input  : "[ANSWER] {reponse} [TARGETS] {cibles} [UNDEFINED] {non_resolus} [EXTRACT]"
    output : "resolved: elem=value, elem=value | bonus: elem=value"

  FinAgent C4 (Qwen2.5-1.5B-Instruct + LoRA, nomme "c3_param_extractor" cote
  FinAgent -- numerotation differente, voir extraction_report.md) :
    input  : "[ANSWER] {reponse} [TARGETS] {cibles} [EXTRACT]"
    output : "resolved: p=v | p2=v2 | nulled: p3"

Meme avertissement que c1_identifier.py : chargement non teste (reseau HF
bloque dans ce sandbox).
"""
from __future__ import annotations

import re

from . import c1_identifier as _c1  # reutilise _load_peft_model / _generate
from .. import config
from .. import gemini_client
from ..taxonomy_mapping import map_native_to_flat


class C4Extractor:
    LABELS = ["RESOLVED", "MENTIONED", "ABSENT"]

    def __init__(self, deployment: str, mode: str = "auto"):
        self.deployment = deployment
        self.mode = self._resolve_mode(deployment, mode)
        if self.mode == "finetuned":
            self._load()

    def _resolve_mode(self, deployment: str, mode: str) -> str:
        d = config.DEPLOYMENT_DEFAULTS[deployment]
        # Contrainte dure au niveau du deploiement, prioritaire sur le mode
        # demande (meme raisonnement que C1Identifier._resolve_mode).
        forced = d.get("c4_mode")
        if forced in ("gemini", "finetuned"):
            return forced
        if mode == "auto":
            return _c1._read_diagnostic_decision(deployment, "c4_decision") or "gemini"
        return mode

    def _load(self):
        d = config.DEPLOYMENT_DEFAULTS[self.deployment]
        if self.deployment == "telos":
            self.tokenizer, self.model, self.device = _c1._load_peft_model(
                d["base_model"], d["checkpoint_c4"], use_gpt2_class=True
            )
        elif self.deployment == "finagent":
            self.tokenizer, self.model, self.device = _c1._load_peft_model(
                d["base_model_c4"], d["checkpoint_c4"], use_gpt2_class=False
            )
        else:
            raise ValueError(f"{self.deployment} n'a pas de C4 fine-tune (utiliser mode='gemini')")

    def extract(self, response: str, targeted_params: list[dict], state) -> None:
        """Met a jour state.F pour chaque parametre cible selon la reponse."""
        if self.mode == "finetuned":
            resolved, bonus = self._finetuned_extract(response, targeted_params, state)
        else:
            resolved, bonus = self._gemini_extract(response, targeted_params, state)

        for name, value in {**resolved, **bonus}.items():
            if name in state.taxonomy["parameters"]:
                state.F[name] = value

        from ..coverage_scorer import CoverageScorer
        state.cov_history.append(CoverageScorer(state.taxonomy).compute(state.F))

    # --- Fine-tuned path -----------------------------------------------

    def _finetuned_extract(self, response, targeted_params, state):
        targets_str = ", ".join(p["name"] for p in targeted_params)
        if self.deployment == "telos":
            undefined = [p["name"] for p in state.taxonomy["parameters"].values()] \
                if False else []  # placeholder si besoin d'etendre plus tard
            prompt = f"[ANSWER] {response} [TARGETS] {targets_str} [UNDEFINED]  [EXTRACT]"
            output = _c1._generate(self.tokenizer, self.model, self.device, prompt)
            resolved_native, bonus_native = _parse_telos_c4_output(output)
        elif self.deployment == "finagent":
            prompt = f"[ANSWER] {response} [TARGETS] {targets_str} [EXTRACT]"
            output = _c1._generate(self.tokenizer, self.model, self.device, prompt, do_sample=False)
            resolved_native, bonus_native = _parse_finagent_c4_output(output)
        else:
            return {}, {}

        resolved = self._map_dict(resolved_native)
        bonus = self._map_dict(bonus_native)
        return resolved, bonus

    def _map_dict(self, native_dict: dict) -> dict:
        mapped = {}
        for native_name, value in native_dict.items():
            flat = map_native_to_flat(self.deployment, native_name)
            if flat:
                mapped[flat] = value
        return mapped

    # --- Gemini path -----------------------------------------------------

    def _gemini_extract(self, response, targeted_params, state):
        system = (
            "You are the Answer Extractor (C4) of a parameter elicitation pipeline. "
            "Given the user's answer and a list of targeted parameters, classify each "
            "parameter as RESOLVED (concrete value given), MENTIONED (discussed but no "
            "concrete value), or ABSENT (not addressed). For RESOLVED parameters, extract "
            "the concrete value. Output ONLY a JSON dict: "
            '{"param_name": {"label": "RESOLVED|MENTIONED|ABSENT", "value": "..."}}'
        )
        params_txt = "\n".join(f"- {p['name']}: {p['description']}" for p in targeted_params)
        user = f"User's answer: {response}\n\nTargeted parameters:\n{params_txt}"
        raw = gemini_client.gemini_call(system, user)
        parsed = gemini_client.parse_json_safe(raw)
        resolved = {
            name: v.get("value", "")
            for name, v in parsed.items()
            if isinstance(v, dict) and v.get("label") == "RESOLVED"
        }
        return resolved, {}


def _parse_telos_c4_output(text: str) -> tuple[dict, dict]:
    text = text.split("\n")[0].split("[")[0].strip()
    resolved, bonus = {}, {}
    m_resolved = re.search(r"resolved:\s*([^|]*)", text, re.IGNORECASE)
    m_bonus = re.search(r"bonus:\s*(.*)", text, re.IGNORECASE)
    if m_resolved:
        resolved = _parse_kv_list(m_resolved.group(1))
    if m_bonus:
        bonus = _parse_kv_list(m_bonus.group(1))
    return resolved, bonus


def _parse_finagent_c4_output(text: str) -> tuple[dict, dict]:
    text = text.split("\n")[0].split("[")[0].strip()
    resolved = {}
    m_resolved = re.search(r"resolved:\s*([^|]*)", text, re.IGNORECASE)
    if m_resolved:
        resolved = _parse_kv_list(m_resolved.group(1))
    # "nulled: p3" -> params explicitement remis a zero -- pas de valeur, ignore ici
    return resolved, {}


def _parse_kv_list(segment: str) -> dict:
    out = {}
    for pair in segment.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if k:
            out[k] = v
    return out
