"""
C1 -- Element Identifier. Fine-tuned (Telos, FinAgent) ou Gemini (Rad-Assist,
fallback, ou "auto" si le diagnostic Section 2 echoue).

Formats de prompt reels (extraits de checkpoints/{telos,chain_pilot}/inference_*.py,
voir aussi checkpoints/extraction_report.md) :

  Telos (GPT-2 + LoRA r=16) :
    input  : "[MISSION] {texte} [IDENTIFY]"
    output : "category: <cat> | elements: name=score, name=score, ..."

  FinAgent (Qwen2.5-0.5B-Instruct + LoRA r=32) :
    input  : "[DOMAIN] {domaine_natif} [MESSAGE] {message}"
    output : "param=weight, param2=weight2, ..."

NOTE POUR L'AGENT LOCAL : le chargement des modeles (torch.load / PeftModel)
n'a pas pu etre teste dans ce sandbox (huggingface.co bloque par le proxy
reseau). Le code est ecrit d'apres les scripts d'inference reels deja
valides (extraction_report.md : "PeftConfig.from_pretrained() ... se
chargent sans erreur"), mais l'appel .generate() lui-meme est non verifie ici.
"""
from __future__ import annotations

import re

from .. import config
from .. import gemini_client
from ..taxonomy_mapping import map_native_to_flat

_MODEL_CACHE: dict[str, tuple] = {}  # (deployment, role) -> (tokenizer, model)


def _load_peft_model(base_model_name: str, checkpoint_dir: str, use_gpt2_class: bool):
    import torch
    from peft import PeftModel

    if use_gpt2_class:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = GPT2LMHeadModel.from_pretrained(base_model_name)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        base = AutoModelForCausalLM.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # acceleration Apple Silicon, non exploitee par defaut
    else:
        device = "cpu"
    model = model.to(device)
    return tokenizer, model, device


def _generate(
    tokenizer, model, device, prompt: str, max_new_tokens: int = 250,
    do_sample: bool = True,
) -> str:
    """Genere et decode UNIQUEMENT les tokens nouvellement produits (slicing par
    index de token, pas par prefixe de chaine -- le round-trip tokenize/decode
    ne reproduit pas toujours le prompt caractere pour caractere, ce qui cassait
    silencieusement le parsing en aval : cf. checkpoints/chain_pilot/inference_c1_c4.py
    (_generate()), qui utilise deja ce slicing token-level comme reference."""
    import torch
    max_position = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 1024
    )
    max_input_tokens = max(16, max_position - max_new_tokens)
    # truncation_side="left" : les prompts ([ANSWER] {texte} [TARGETS] ... [EXTRACT])
    # placent l'instruction critique (marqueurs + cibles) a la fin. Tronquer par
    # defaut a droite supprimerait silencieusement ces marqueurs pour toute reponse
    # longue et cassait l'extraction en aval.
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens
        ).to(device)
    finally:
        tokenizer.truncation_side = original_side
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class C1Identifier:
    def __init__(self, deployment: str, mode: str = "auto"):
        self.deployment = deployment
        self.mode = self._resolve_mode(deployment, mode)
        if self.mode == "finetuned":
            self._load()

    def _resolve_mode(self, deployment: str, mode: str) -> str:
        defaults = config.DEPLOYMENT_DEFAULTS[deployment]
        # Contrainte dure au niveau du deploiement (ex: rad_assist n'a aucun
        # adapter PEFT) -- prioritaire sur le mode demande par l'appelant, meme
        # si celui-ci n'est pas "auto" (run_phase2.py force "finetuned" pour
        # ALI_v1/v3 sur TOUS les deploiements ; rad_assist doit quand meme
        # retomber sur gemini).
        forced = defaults.get("c1_mode")
        if forced in ("gemini", "finetuned"):
            return forced
        if mode == "auto":
            # "auto" pour telos/finagent -> resultat du diagnostic (Section 2),
            # lu depuis diagnostics/diagnostic_report.json s'il existe.
            return _read_diagnostic_decision(deployment, "c1_decision") or "gemini"
        return mode

    def _load(self):
        d = config.DEPLOYMENT_DEFAULTS[self.deployment]
        if self.deployment == "telos":
            self.tokenizer, self.model, self.device = _load_peft_model(
                d["base_model"], d["checkpoint_c1"], use_gpt2_class=True
            )
        elif self.deployment == "finagent":
            self.tokenizer, self.model, self.device = _load_peft_model(
                d["base_model_c1"], d["checkpoint_c1"], use_gpt2_class=False
            )
        else:
            raise ValueError(f"{self.deployment} n'a pas de C1 fine-tune (utiliser mode='gemini')")

    def identify(self, state) -> list[dict]:
        """Retourne la liste des params non resolus a cibler en priorite
        (dicts {name, weight, description}, deja mappes sur la taxonomie plate)."""
        unresolved = [
            {"name": name, **p} for name, p in state.taxonomy["parameters"].items()
            if name not in state.F
        ]
        if not unresolved:
            return []

        if self.mode == "finetuned":
            native_names = self._finetuned_identify(state)
            mapped = self._map_native_names(native_names)
            # Ne garder que les params effectivement non-resolus, dans l'ordre
            # de priorite renvoye par C1 natif ; sinon retomber sur le poids.
            by_name = {u["name"]: u for u in unresolved}
            ordered = [by_name[m] for m in mapped if m in by_name]
            remaining = [u for u in unresolved if u["name"] not in {o["name"] for o in ordered}]
            remaining.sort(key=lambda p: p["weight"], reverse=True)
            return ordered + remaining
        else:
            names = gemini_identify_params(state, unresolved)
            by_name = {u["name"]: u for u in unresolved}
            ordered = [by_name[n] for n in names if n in by_name]
            remaining = [u for u in unresolved if u["name"] not in {o["name"] for o in ordered}]
            remaining.sort(key=lambda p: p["weight"], reverse=True)
            return ordered + remaining or unresolved

    def _map_native_names(self, native_names: list[str]) -> list[str]:
        mapped = []
        for n in native_names:
            flat = map_native_to_flat(self.deployment, n)
            if flat:
                mapped.append(flat)
        return mapped

    def _finetuned_identify(self, state) -> list[str]:
        if self.deployment == "telos":
            prompt = f"[MISSION] {state.prompt} [IDENTIFY]"
            output = _generate(self.tokenizer, self.model, self.device, prompt)
            return _parse_telos_c1_output(output)
        elif self.deployment == "finagent":
            native_domain = getattr(state, "native_domain", "trading")
            prompt = f"[DOMAIN] {native_domain} [MESSAGE] {state.prompt}"
            output = _generate(self.tokenizer, self.model, self.device, prompt, do_sample=False)
            return _parse_finagent_c1_output(output)
        return []


def _parse_telos_c1_output(text: str) -> list[str]:
    """'category: <cat> | elements: name=score, name=score, ...' -> [names triees par score desc]"""
    text = text.split("\n")[0].split("[")[0].strip()
    parts = text.split("|")
    if len(parts) < 2:
        return []
    elem_part = "|".join(parts[1:])
    pairs = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(-?\d+)", elem_part)
    pairs.sort(key=lambda p: int(p[1]), reverse=True)
    return [name for name, _ in pairs]


def _parse_finagent_c1_output(text: str) -> list[str]:
    """'param=weight, param2=weight2' -> [names triees par weight desc]"""
    text = text.split("\n")[0].strip()
    pairs = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)", text)
    pairs.sort(key=lambda p: float(p[1]), reverse=True)
    return [name for name, _ in pairs]


def gemini_identify_params(state, unresolved: list[dict]) -> list[str]:
    system = (
        "You are the Element Identifier (C1) of a parameter elicitation pipeline. "
        "Given the conversation so far and a list of unresolved parameters, "
        "return a JSON list of parameter names, ordered by priority "
        "(most important / most blocking first). Output ONLY a JSON array of strings."
    )
    history_txt = "\n".join(f"Q: {q}\nA: {a}" for q, a in state.history) or "(no turns yet)"
    params_txt = "\n".join(f"- {p['name']}: {p['description']} (weight={p['weight']})" for p in unresolved)
    user = f"Initial prompt: {state.prompt}\n\nConversation:\n{history_txt}\n\nUnresolved parameters:\n{params_txt}"
    response = gemini_client.gemini_call(system, user)
    parsed = gemini_client.parse_json_safe("{\"list\": " + response.strip() + "}") if response.strip().startswith("[") else {}
    if "list" in parsed and isinstance(parsed["list"], list):
        return [str(x) for x in parsed["list"]]
    # fallback: extraire des noms entre guillemets
    return re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', response)


def _read_diagnostic_decision(deployment: str, key: str) -> str | None:
    import json
    path = config.REPO_ROOT / "checkpoints" / "ali_standalone" / "diagnostics" / "diagnostic_report.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            report = json.load(f)
        return report.get(deployment, {}).get(key)
    except Exception:
        return None
