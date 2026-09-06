"""
Instrumentation non invasive du pipeline ALI : tokens, caracteres, horloge.

Aucune modification des composants C0-C4 n'est necessaire : ce module remplace a
chaud `gemini_client.gemini_call` et `c1_identifier._generate`, et enveloppe les
cinq methodes de composant pour attribuer chaque appel a son composant.

Ce qui est mesure, par appel :
    component   C0 | C1 | C2 | C3 | C4 | oracle
    backend     gemini | local | embedding
    in_tok/out_tok   -- Gemini : vrais compteurs `usage_metadata` ;
                        local  : comptes par le tokenizer du modele lui-meme
    in_chars/out_chars
    wall_s      horloge murale de l'appel

L'oracle (utilisateur simule) est instrumente lui aussi mais marque a part : il
est un artefact experimental partage avec les baselines, pas un cout du systeme.

Usage :
    from ali import cost_instrumentation as ci
    ci.install()
    ...  # executer le pipeline
    episode = ci.pop_episode()          # liste d'appels, remise a zero
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

_CALLS: list[dict] = []
_CURRENT: list[str] = []          # pile des composants actifs
_INSTALLED = False


def _tag() -> str:
    return _CURRENT[-1] if _CURRENT else "?"


@contextmanager
def component(name: str):
    _CURRENT.append(name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _CURRENT.pop()
        _CALLS.append({"component": name, "backend": "wrapper", "in_tok": 0,
                       "out_tok": 0, "in_chars": 0, "out_chars": 0,
                       "wall_s": dt, "span": True})


def record(**kw: Any) -> None:
    row = {"component": _tag(), "backend": "?", "in_tok": 0, "out_tok": 0,
           "in_chars": 0, "out_chars": 0, "wall_s": 0.0, "span": False}
    row.update(kw)
    _CALLS.append(row)


def pop_episode() -> list[dict]:
    global _CALLS
    out, _CALLS = _CALLS, []
    return out


# --------------------------------------------------------------------- patchs
def install() -> None:
    """Remplace a chaud les points d'appel. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import gemini_client
    from .components import c1_identifier

    # --- Gemini : on veut les VRAIS compteurs, donc on refait l'appel ici
    _orig_get_client = gemini_client.get_client

    def gemini_call(system_prompt: str, user_prompt: str,
                    model: str = "gemini-2.5-flash",
                    temperature: float = 0.0, max_retries: int = 3) -> str:
        client = _orig_get_client()
        last_err = None
        for attempt in range(max_retries):
            t0 = time.perf_counter()
            try:
                resp = client.models.generate_content(
                    model=model, contents=user_prompt,
                    config={"system_instruction": system_prompt,
                            "temperature": temperature},
                )
                dt = time.perf_counter() - t0
                text = resp.text or ""
                um = getattr(resp, "usage_metadata", None)
                record(backend="gemini", model=model,
                       in_tok=getattr(um, "prompt_token_count", 0) or 0,
                       out_tok=getattr(um, "candidates_token_count", 0) or 0,
                       in_chars=len(system_prompt) + len(user_prompt),
                       out_chars=len(text), wall_s=dt)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini call failed after {max_retries}: {last_err}")

    gemini_client.gemini_call = gemini_call

    # --- generation locale (C1/C4 fine-tunes) : tokens via le tokenizer du modele
    _orig_generate = c1_identifier._generate

    def _generate(tokenizer, model, device, prompt: str, **kw):
        t0 = time.perf_counter()
        out = _orig_generate(tokenizer, model, device, prompt, **kw)
        dt = time.perf_counter() - t0
        try:
            n_in = len(tokenizer.encode(prompt))
            n_out = len(tokenizer.encode(out))
        except Exception:                                   # noqa: BLE001
            n_in = n_out = 0
        record(backend="local", model=getattr(model.config, "_name_or_path", "local"),
               in_tok=n_in, out_tok=n_out,
               in_chars=len(prompt), out_chars=len(out), wall_s=dt)
        return out

    c1_identifier._generate = _generate


def wrap_pipeline(pipeline) -> None:
    """Enveloppe C0-C4 pour attribuer chaque appel interne a son composant."""
    for attr, meth, tag in (("c0", "extract", "C0"), ("c1", "identify", "C1"),
                            ("c2", "cluster", "C2"), ("c3", "generate", "C3"),
                            ("c4", "extract", "C4")):
        comp = getattr(pipeline, attr)
        orig = getattr(comp, meth)

        def make(orig=orig, tag=tag):
            def wrapped(*a, **k):
                with component(tag):
                    return orig(*a, **k)
            return wrapped

        setattr(comp, meth, make())


def wrap_oracle(user_fn):
    """L'oracle est mesure mais etiquete a part : artefact experimental."""
    def wrapped(question: str) -> str:
        with component("oracle"):
            return user_fn(question)
    return wrapped


# ------------------------------------------------------------------ agregation
def summarise(calls: list[dict]) -> dict:
    """Agrege un episode. Les 'span' ne portent que du temps (pas de tokens),
    et seuls les spans de PREMIER niveau comptent pour le temps total, sinon on
    additionnerait deux fois le temps des appels imbriques."""
    real = [c for c in calls if not c["span"]]
    spans = [c for c in calls if c["span"]]
    sys_calls = [c for c in real if c["component"] != "oracle"]
    ora_calls = [c for c in real if c["component"] == "oracle"]

    def agg(rows):
        return {"calls": len(rows),
                "in_tok": sum(r["in_tok"] for r in rows),
                "out_tok": sum(r["out_tok"] for r in rows),
                "chars": sum(r["in_chars"] + r["out_chars"] for r in rows),
                "wall_s": sum(r["wall_s"] for r in rows)}

    by_comp = {}
    for tag in ("C0", "C1", "C2", "C3", "C4"):
        rows = [c for c in real if c["component"] == tag]
        span_s = sum(s["wall_s"] for s in spans if s["component"] == tag)
        by_comp[tag] = {**agg(rows), "wall_s_incl": span_s}

    return {"system": agg(sys_calls), "oracle": agg(ora_calls), "by_component": by_comp}
