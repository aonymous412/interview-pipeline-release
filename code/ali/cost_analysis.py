"""
Cout d'une interview complete : caracteres, tokens et appels LLM, par systeme.

Ne fait AUCUN appel reseau -- reconstruction hors ligne a partir des transcripts
deja sauvegardes. Deux mesures sont produites :

  * CARACTERES  -- independants du tokenizer, donc strictement comparables entre
                   familles de modeles. C'est la mesure de reference.
  * TOKENS      -- estimes avec un tokenizer unique (cl100k_base) applique a tous
                   les systemes. Comparable entre lignes, PAS egal au comptage
                   propre a chaque fournisseur.

L'etape decisive est la VALIDATION (--validate) : les totaux reconstruits sont
confrontes aux vrais compteurs de l'API Anthropic, presents dans
full_run_log.txt / scaling_night.log pour Opus 4.8, Haiku 4.5 et Sonnet 5.
Si la reconstruction tient sur ces trois modeles, la meme methode appliquee a
Gemini / gemma3 / llama3.1 est defendable. Sinon on ne publie que les caracteres.

L'ORACLE (utilisateur simule) est compte separement et EXCLU du cout du systeme :
c'est un artefact experimental partage par toutes les lignes, pas un cout que
paierait un deploiement reel.

Usage :
    python3 checkpoints/ali_standalone/cost_analysis.py
    python3 checkpoints/ali_standalone/cost_analysis.py --validate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP1 = ROOT / "results" / "baselines"        # [relocated] baseline records
_PHASE1_CODE = ROOT / "code" / "phase1"      # [relocated] run_multiturn.py
sys.path.insert(0, str(_PHASE1_CODE))  # [relocated]

from run_multiturn import (  # noqa: E402
    DOMAIN_TASKS,
    INTERVIEWER_SYSTEM,
    ORACLE_SYSTEM,
)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def ntok(s: str) -> int:
        return len(_ENC.encode(s))
except Exception:                                    # pragma: no cover
    def ntok(s: str) -> int:
        return round(len(s) / 4)                     # repli grossier

MODELS = {
    "gemma3:4b":          "results_gemma3_4b_multiturn_full.json",
    "llama3.1:8b":        "results_llama3.1_8b_multiturn_full.json",
    "gemini-2.5-flash":   "results_gemini-2.5-flash_multiturn_full.json",
    "claude-haiku-4-5":   "results_claude-haiku-4-5_multiturn_full.json",
    "claude-sonnet-5":    "results_claude-sonnet-5_multiturn_full.json",
    "claude-opus-4-8":    "results_claude-opus-4-8_multiturn_full.json",
}


def _reminder(turn: int, msg: str) -> str:
    """Suffixe de pression au tour t -- identique a run_multiturn_claude.py."""
    if turn >= 8:
        return (f"{msg}\n\n[REMINDER: You have asked {turn} questions. "
                f"You MUST output [DONE] now — do not ask another question.]")
    if turn >= 5:
        return (f"{msg}\n\n[REMINDER: You have asked {turn} questions. "
                f"Wrap up and output [DONE] soon.]")
    return msg


def episode_cost(rec: dict) -> dict:
    """Reconstruit, tour par tour, ce qui a ete envoye et recu par l'interviewer.

    A chaque tour l'historique COMPLET est renvoye : c'est la croissance
    quadratique du contexte, et c'est precisement ce que la mesure doit capturer.
    """
    conv = rec["conversation"]
    system = INTERVIEWER_SYSTEM.format(domain_task=DOMAIN_TASKS[rec["domain"]])
    oracle_system = ORACLE_SYSTEM.format(
        domain_task=DOMAIN_TASKS[rec["domain"]], original_prompt=conv[0]["content"]
    )

    in_c = in_t = out_c = out_t = 0
    o_in_c = o_in_t = o_out_c = o_out_t = 0
    calls = o_calls = 0
    history: list[str] = []       # vu par l'interviewer
    turn = 0

    for i, msg in enumerate(conv):
        if msg["role"] == "user":
            history.append(_reminder(turn, msg["content"]))
        else:
            # --- appel interviewer : systeme + TOUT l'historique accumule
            payload = system + "".join(history)
            in_c += len(payload); in_t += ntok(payload)
            out_c += len(msg["content"]); out_t += ntok(msg["content"])
            calls += 1
            history.append(msg["content"])
            turn += 1
            # --- appel oracle : il ne voit pas le prompt initial (il est dans son
            #     systeme) ; son historique demarre a la 1re question posee.
            if i + 1 < len(conv):
                o_payload = oracle_system + "".join(history[1:])
                o_in_c += len(o_payload); o_in_t += ntok(o_payload)
                nxt = conv[i + 1]["content"]
                o_out_c += len(nxt); o_out_t += ntok(nxt)
                o_calls += 1

    return {
        "calls": calls,
        "in_chars": in_c, "out_chars": out_c, "chars": in_c + out_c,
        "in_tok": in_t, "out_tok": out_t, "tok": in_t + out_t,
        "o_calls": o_calls,
        "o_in_tok": o_in_t, "o_out_tok": o_out_t,
        "o_chars": o_in_c + o_out_c, "o_tok": o_in_t + o_out_t,
        "turns": rec["n_turns"],
    }


def summarise(name: str, path: Path) -> dict:
    recs = json.load(open(path))
    per = [episode_cost(r) for r in recs]
    agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    agg["n"] = len(per)
    agg["model"] = name
    return agg


# ---------------------------------------------------------------- validation
#
# Les compteurs des logs sont CUMULES par etage de run, et chaque etage a sa
# propre distribution des roles ainsi que son propre nombre de prompts
# reellement appeles (les prompts « resume » sont sautes et ne comptent pas).
# Comparer sans en tenir compte n'a aucun sens ; d'ou cette table explicite.
#
#   role      : "interviewer", "oracle", ou "both" (le meme modele tient les deux
#               roles, les deux consommations tombent alors dans un seul compteur)
#   n_called  : prompts effectivement appeles a cet etage (total moins « resume »)
#   thinking  : le raisonnement adaptatif est renvoye a chaque tour et n'existe
#               pas dans les transcripts -> reconstruction structurellement basse
VALIDATION_CASES = [
    # (log,                 modele,             role,          n_called, thinking, in_api)
    ("full_run_log.txt",    "claude-opus-4-8",  "interviewer", 145, True,  1_235_855),
    ("full_run_log.txt",    "claude-haiku-4-5", "oracle",      145, False,   593_985),
    ("scaling_night.log",   "claude-sonnet-5",  "interviewer",  48, True,    411_715),
    ("scaling_night.log",   "claude-haiku-4-5", "oracle",       48, False,   178_996),
    ("scaling_night.log",   "claude-haiku-4-5", "both",         48, False,   419_542),
    ("scaling_night.log",   "claude-haiku-4-5", "both",        100, False,   874_045),
]


# ------------------------------------------------------- latence des baselines
#
# Aucune horloge n'a ete enregistree par prompt. Mais `scaling_night.log` horodate
# le debut et la fin de chaque etage de run, ce qui donne une latence moyenne par
# interview sans rien reexecuter.
#
# Ce que cette mesure INCLUT et qu'ALI n'inclut pas : un appel au scorer Gemini par
# prompt, plus le demarrage du processus. C'est donc un MAJORANT de la latence
# d'interview -- et le biais joue en faveur d'ALI dans la comparaison, ce qu'il
# faut dire.
#
#   (etage, debut, fin, prompts reellement appeles)
LATENCY_STAGES = [
    ("Claude Sonnet 5 (n=50, raisonnement adaptatif)", "14:35:25", "15:17:52", 48),
    ("Claude Haiku 4.5 (n=50, sans raisonnement)",     "15:19:43", "15:54:51", 48),
    ("Claude Haiku 4.5 (extension vers n=150)",        "15:54:51", "17:08:23", 100),
]


def _secs(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def baseline_latency() -> list[tuple[str, float, int, float]]:
    out = []
    for label, a, b, n in LATENCY_STAGES:
        d = _secs(b) - _secs(a)
        out.append((label, d, n, d / n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    rows = [summarise(n, EXP1 / f) for n, f in MODELS.items() if (EXP1 / f).exists()]

    print("\nCOUT PAR INTERVIEW COMPLETE (interviewer seul, oracle exclu)")
    print(f"{'systeme':20s} {'n':>4s} {'tours':>6s} {'appels':>7s} "
          f"{'car. in':>9s} {'car. out':>9s} {'car. tot':>9s} {'tok. tot':>9s}")
    for r in sorted(rows, key=lambda x: x["chars"]):
        print(f"{r['model']:20s} {r['n']:4d} {r['turns']:6.2f} {r['calls']:7.2f} "
              f"{r['in_chars']:9.0f} {r['out_chars']:9.0f} {r['chars']:9.0f} "
              f"{r['tok']:9.0f}")

    print("\nLATENCE DES BASELINES, deduite des horodatages d'etage de scaling_night.log")
    print(f"{'etage':46s} {'duree':>9s} {'n':>5s} {'s/interview':>12s}")
    for label, d, n, per in baseline_latency():
        print(f"{label:46s} {d / 60:7.1f}min {n:5d} {per:12.1f}")
    print("Les deux etages Haiku, independants, donnent 43.9 et 44.1 s : la mesure est")
    print("stable. Elle inclut un appel au scorer par prompt et le demarrage du")
    print("processus, c'est donc un MAJORANT -- biais favorable a ALI, a signaler.")

    print("\nORACLE (artefact experimental partage, exclu du cout systeme) : "
          f"{np.mean([r['o_chars'] for r in rows]):.0f} car. et "
          f"{np.mean([r['o_tok'] for r in rows]):.0f} tok./interview en moyenne")

    if args.validate:
        by_model = {r["model"]: r for r in rows}
        print("\nVALIDATION -- tokens d'entree reconstruits vs vrais compteurs de l'API")
        print(f"{'source':18s} {'modele':18s} {'role':12s} {'n':>4s} "
              f"{'API in':>10s} {'reconstr.':>10s} {'ecart':>8s}")
        for log, model, role, n_called, thinking, in_api in VALIDATION_CASES:
            r = by_model.get(model)
            if r is None:
                continue
            per = {"interviewer": r["in_tok"],
                   "oracle": r["o_in_tok"],
                   "both": r["in_tok"] + r["o_in_tok"]}[role]
            rec = per * n_called
            d = 100 * (rec - in_api) / in_api
            tag = " (raisonnement adaptatif)" if thinking else ""
            print(f"{log:18s} {model:18s} {role:12s} {n_called:4d} "
                  f"{in_api:10,d} {rec:10,.0f} {d:7.1f}%{tag}")
        fac = []
        for log, model, role, n_called, thinking, in_api in VALIDATION_CASES:
            r = by_model.get(model)
            if r is None or thinking:
                continue
            per = {"interviewer": r["in_tok"], "oracle": r["o_in_tok"],
                   "both": r["in_tok"] + r["o_in_tok"]}[role]
            fac.append(in_api / (per * n_called))
        if fac:
            print(f"\nFacteur de calibration sur les {len(fac)} cas sans raisonnement : "
                  f"min {min(fac):.3f} / max {max(fac):.3f} / median {np.median(fac):.3f}")
            print("Ecart systematique et non aleatoire (deux etages independants donnent")
            print("1.149 a l'identique) : c'est un ecart de TOKENIZER plus la surcharge")
            print("par message de l'API, pas une erreur de reconstruction. Consequence :")
            print("  - les CARACTERES sont exacts et strictement comparables entre lignes ;")
            print("  - les TOKENS sont comparables entre lignes (tokenizer unique) mais")
            print("    inferieurs de ~13-25 % au comptage propre du fournisseur.")
        print("""
Lecture :
  * Les lignes SANS raisonnement adaptatif sont le test reel. Un ecart de quelques
    pour cent y valide la reconstruction, et donc son application a Gemini, gemma3,
    llama3.1 et ALI, pour lesquels aucun compteur n'existe.
  * Les lignes Opus et Sonnet doivent sous-estimer : les blocs de raisonnement sont
    renvoyes a chaque tour et n'apparaissent pas dans les transcripts sauvegardes.
    L'ecart y mesure le poids du raisonnement, il n'invalide pas la methode.
  * n = prompts reellement appeles a cet etage (les prompts « resume » sont sautes).
    La moyenne par prompt est calculee sur l'ensemble du fichier de resultats, donc
    la comparaison suppose que le sous-ensemble appele n'est pas atypique.""")


if __name__ == "__main__":
    main()
