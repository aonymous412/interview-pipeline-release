"""
Phase 2 - Section 6 : Ablation empirique des composants ALI.

Sur un sous-ensemble de 30 prompts (10/domaine, equilibre en vagueness),
compare la config ALI complete a 4 variantes qui desactivent ou degradent un
mecanisme structurel a la fois. Objectif : confirmer empiriquement l'ablation
qui etait simulation-derived dans la v8 (tab:ablation).

Operationnalisation des 4 ablations (le pipeline reel ali_standalone/ n'a pas
de hyperparametre "delta" ou "bonus early-turn" explicite -- ce concept vient
de la simulation v8. On l'operationnalise ici de la facon la plus fidele
possible a l'intention de chaque ablation) :

  1. no_c2            : C2 desactive -- chaque parametre manquant devient son
                        propre cluster (1 param/tour) au lieu du regroupement
                        K-Means. Teste si le clustering ameliore l'efficacite
                        (turns) a couverture finale egale.
  2. no_priority       : pas de tri par poids -- C1 retourne les parametres
                        manquants dans l'ordre d'insertion de la taxonomie
                        (ordre arbitraire), au lieu de prioriser les parametres
                        a poids eleve. Operationnalise "pas de bonus pour
                        resoudre tot les parametres critiques" (delta=0).
  3. c1_random         : C1 retourne les parametres manquants dans un ordre
                        aleatoire (differe de no_priority : re-tire a chaque
                        tour, pas un ordre fixe).
  4. c3_tier1_only     : C3 force toujours le tier "clarification", meme si le
                        parametre a deja ete mentionne ou partiellement
                        resolu (desactive boundary/confirmation).

Usage : python3 checkpoints/ali_standalone/ablation.py --n 30
(depuis la racine du repo, apres avoir configure .env)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [relocated] code/ on sys.path
from ali import config  # noqa: E402
from ali.ali_pipeline import ALIPipeline  # noqa: E402
from ali.user_simulator import GeminiUserSimulator  # noqa: E402
from ali.run_phase2 import (  # noqa: E402
    DOMAIN_TO_DEPLOYMENT, load_prompt_texts,
)

ABLATIONS = ["full", "no_c2", "no_priority", "c1_random", "c3_tier1_only"]


def _unresolved(state) -> list[dict]:
    return [
        {"name": name, **p} for name, p in state.taxonomy["parameters"].items()
        if name not in state.F
    ]


def apply_ablation(pipeline: ALIPipeline, ablation: str) -> None:
    if ablation == "full":
        return

    if ablation == "no_c2":
        pipeline.c2.cluster = lambda params: [[p] for p in params]

    elif ablation == "no_priority":
        def identify_insertion_order(state):
            return _unresolved(state)
        pipeline.c1.identify = identify_insertion_order

    elif ablation == "c1_random":
        def identify_random(state):
            u = _unresolved(state)
            random.shuffle(u)
            return u
        pipeline.c1.identify = identify_random

    elif ablation == "c3_tier1_only":
        pipeline.c3._select_tier = lambda param, state: "clarification"

    else:
        raise ValueError(f"ablation inconnue: {ablation}")


def sample_30(prompts_df: pd.DataFrame, n_per_domain: int = 10) -> pd.DataFrame:
    parts = []
    for domain, group in prompts_df.groupby("domain"):
        parts.append(group.sample(n=min(n_per_domain, len(group)), random_state=42))
    return pd.concat(parts).reset_index(drop=True)


def run_ablation(ablation: str, prompts_df: pd.DataFrame, out_path: Path) -> list[dict]:
    results: list[dict] = []
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done_ids = {r["prompt_id"] for r in results}

    for _, row in prompts_df.iterrows():
        if row["prompt_id"] in done_ids:
            continue
        deployment = DOMAIN_TO_DEPLOYMENT[row["domain"]]
        try:
            pipeline = ALIPipeline(deployment=deployment, c1_mode="finetuned", c4_mode="gemini")
            apply_ablation(pipeline, ablation)
            simulator = GeminiUserSimulator(initial_prompt=row["prompt"])
            result = pipeline.run(row["prompt"], simulator, prompt_id=row["prompt_id"])
        except Exception as e:
            print(f"  [ERREUR] {ablation}/{row['prompt_id']}: {e} -- saute")
            continue
        result["ablation"] = ablation
        result["domain"] = row["domain"]
        result["vagueness"] = row["vagueness"]
        results.append(result)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [{ablation}] {row['prompt_id']} cov={result['final_cov']:.3f} "
              f"turns={result['n_turns']} ({len(results)} sauvegardes)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="prompts par domaine (30 total = 10x3)")
    parser.add_argument("--ablation", default="all", help="|".join(ABLATIONS) + " | all")
    args = parser.parse_args()

    prompts_df = pd.read_csv(config.BASELINE_CSV, skipinitialspace=True)
    texts = load_prompt_texts()
    prompts_df["prompt"] = prompts_df["prompt_id"].map(texts)
    sample = sample_30(prompts_df, n_per_domain=args.n)

    ablations_to_run = ABLATIONS if args.ablation == "all" else [args.ablation]
    out_dir = config.RECOMPUTED_DIR  # [relocated]
    summary = {}
    for ablation in ablations_to_run:
        print(f"--- Ablation: {ablation} ---")
        out_path = out_dir / f"results_ablation_{ablation}.json"
        results = run_ablation(ablation, sample, out_path)
        if results:
            mean_cov = sum(r["final_cov"] for r in results) / len(results)
            mean_turns = sum(r["n_turns"] for r in results) / len(results)
            summary[ablation] = {
                "n": len(results), "mean_cov": round(mean_cov, 4), "mean_turns": round(mean_turns, 2),
            }

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
