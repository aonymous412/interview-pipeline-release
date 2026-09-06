"""
Section 5 (evaluate.py) + Section 6 (ablation) -- bootstrap CI, tables LaTeX,
et point de decision critique (ALI_v1 <= 52.8% => arreter et diagnostiquer).

A executer APRES run_phase2.py (necessite results_ALI_v{1,2,3}.json).
Ne fait aucun appel reseau -- executable dans n'importe quel environnement
une fois les fichiers de resultats presents.

Usage :
    cd ali_standalone && python3 evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASELINE_ROW = {
    "Single-pass (n=150)": {"mean": 0.407, "std": 0.201, "ci": (0.375, 0.439), "turns": 1, "tokens": 1785},
    "Gemini multi-tour (n=50)": {"mean": 0.528, "std": 0.158, "ci": (0.494, 0.562), "turns": 7.0, "tokens": 8400},
    "Llama multi-tour (n=50)": {"mean": 0.475, "std": 0.173, "ci": (0.447, 0.503), "turns": 5.9, "tokens": 7080},
}
DECISION_THRESHOLD = 0.528  # meilleur LLM multi-tour (Gemini)


def bootstrap_ci(scores, n_bootstrap=10000, alpha=0.05, seed=42):
    if len(scores) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    boot_means = [rng.choice(scores, len(scores), replace=True).mean() for _ in range(n_bootstrap)]
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(lo, 4), round(hi, 4)


def summarize_config(results: list[dict]) -> dict:
    covs = [r["final_cov"] for r in results]
    turns = [r["n_turns"] for r in results]
    return {
        "n": len(results),
        "mean_cov": round(float(np.mean(covs)), 4),
        "std_cov": round(float(np.std(covs)), 4),
        "ci": bootstrap_ci(covs),
        "mean_turns": round(float(np.mean(turns)), 2),
        "formal_stop": True,
    }


def latex_row(name: str, stats: dict) -> str:
    ci_lo, ci_hi = stats["ci"] if isinstance(stats.get("ci"), tuple) else stats["ci"]
    return (
        f"{name} & {stats['mean_cov']*100:.1f}\\% & {stats['std_cov']*100:.1f}\\% & "
        f"[{ci_lo*100:.1f}-{ci_hi*100:.1f}\\%] & {stats['mean_turns']} & -- & Oui (cov$\\geq\\theta$) \\\\"
    )


def main():
    here = Path(__file__).resolve().parents[2] / "results" / "ali"  # [relocated]
    summary = {}
    for cfg_name in ["ALI_v1", "ALI_v2", "ALI_v3"]:
        path = here / f"results_{cfg_name}.json"
        if not path.exists():
            print(f"[SKIP] {path} introuvable -- executer run_phase2.py d'abord.")
            continue
        with open(path) as f:
            results = json.load(f)
        summary[cfg_name] = summarize_config(results)

    if not summary:
        print("Aucun resultat trouve. Ce script doit tourner APRES run_phase2.py "
              "(qui necessite un acces reseau a Gemini + HuggingFace, indisponible "
              "dans ce sandbox). Voir HANDOFF.md.")
        return

    print("=== Tableau tab:multiturn etendu ===\n")
    print("System & Mean cov & Std & 95% CI & Turns & Tokens & Arret formel \\\\")
    for name, s in BASELINE_ROW.items():
        print(f"{name} & {s['mean']*100:.1f}\\% & {s['std']*100:.1f}\\% & "
              f"[{s['ci'][0]*100:.1f}-{s['ci'][1]*100:.1f}\\%] & {s['turns']} & {s['tokens']} & Non \\\\")
    for cfg_name, stats in summary.items():
        print(latex_row(cfg_name, stats))

    if "ALI_v1" in summary:
        v1_mean = summary["ALI_v1"]["mean_cov"]
        print(f"\n=== Point de decision critique ===")
        print(f"ALI_v1 mean_cov = {v1_mean*100:.1f}% vs seuil {DECISION_THRESHOLD*100:.1f}% "
              f"(meilleur LLM multi-tour, Gemini)")
        if v1_mean <= DECISION_THRESHOLD:
            print("!! ALI_v1 <= baseline -- ARRETER et diagnostiquer avant de continuer "
                  "(cf. prompt Phase 2, Section 9).")
        else:
            print("OK -- ALI_v1 > baseline, la these centrale est supportee empiriquement.")

    out_dir = Path(__file__).resolve().parents[2] / "recomputed"  # [relocated]
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "evaluation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
