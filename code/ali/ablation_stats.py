"""
T1 -- Analyse statistique de l'ablation empirique (n=30 apparie par variante).

Complement de ablation.py : la ou ablation_summary.json ne contient que des
moyennes, ce script produit tout ce que la phrase "aucune variante ne se
distingue significativement" de section 5.6 exige pour etre testee :

  1. Descriptif par variante : n, moyenne, ecart-type, IC 95% bootstrap
     (meme methode que evaluate.py : percentile, 10 000 resamples, seed 42).
  2. Tests apparies full vs chaque variante : Wilcoxon signed-rank sur les
     differences appariees (pas d'hypothese de normalite sur des couvertures
     bornees a [0,1]), + difference moyenne appariee et son IC bootstrap.
  3. Correction Holm-Bonferroni sur les 4 comparaisons, par metrique.
  4. Test d'equivalence (TOST) non parametrique : deux Wilcoxon unilateraux
     sur les differences decalees de +/-Delta. Marges fixees AVANT analyse
     (cf. ablation_n30_report.md) : Delta = 2 points de couverture, 1 tour.
  5. Puissance : plus petit effet detectable (test t apparie, alpha=0.05,
     puissance 0.8) a partir du sd observe des differences -- pour la section
     Limitations.

Aucun appel reseau ; deterministe (seed 42).

Usage :
    python3 checkpoints/ali_standalone/ablation_stats.py           # analyse
    python3 checkpoints/ali_standalone/ablation_stats.py --check   # sanite seule

Sorties : ablation_stats.json + ablation_table.tex (tableau pret a coller).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parents[2] / "results" / "ali"       # [relocated] reads
OUT = Path(__file__).resolve().parents[2] / "recomputed"             # [relocated] writes
OUT.mkdir(exist_ok=True)
ABLATIONS = ["full", "no_c2", "no_priority", "c1_random", "c3_tier1_only"]
METRICS = ["final_cov", "n_turns"]
# Marges d'equivalence TOST, fixees avant analyse (ablation_n30_report.md) :
TOST_DELTA = {"final_cov": 0.02, "n_turns": 1.0}
ALPHA = 0.05
N_BOOTSTRAP = 10_000
SEED = 42

LABELS = {
    "full": "ALI complet",
    "no_c2": "sans C2 (pas de clustering)",
    "no_priority": "sans priorisation C1",
    "c1_random": "C1 ordre aleatoire",
    "c3_tier1_only": "C3 clarification seule",
}


def load_paired() -> tuple[dict[str, dict[str, dict]], list[str]]:
    """Charge les 5 fichiers, indexes par prompt_id, et le jeu apparie commun."""
    data: dict[str, dict[str, dict]] = {}
    for ab in ABLATIONS:
        path = HERE / f"results_ablation_{ab}.json"
        with open(path) as f:
            results = json.load(f)
        by_id = {r["prompt_id"]: r for r in results}
        if len(by_id) != len(results):
            raise ValueError(f"{path.name}: prompt_id dupliques")
        data[ab] = by_id
    common = set.intersection(*(set(d) for d in data.values()))
    return data, sorted(common)


def sanity_check(data: dict[str, dict[str, dict]], common: list[str]) -> bool:
    """Verifie l'appariement parfait et l'equilibre par domaine. True si OK."""
    ok = True
    for ab in ABLATIONS:
        ids = set(data[ab])
        extra, missing = ids - set(common), set(common) - ids
        domains: dict[str, int] = {}
        for pid in ids:
            dom = data[ab][pid]["domain"]
            domains[dom] = domains.get(dom, 0) + 1
        print(f"  {ab:15s} n={len(ids):3d}  domaines={dict(sorted(domains.items()))}")
        if extra or missing:
            ok = False
            if missing:
                print(f"    !! absents vs jeu commun: {sorted(missing)}")
    print(f"  jeu commun apparie : n={len(common)}")
    if len({len(data[ab]) for ab in ABLATIONS}) != 1:
        ok = False
        print("  !! les variantes n'ont pas toutes le meme nombre d'entrees "
              "-- relancer ablation.py jusqu'a appariement parfait")
    return ok


def bootstrap_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    """IC 95% bootstrap percentile -- meme methode que evaluate.py."""
    rng = np.random.default_rng(seed)
    boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(round(lo, 4)), float(round(hi, 4))


def wilcoxon_safe(diffs: np.ndarray, alternative: str = "two-sided") -> float:
    """Wilcoxon signed-rank ; p=1.0 si toutes les differences sont nulles."""
    if np.allclose(diffs, 0):
        return 1.0
    return float(stats.wilcoxon(diffs, alternative=alternative).pvalue)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """P-values ajustees Holm-Bonferroni (step-down), ordre d'entree preserve."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return [float(round(p, 6)) for p in adjusted]


def tost(diffs: np.ndarray, delta: float) -> dict:
    """TOST non parametrique : H0 |effet| >= delta rejetee si p < alpha.

    p_lower : Wilcoxon unilateral sur (d + delta) > 0  (effet > -delta)
    p_upper : Wilcoxon unilateral sur (d - delta) < 0  (effet < +delta)
    """
    p_lower = wilcoxon_safe(diffs + delta, alternative="greater")
    p_upper = wilcoxon_safe(diffs - delta, alternative="less")
    p = max(p_lower, p_upper)
    return {
        "delta": delta,
        "p_lower": round(p_lower, 6),
        "p_upper": round(p_upper, 6),
        "p_tost": round(p, 6),
        "equivalent": bool(p < ALPHA),
    }


def min_detectable_effect(sd_diff: float, n: int, alpha: float = ALPHA, power: float = 0.8) -> float:
    """Plus petit effet detectable par un test t apparie bilateral (bisection
    sur la puissance exacte via la loi t non centrale)."""
    if sd_diff == 0 or n < 2:
        return 0.0
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    def power_at(effect: float) -> float:
        nc = effect / (sd_diff / np.sqrt(n))
        return 1 - stats.nct.cdf(t_crit, df, nc) + stats.nct.cdf(-t_crit, df, nc)

    lo, hi = 0.0, 10 * sd_diff
    for _ in range(200):
        mid = (lo + hi) / 2
        if power_at(mid) < power:
            lo = mid
        else:
            hi = mid
    return float(round(hi, 6))


def analyze(data: dict[str, dict[str, dict]], common: list[str]) -> dict:
    out: dict = {
        "n_paired": len(common),
        "prompt_ids": common,
        "alpha": ALPHA,
        "tost_delta": TOST_DELTA,
        "methods": {
            "ci": f"bootstrap percentile, {N_BOOTSTRAP} resamples, seed {SEED} (idem evaluate.py)",
            "paired_test": "Wilcoxon signed-rank, bilateral, full vs variante",
            "multiplicity": "Holm-Bonferroni sur les 4 comparaisons, par metrique",
            "equivalence": "TOST non parametrique (2 Wilcoxon unilateraux decales de +/-delta)",
            "power": "test t apparie bilateral, alpha=0.05, puissance 0.8, loi t non centrale",
        },
        "descriptive": {},
        "paired_vs_full": {},
    }

    values = {
        (ab, m): np.array([data[ab][pid][m] for pid in common], dtype=float)
        for ab in ABLATIONS for m in METRICS
    }

    for ab in ABLATIONS:
        out["descriptive"][ab] = {}
        for m in METRICS:
            v = values[(ab, m)]
            out["descriptive"][ab][m] = {
                "n": len(v),
                "mean": round(float(v.mean()), 4),
                "std": round(float(v.std(ddof=1)), 4),
                "ci95": bootstrap_ci(v),
            }

    variants = [ab for ab in ABLATIONS if ab != "full"]
    for m in METRICS:
        raw_p: list[float] = []
        rows: dict[str, dict] = {}
        for ab in variants:
            diffs = values[(ab, m)] - values[("full", m)]
            p = wilcoxon_safe(diffs)
            raw_p.append(p)
            rows[ab] = {
                "mean_diff": round(float(diffs.mean()), 4),
                "diff_ci95": bootstrap_ci(diffs),
                "sd_diff": round(float(diffs.std(ddof=1)), 4),
                "wilcoxon_p": round(p, 6),
                "tost": tost(diffs, TOST_DELTA[m]),
                "min_detectable_effect": min_detectable_effect(float(diffs.std(ddof=1)), len(diffs)),
            }
        for ab, p_adj in zip(variants, holm_bonferroni(raw_p)):
            rows[ab]["wilcoxon_p_holm"] = p_adj
            rows[ab]["significant_holm"] = bool(p_adj < ALPHA)
        out["paired_vs_full"][m] = rows

    return out


def latex_table(res: dict) -> str:
    """Tableau pret a coller : Variante | n | Cov [IC] | Delta vs full | p | Tours [IC]."""
    lines = [
        f"% Genere par ablation_stats.py -- ablation appariee, n={res['n_paired']} prompts communs",
        "% p : Wilcoxon signed-rank apparie vs full, ajuste Holm-Bonferroni (4 comparaisons)",
        "% TOST : equivalence dans +/-2 pts de couverture (dague = demontree, p<0.05)",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Variante & $n$ & Cov.\\ [IC 95\\%] & $\\Delta$ vs full & $p$ & Tours [IC 95\\%] \\\\",
        "\\midrule",
    ]
    for ab in ABLATIONS:
        d_cov = res["descriptive"][ab]["final_cov"]
        d_turns = res["descriptive"][ab]["n_turns"]
        cov = f"{d_cov['mean']*100:.1f}\\% [{d_cov['ci95'][0]*100:.1f}--{d_cov['ci95'][1]*100:.1f}]"
        turns = f"{d_turns['mean']:.2f} [{d_turns['ci95'][0]:.2f}--{d_turns['ci95'][1]:.2f}]"
        if ab == "full":
            delta, p = "--", "--"
        else:
            row = res["paired_vs_full"]["final_cov"][ab]
            sign = "+" if row["mean_diff"] >= 0 else ""
            delta = f"{sign}{row['mean_diff']*100:.1f}"
            if row["tost"]["equivalent"]:
                delta += "$^\\dagger$"
            p = f"{row['wilcoxon_p_holm']:.3f}"
        lines.append(f"{LABELS[ab]} & {d_cov['n']} & {cov} & {delta} & {p} & {turns} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="controles de sanite seulement")
    args = parser.parse_args()

    data, common = load_paired()
    print("=== Controles de sanite ===")
    ok = sanity_check(data, common)
    if args.check:
        sys.exit(0 if ok else 1)
    if not ok:
        print("!! Appariement imparfait -- analyse sur le jeu commun uniquement.")

    res = analyze(data, common)

    print("\n=== Descriptif (couverture) ===")
    for ab in ABLATIONS:
        d = res["descriptive"][ab]["final_cov"]
        t = res["descriptive"][ab]["n_turns"]
        print(f"  {ab:15s} cov {d['mean']*100:5.1f}% +/-{d['std']*100:4.1f} "
              f"[{d['ci95'][0]*100:.1f}-{d['ci95'][1]*100:.1f}]  "
              f"tours {t['mean']:.2f} [{t['ci95'][0]:.2f}-{t['ci95'][1]:.2f}]")

    for m in METRICS:
        print(f"\n=== full vs variantes ({m}) ===")
        for ab, row in res["paired_vs_full"][m].items():
            eq = "EQUIVALENT" if row["tost"]["equivalent"] else "non demontre"
            print(f"  {ab:15s} diff={row['mean_diff']:+.4f} "
                  f"[{row['diff_ci95'][0]:+.4f},{row['diff_ci95'][1]:+.4f}] "
                  f"p={row['wilcoxon_p']:.4f} (Holm {row['wilcoxon_p_holm']:.4f}) "
                  f"TOST(+/-{row['tost']['delta']}): p={row['tost']['p_tost']:.4f} {eq} "
                  f"MDE={row['min_detectable_effect']:.4f}")

    with open(OUT / "ablation_stats.json", "w") as f:  # [relocated]
        json.dump(res, f, indent=2, ensure_ascii=False)
    table = latex_table(res)
    with open(OUT / "ablation_table.tex", "w") as f:  # [relocated]
        f.write(table + "\n")
    print("\n=== Tableau LaTeX (ablation_table.tex) ===\n")
    print(table)
    print("\nEcrit : ablation_stats.json, ablation_table.tex")


if __name__ == "__main__":
    main()
