"""
H1' sanity check (Track B, Théorème 3) — corrélations inter-clusters d'ALI.

Le Théorème 3 (voir Knowledge Base/Experiments/Phase 3 - Volets/2 Theoretical
Formalization.md) n'établit la borne gloutonne (1-1/e) que sous l'hypothèse H1' :
INDÉPENDANCE INTER-CLUSTERS (les corrélations intra-cluster restent autorisées).
H1' n'est pas vérifiée empiriquement dans le papier — elle est posée.

Ce script fournit un CONTRÔLE DESCRIPTIF, PAS une preuve : les résolutions
observées sont confondues par la politique d'interview d'ALI (l'ordre des
questions dépend de la couverture déjà atteinte). On peut donc seulement dire
"cohérent avec H1'" si les corrélations inter-clusters sont faibles, jamais
"H1' est vraie".

Méthode :
  1. Reconstituer les clusters EXACTEMENT comme C2 (KMeans déterministe,
     random_state=42, sur les embeddings all-MiniLM-L6-v2 des descriptions) —
     un clustering par domaine, identique pour tous les épisodes du domaine.
  2. Depuis results_ALI_v1.json, matrice binaire (épisode × paramètre) : 1 si le
     paramètre est dans F_resolved, sinon 0.
  3. Corrélation phi (Pearson sur binaire) pour chaque paire de paramètres, sur
     les épisodes du domaine.
  4. Comparer la moyenne des |corr| INTRA-cluster vs INTER-cluster.

Usage : python checkpoints/ali_standalone/h1_sanity_check.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# stdout Windows en cp1252 par défaut -> forcer UTF-8 pour les accents / flèches
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

B_PERM = 5000        # itérations de permutation
RNG = np.random.default_rng(42)

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results" / "ali" / "results_ALI_v1.json"      # [relocated]
TAXONOMIES = REPO / "data" / "taxonomies.json"                  # [relocated]
_OUT = REPO / "recomputed"; _OUT.mkdir(exist_ok=True)           # [relocated]
OUT_JSON = _OUT / "h1_sanity_check_report.json"
OUT_PNG = _OUT / "h1_sanity_check_heatmap.png"                  # [relocated]

# domaine (taxonomies.json) -> deployment (results_ALI_v1.json)
DOMAINS = ["Consulting", "Medical", "Payments"]


def reconstruct_clusters(params: list[dict]) -> list[list[str]]:
    """Rejoue C2Clusterer.cluster() à l'identique et renvoie les noms par cluster."""
    from components.c2_clusterer import C2Clusterer  # import tardif (torch lourd)

    clusterer = C2Clusterer()
    clusters = clusterer.cluster(params)
    return [[p["name"] for p in c] for c in clusters]


def cluster_of(name: str, clusters: list[list[str]]) -> int:
    for i, c in enumerate(clusters):
        if name in c:
            return i
    return -1


def mean_abs_corr(M: np.ndarray, pairs: list[tuple[int, int]]) -> float | None:
    """|corr| moyen sur une liste de paires de colonnes (NaN ignorés)."""
    vals = []
    for i, j in pairs:
        r = np.corrcoef(M[:, i], M[:, j])[0, 1]
        if not np.isnan(r):
            vals.append(abs(r))
    return float(np.mean(vals)) if vals else None


def residualize(M: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Retire l'effet linéaire des covariables Z de chaque colonne de M
    (résidus des moindres carrés) — base de la corrélation partielle."""
    beta, *_ = np.linalg.lstsq(Z, M, rcond=None)
    return M - Z @ beta


_corr_matrices: dict = {}


def make_heatmaps(mats: dict) -> None:
    """Une figure, un sous-graphe par domaine : matrice de corrélation des
    paramètres actifs, réordonnés par cluster (blocs intra-cluster visibles)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(mats), figsize=(6 * len(mats), 5.5))
    if len(mats) == 1:
        axes = [axes]
    for ax, (domain, (M, active, names, clusters)) in zip(axes, mats.items()):
        order = sorted(active, key=lambda i: (cluster_of(names[i], clusters), i))
        sub = M[:, order]
        C = np.corrcoef(sub, rowvar=False)
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
        labels = [f"{names[i][:16]} [c{cluster_of(names[i], clusters)}]" for i in order]
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"{domain} (params actifs, ordonnés par cluster)", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.6, label="corrélation")
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")


def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))  # pour 'components'

    taxo = json.loads(TAXONOMIES.read_text(encoding="utf-8"))
    episodes = json.loads(RESULTS.read_text(encoding="utf-8"))

    report: dict = {"note": "Contrôle descriptif — ne prouve PAS H1' (confondu "
                            "par la politique d'interview). Voir docstring.",
                    "domains": {}}

    for domain in DOMAINS:
        params_dict = taxo[domain]["parameters"]
        params = [{"name": k, "weight": v["weight"], "description": v["description"]}
                  for k, v in params_dict.items()]
        names = [p["name"] for p in params]

        clusters = reconstruct_clusters(params)

        # épisodes de ce domaine
        dom_eps = [e for e in episodes if e.get("domain") == domain]
        # matrice binaire résolu (n_episodes × n_params)
        M = np.array([[1 if n in e.get("F_resolved", {}) else 0 for n in names]
                      for e in dom_eps], dtype=float)

        # paramètres à variance nulle (toujours/jamais résolus) : corr indéfinie
        variances = M.var(axis=0)
        active = [i for i in range(len(names)) if variances[i] > 0]

        # paires inter-clusters (indices dans `active`) — c'est sur elles que porte H1'
        inter_pairs = [(i, j) for i, j in combinations(active, 2)
                       if cluster_of(names[i], clusters) != cluster_of(names[j], clusters)]
        intra_pairs = [(i, j) for i, j in combinations(active, 2)
                       if cluster_of(names[i], clusters) == cluster_of(names[j], clusters)]

        obs_inter = mean_abs_corr(M, inter_pairs)
        obs_intra = mean_abs_corr(M, intra_pairs)

        # --- TEST DE PERMUTATION (brut) ---
        # H0 (H1' au niveau inter-cluster) : les colonnes sont indépendantes.
        # On brise toute corrélation en permutant CHAQUE colonne indépendamment
        # (marginales de résolution préservées), et on recalcule le |corr| moyen
        # inter-clusters. La distribution obtenue = "plancher de hasard" à n fixé.
        null_inter = np.empty(B_PERM)
        for b in range(B_PERM):
            Mp = np.empty_like(M)
            for c in range(M.shape[1]):
                Mp[:, c] = RNG.permutation(M[:, c])
            null_inter[b] = mean_abs_corr(Mp, inter_pairs)
        # p-value unilatérale : proba que le hasard fasse AU MOINS aussi fort
        p_inter = (np.sum(null_inter >= obs_inter) + 1) / (B_PERM + 1)
        null_mean = float(np.mean(null_inter))

        # --- VERSION CONTRÔLÉE PAR LA DIFFICULTÉ (vaguesse) ---
        # Confondeur : un prompt vague laisse des params non résolus dans TOUS les
        # clusters -> corrélation positive artificielle partout. On teste donc
        # l'indépendance CONDITIONNELLE à la vaguesse.
        #   - statistique : |corr partielle| moyen inter-clusters (résidus ~ vaguesse)
        #   - null STRATIFIÉ : permuter chaque colonne DANS chaque niveau de vaguesse
        #     (préserve l'effet difficulté, casse les liens param-param résiduels)
        vag = [e.get("vagueness") for e in dom_eps]
        levels = sorted(set(vag))
        Z = np.column_stack([np.ones(len(vag))] +
                            [np.array([1.0 if v == lv else 0.0 for v in vag])
                             for lv in levels[1:]])          # intercept + dummies
        strata = {lv: [r for r in range(len(vag)) if vag[r] == lv] for lv in levels}

        Mr = residualize(M, Z)                               # résidus ~ vaguesse
        obs_inter_pc = mean_abs_corr(Mr, inter_pairs)
        obs_intra_pc = mean_abs_corr(Mr, intra_pairs)

        null_inter_pc = np.empty(B_PERM)
        for b in range(B_PERM):
            Mp = M.copy()
            for c in range(M.shape[1]):
                for lv in levels:                            # permutation intra-strate
                    rows = strata[lv]
                    Mp[rows, c] = RNG.permutation(Mp[rows, c])
            null_inter_pc[b] = mean_abs_corr(residualize(Mp, Z), inter_pairs)
        p_inter_pc = (np.sum(null_inter_pc >= obs_inter_pc) + 1) / (B_PERM + 1)
        null_mean_pc = float(np.mean(null_inter_pc))

        report["domains"][domain] = {
            "n_episodes": len(dom_eps),
            "n_params": len(names),
            "n_params_active": len(active),
            "n_params_constant": len(names) - len(active),
            "n_clusters": len(clusters),
            "clusters": clusters,
            "mean_abs_corr_intra": round(obs_intra, 4) if obs_intra is not None else None,
            "mean_abs_corr_inter": round(obs_inter, 4) if obs_inter is not None else None,
            "n_pairs_intra": len(intra_pairs),
            "n_pairs_inter": len(inter_pairs),
            "permutation_test_brut": {
                "null_mean_abs_corr_inter": round(null_mean, 4),
                "p_value_inter_gt_chance": round(float(p_inter), 4),
                "B": B_PERM,
                "interpretation": ("inter-clusters INDISTINGUABLE du hasard (p>=0.05) "
                                   "→ cohérent avec H1'" if p_inter >= 0.05 else
                                   "inter-clusters AU-DESSUS du hasard (p<0.05) "
                                   "→ tension avec H1'"),
            },
            "controle_vaguesse": {
                "mean_abs_partial_corr_intra": round(obs_intra_pc, 4) if obs_intra_pc else None,
                "mean_abs_partial_corr_inter": round(obs_inter_pc, 4) if obs_inter_pc else None,
                "null_mean_abs_partial_corr_inter": round(null_mean_pc, 4),
                "p_value_inter_gt_chance": round(float(p_inter_pc), 4),
                "interpretation": ("inter-clusters INDISTINGUABLE du hasard une fois la "
                                   "difficulté retirée (p>=0.05) → cohérent avec H1'"
                                   if p_inter_pc >= 0.05 else
                                   "inter-clusters AU-DESSUS du hasard même en contrôlant "
                                   "la difficulté (p<0.05) → tension persiste"),
            },
        }
        di = report["domains"][domain]
        print(f"\n=== {domain} (n={len(dom_eps)} épisodes, {len(clusters)} clusters, "
              f"{len(active)}/{len(names)} params à variance non nulle) ===")
        print(f"  BRUT    : |corr| inter {di['mean_abs_corr_inter']} vs hasard "
              f"{round(null_mean, 4)}  →  p={round(float(p_inter), 4)}")
        print(f"  CONTRÔLÉ (vaguesse retirée) : |corr partielle| inter "
              f"{round(obs_inter_pc, 4) if obs_inter_pc else None} vs hasard "
              f"{round(null_mean_pc, 4)}  →  p={round(float(p_inter_pc), 4)}")
        print(f"            {di['controle_vaguesse']['interpretation']}")

        _corr_matrices[domain] = (M, active, names, clusters)

    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport écrit : {OUT_JSON}")

    make_heatmaps(_corr_matrices)
    print(f"Heatmap écrite : {OUT_PNG}")

    # verdict global (descriptif)
    ps_brut = [d["permutation_test_brut"]["p_value_inter_gt_chance"]
               for d in report["domains"].values()]
    ps_ctrl = [d["controle_vaguesse"]["p_value_inter_gt_chance"]
               for d in report["domains"].values()]
    coh_brut = sum(1 for p in ps_brut if p >= 0.05)
    coh_ctrl = sum(1 for p in ps_ctrl if p >= 0.05)
    print(f"\nVerdict descriptif :")
    print(f"  BRUT     : {coh_brut}/3 domaines cohérents avec H1' (inter = hasard)")
    print(f"  CONTRÔLÉ : {coh_ctrl}/3 domaines cohérents avec H1' une fois la difficulté retirée")
    print("Rappel : contrôle descriptif, encore confondu par la politique d'interview "
          "(non corrigeable ici) — jamais une preuve de H1'.")


if __name__ == "__main__":
    main()
