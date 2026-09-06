"""
cov(F) = sum(w_d(p) for p in RESOLVED) / sum(w_d(p) for p in P_d)

Toujours calcule contre la taxonomie PLATE (taxonomies.json), pour rester
comparable a tab:multiturn (Phase 1). Voir taxonomy_mapping.py pour la
justification de ce choix et ses limites.
"""
from __future__ import annotations

import json

from . import config


def load_taxonomies() -> dict:
    with open(config.TAXONOMIES_JSON) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_taxonomy(domain_key: str) -> dict:
    """domain_key: 'Consulting' | 'Medical' | 'Payments' (cle taxonomies.json)."""
    taxos = load_taxonomies()
    if domain_key not in taxos:
        raise KeyError(f"Domaine inconnu dans taxonomies.json: {domain_key}")
    return taxos[domain_key]


class CoverageScorer:
    def __init__(self, taxonomy: dict):
        self.taxonomy = taxonomy
        self.params = taxonomy["parameters"]
        # IMPORTANT : recalcule dynamiquement depuis parameters, ne PAS utiliser
        # taxonomy["total_weight"]. Ce champ est perime pour "Consulting" dans
        # taxonomies.json (declare 1078, somme reelle des 17 poids = 978) --
        # bug trouve en testant ce module contre le fichier reel. Le scorer
        # Phase 1 (experiments/phase1_results/config/scoring.py, ligne 265)
        # fait deja ce recalcul dynamique ; on reproduit exactement la meme
        # methode pour rester comparable a tab:multiturn.
        self.total_weight = sum(p["weight"] for p in self.params.values())

    def compute(self, F: dict) -> float:
        """F: {param_name: value} -- seuls les params RESOLVED (avec une
        valeur) doivent etre presents dans F."""
        resolved_weight = sum(
            self.params[p]["weight"] for p in F if p in self.params
        )
        if self.total_weight == 0:
            return 1.0
        return resolved_weight / self.total_weight

    def missing_params(self, F: dict) -> list[str]:
        return [p for p in self.params if p not in F]

    def param_weight(self, name: str) -> int:
        return self.params.get(name, {}).get("weight", 0)
