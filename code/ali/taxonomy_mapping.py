"""
Tache 3 -- Mapping taxonomie native (par deploiement) -> taxonomie plate
(taxonomies.json), decide avec l'utilisateur pour rester comparable a
tab:multiturn (Phase 1).

Contexte (voir checkpoints/extraction_report.md) :
  - Telos   : C1/C4 sont entraines sur 15 categories de missions.jsonl,
              chacune avec ~16 elements PROPRES (191 noms uniques au total).
              La taxonomie plate "Consulting" n'a que 17 params.
  - FinAgent : C1/C4 sont entraines sur 5 domaines natifs (trading, payroll,
              content, procurement, generic), chacun avec ses propres
              mandatory/optional parameters. La taxonomie plate "Payments"
              (11 params) est calquee sur un agent de trading -- elle ne
              recouvre bien QUE le domaine natif "trading".
  - Rad-Assist : 8 variables cliniques fixes, architecture non-ALI (pas de
              C1/C4 au sens propre). Mapping dedie plus bas.

Principe : cov(F) est toujours calcule contre la taxonomie plate (pour rester
comparable a Single-pass / Gemini multi-tour / Llama multi-tour dans
tab:multiturn). Ce module traduit les noms de parametres retournes par les
composants natifs (C1 natif, C4 natif) vers les noms de la taxonomie plate.

IMPORTANT -- limite connue (a documenter en Limitations du papier) :
Certains parametres de la taxonomie plate n'ont AUCUN equivalent natif dans
un domaine/categorie donne (ex: FinAgent/payroll n'a pas d'equivalent a
risk_tolerance ou stop_loss). Ces parametres resteront structurellement
ABSENT quel que soit le nombre de tours -- ce n'est pas un bug du pipeline,
c'est une consequence du choix (valide par l'utilisateur) de scorer contre
la taxonomie plate plutot que la taxonomie native. get_uncoverable_params()
expose cette liste pour le rapport.
"""
from __future__ import annotations

import re
from functools import lru_cache


def _normalize(name: str) -> str:
    return re.sub(r"_+", "_", name.strip().lower().replace(" ", "_")).strip("_")


def fuzzy_match(native_name: str, flat_names: list[str]) -> str | None:
    """
    Reprend l'heuristique de checkpoints/telos/inference_c1.py::_fuzzy_match_element :
    match exact -> normalise -> sous-chaine (>=5 caracteres, sans underscore).
    """
    norm = _normalize(native_name)
    for flat in flat_names:
        if _normalize(flat) == norm:
            return flat
    flat_flat = {f: _normalize(f).replace("_", "") for f in flat_names}
    norm_flat = norm.replace("_", "")
    for flat, ff in flat_flat.items():
        if ff == norm_flat:
            return flat
    for flat, ff in flat_flat.items():
        if len(norm_flat) >= 5 and (norm_flat in ff or ff in norm_flat):
            return flat
    return None


# ---------------------------------------------------------------------------
# Telos (Consulting, 17 params plats)
# ---------------------------------------------------------------------------

FLAT_CONSULTING = [
    "main_content_purpose", "target_audience", "pages_structure", "core_features",
    "design_style", "content_ready", "existing_branding", "technical_platform",
    "integrations", "budget_range", "timeline", "seo_requirements", "user_accounts",
    "content_management", "analytics_needs", "multilingual_support", "accessibility",
]

# Overrides manuels pour les cas ou le nom natif differe trop du nom plat pour
# etre trouve par fuzzy_match (verifie contre les 191 noms natifs reels de
# checkpoints/telos/taxonomy_missions.jsonl).
TELOS_OVERRIDES = {
    "tech_platform": "technical_platform",
    "platform": "technical_platform",
    "platform_preference": "technical_platform",
    "platform_deployment": "technical_platform",
    "platforms": "technical_platform",
    "tech_stack": "technical_platform",
    "existing_platform": "technical_platform",
    "distribution_platform": "technical_platform",
    "multilingual": "multilingual_support",
    "user_auth": "user_accounts",
    "user_auth_roles": "user_accounts",
    "authentication": "user_accounts",
    "access_control": "user_accounts",
    "analytics": "analytics_needs",
    "analytics_monitoring": "analytics_needs",
    "analytics_tools": "analytics_needs",
    "analytics_tracking": "analytics_needs",
    "key_metrics": "analytics_needs",
    "success_metrics": "analytics_needs",
    # timeline / content_management n'ont pas d'equivalent natif fiable dans la
    # plupart des categories -- laisses au fuzzy matcher / a None (uncoverable).
    "campaign_duration": "timeline",
    "campaign_dates": "timeline",
    "content_calendar": "content_management",
    "content_strategy": "content_management",
}


def map_telos_element(native_name: str) -> str | None:
    if native_name in TELOS_OVERRIDES:
        return TELOS_OVERRIDES[native_name]
    return fuzzy_match(native_name, FLAT_CONSULTING)


def telos_uncoverable_params(category_element_names: list[str]) -> list[str]:
    """Params de la taxonomie plate Consulting qu'aucun element de cette
    categorie de mission ne peut renseigner."""
    mapped = {map_telos_element(n) for n in category_element_names}
    mapped.discard(None)
    return [p for p in FLAT_CONSULTING if p not in mapped]


# ---------------------------------------------------------------------------
# FinAgent (Payments, 11 params plats -- calques sur le domaine "trading")
# ---------------------------------------------------------------------------

FLAT_PAYMENTS = [
    "volume_per_trade", "payment_from", "risk_tolerance", "authorization_controls",
    "daily_limit", "currency", "asset_type", "exchanges", "trading_schedule",
    "stop_loss", "reporting",
]

# Verifie manuellement contre les 5 templates de
# checkpoints/chain_pilot/taxonomy_extracted_chain_pilot.py
FINAGENT_OVERRIDES = {
    # trading (bon recouvrement)
    "time_window": "trading_schedule",
    "spread_check": "stop_loss",          # approximatif -- a valider
    "trading_frequency": "trading_schedule",
    "weekly_limit": "daily_limit",        # approximatif (cumul different)
    # payroll (recouvrement faible -- la plupart resteront uncoverable)
    "approval_threshold": "authorization_controls",
    "payment_to": "payment_from",         # sens inverse, approximatif
    # procurement
    "monthly_budget": "daily_limit",      # approximatif (mensuel vs journalier)
    # generic / content : essentiellement aucun recouvrement
    "main_constraints": "authorization_controls",  # tres approximatif
}

FINAGENT_NATIVE_DOMAINS = {
    "trading": ["payment_from", "currency", "volume_per_trade", "daily_limit",
                "weekly_limit", "asset_type", "exchanges", "trading_frequency",
                "time_window", "spread_check"],
    "payroll": ["position_types", "salary_range_min", "salary_range_max",
                "payment_frequency", "payment_from", "payment_to", "location",
                "currency", "bonus_allowance", "excluded_positions",
                "special_rules", "approval_threshold"],
    "content": ["blocked_topics", "strictness_level", "context", "allowed_subjects",
                "common_violations", "edge_cases", "escalation_rules", "exceptions"],
    "procurement": ["product_categories", "monthly_budget", "currency", "payment_from",
                     "monthly_volume", "payment_to", "suppliers", "quality_requirements",
                     "delivery_constraints", "approval_threshold", "forbidden_categories"],
    "generic": ["agent_purpose", "main_constraints", "categories", "limits",
                "rules", "exceptions"],
}


def map_finagent_element(native_name: str) -> str | None:
    if native_name in FINAGENT_OVERRIDES:
        return FINAGENT_OVERRIDES[native_name]
    return fuzzy_match(native_name, FLAT_PAYMENTS)


def finagent_uncoverable_params(native_domain: str) -> list[str]:
    native_names = FINAGENT_NATIVE_DOMAINS.get(native_domain, [])
    mapped = {map_finagent_element(n) for n in native_names}
    mapped.discard(None)
    gap = [p for p in FLAT_PAYMENTS if p not in mapped]
    return gap


# ---------------------------------------------------------------------------
# Rad-Assist (Medical, 17 params plats -- architecture non-ALI, 8 vars fixes)
# ---------------------------------------------------------------------------

FLAT_MEDICAL = [
    "primary_site", "stage_classification", "treatment_received", "histology_subtype",
    "performance_status", "response_to_treatment", "current_medications", "biomarkers",
    "imaging_findings", "comorbidities", "patient_demographics", "lab_values",
    "treatment_plan", "follow_up_schedule", "patient_preferences", "contraindications",
    "family_history",
]

# 8 CLINICAL_VARIABLES de checkpoints/rad_assist/inference_report_to_matrix.py
# (voir extraction_report.md) : recouvrement partiel avec la taxonomie plate.
RAD_ASSIST_OVERRIDES = {
    "performance_status": "performance_status",       # match direct
    "tumor_size_mm": "imaging_findings",
    "sum_of_diameters_mm": "imaging_findings",
    "lesion_count": "imaging_findings",
    "non_target_status": "response_to_treatment",
    "new_lesion_flag": "response_to_treatment",
    "tumor_density_hu": "imaging_findings",
    "enhancement_pattern": "imaging_findings",
}


def map_rad_assist_element(native_name: str) -> str | None:
    return RAD_ASSIST_OVERRIDES.get(native_name) or fuzzy_match(native_name, FLAT_MEDICAL)


def rad_assist_uncoverable_params() -> list[str]:
    mapped = {map_rad_assist_element(n) for n in RAD_ASSIST_OVERRIDES}
    mapped.discard(None)
    return [p for p in FLAT_MEDICAL if p not in mapped]


# ---------------------------------------------------------------------------
# API unifiee
# ---------------------------------------------------------------------------

MAPPERS = {
    "telos": map_telos_element,
    "finagent": map_finagent_element,
    "rad_assist": map_rad_assist_element,
}


def map_native_to_flat(deployment: str, native_name: str) -> str | None:
    mapper = MAPPERS.get(deployment)
    if mapper is None:
        raise ValueError(f"Deploiement inconnu: {deployment}")
    return mapper(native_name)


@lru_cache(maxsize=None)
def get_uncoverable_params(deployment: str, native_domain: str | None = None) -> tuple:
    """Params de la taxonomie plate qu'aucun element natif ne peut renseigner.
    A inclure telles quelles dans le rapport / section Limitations."""
    if deployment == "finagent":
        return tuple(finagent_uncoverable_params(native_domain or "trading"))
    if deployment == "rad_assist":
        return tuple(rad_assist_uncoverable_params())
    if deployment == "telos":
        # Sans categorie precisee, on ne peut pas calculer le gap exact --
        # l'appelant doit passer les elements de la categorie detectee par C0/C1.
        return tuple()
    raise ValueError(f"Deploiement inconnu: {deployment}")
