"""Central configuration of the ALI pipeline (Phase 2).

Paths are resolved relative to the root of this archive. REPO_ROOT is the
directory holding code/, data/, results/ and reports/; scripts may be run
from anywhere. In the research repo these constants pointed at the repo
layout instead -- see RELOCATION.txt and PROVENANCE.md.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
CONFIG_DIR = REPO_ROOT / "data"                        # [relocated]
EXP3_DIR = REPO_ROOT / "data" / "annotations"          # [relocated]
EXP2_DIR = REPO_ROOT / "results" / "baselines"         # [relocated]

TAXONOMIES_JSON = CONFIG_DIR / "taxonomies.json"
BASELINE_CSV = EXP2_DIR / "baseline_extended_n150.csv"
ANNOTATION_CSV = EXP3_DIR / "human_annotations_516.csv"  # [relocated]
TAXONOMY_PAYMENTS_V2 = CHECKPOINTS_DIR / "ali_standalone" / "taxonomy_correction" / "taxonomy_payments_v2.json"

# Decisions actees (cf. checkpoints/extraction_report.md et prompt Phase 2 v2, Section 1)
# Ne pas re-diagnostiquer rad_assist : aucun adapter PEFT n'existe (Mistral 7B GGUF merge).
DEPLOYMENT_DEFAULTS = {
    "rad_assist": {
        "domain_key": "Medical",       # cle dans taxonomies.json
        "c1_mode": "gemini",           # force -- decision deja actee, pas de "auto"
        "c4_mode": "gemini",
        "theta": None,                 # pas de seuil ALI natif (pas d'architecture C0-C4 reelle)
        "native_taxonomy": "8 variables cliniques fixes (report_to_matrix.py CLINICAL_VARIABLES) -- "
                            "PAS le meme format que Telos/FinAgent. Voir taxonomy_mapping.py.",
    },
    "telos": {
        "domain_key": "Consulting",
        "c1_mode": "auto",             # auto = resultat diagnostic (Section 2)
        "c4_mode": "auto",
        "base_model": "gpt2",
        "checkpoint_c1": str(CHECKPOINTS_DIR / "telos" / "c1"),
        "checkpoint_c4": str(CHECKPOINTS_DIR / "telos" / "c4"),
        "missions_jsonl": str(CHECKPOINTS_DIR / "telos" / "taxonomy_missions.jsonl"),
    },
    "finagent": {
        "domain_key": "Payments",
        "c1_mode": "auto",
        "c4_mode": "auto",
        "base_model_c1": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_model_c4": "Qwen/Qwen2.5-1.5B-Instruct",
        # NB: le dossier reel est "chain_pilot" (underscore), pas "finagent"
        "checkpoint_c1": str(CHECKPOINTS_DIR / "chain_pilot" / "c1"),
        "checkpoint_c4": str(CHECKPOINTS_DIR / "chain_pilot" / "c4"),
        "domain_templates_py": str(CHECKPOINTS_DIR / "chain_pilot" / "taxonomy_extracted_chain_pilot.py"),
    },
}

RESULTS_ALI_DIR = REPO_ROOT / "results" / "ali"        # [relocated]
RESULTS_BASELINES_DIR = REPO_ROOT / "results" / "baselines"  # [relocated]
PHASE1_CODE_DIR = REPO_ROOT / "code" / "phase1"        # [relocated]
# Re-runs write here; shipped results/ and reports/ stay read-only evidence.
RECOMPUTED_DIR = REPO_ROOT / "recomputed"              # [relocated]
RECOMPUTED_DIR.mkdir(exist_ok=True)

MAX_TURNS = 15
KAPPA_THRESHOLD = 0.65
RECALL_THRESHOLD = 0.75

# Convention .env du projet (voir experiments/phase1_results/config/scoring.py) :
# une ligne contenant "gemini" et "=" a la racine du repo, ex: gemini=AIza...
ENV_FILE = REPO_ROOT / ".env"
