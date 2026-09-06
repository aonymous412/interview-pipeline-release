"""
Section 5 -- Evaluation sur les 150 prompts Phase 1.

NON EXECUTABLE dans ce sandbox Cowork : necessite (1) l'API Gemini
(generativelanguage.googleapis.com bloque par le proxy reseau) et (2) le
chargement des checkpoints PEFT (huggingface.co bloque). A executer sur
l'agent local avec un fichier .env valide (cle "gemini=...") a la racine du
repo. Voir HANDOFF.md pour les instructions completes.

Usage (local) :
    cd ali_standalone && python3 run_phase2.py --config ALI_v1
    cd ali_standalone && python3 run_phase2.py --config all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [relocated] code/ on sys.path
from ali import config  # noqa: E402
from ali.ali_pipeline import ALIPipeline  # noqa: E402
from ali.user_simulator import GeminiUserSimulator  # noqa: E402

CONFIGS = [
    {"name": "ALI_v1", "c1": "finetuned", "c4": "gemini"},     # chemin principal
    {"name": "ALI_v2", "c1": "gemini", "c4": "gemini"},        # full-Gemini baseline
    {"name": "ALI_v3", "c1": "finetuned", "c4": "finetuned"},  # si kappa_C4 >= 0.65
]

DOMAIN_TO_DEPLOYMENT = {
    "Consulting": "telos",
    "Medical": "rad_assist",
    "Payments": "finagent",
}


def load_prompts() -> pd.DataFrame:
    return pd.read_csv(config.BASELINE_CSV, skipinitialspace=True)


def run_config(cfg: dict, prompts_df: pd.DataFrame, out_path: Path) -> list[dict]:
    """Sauvegarde incrementale (apres chaque prompt) + resume depuis un fichier
    partiel existant + tolerance aux erreurs par prompt (un crash sur un prompt
    n'interrompt pas le run de plusieurs heures, il est juste logue et saute)."""
    results: list[dict] = []
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        done_ids = {r["prompt_id"] for r in results}
        print(f"Reprise : {len(done_ids)} prompts deja traites dans {out_path}")

    for i, (_, row) in enumerate(prompts_df.iterrows()):
        if row["prompt_id"] in done_ids:
            continue
        deployment = DOMAIN_TO_DEPLOYMENT[row["domain"]]
        # NB: rad_assist force gemini/gemini quel que soit cfg (decision actee) --
        # ALIPipeline._resolve_mode l'impose deja via config.DEPLOYMENT_DEFAULTS.
        try:
            pipeline = ALIPipeline(deployment=deployment, c1_mode=cfg["c1"], c4_mode=cfg["c4"])
            simulator = GeminiUserSimulator(initial_prompt=row["prompt"])
            result = pipeline.run(row["prompt"], simulator, prompt_id=row["prompt_id"])
        except Exception as e:
            print(f"  [ERREUR] {row['prompt_id']} ({deployment}) : {e} -- prompt saute")
            continue
        result["config"] = cfg["name"]
        result["vagueness"] = row["vagueness"]
        result["domain"] = row["domain"]
        results.append(result)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [{i + 1}/{len(prompts_df)}] {row['prompt_id']} ({deployment}) "
              f"cov={result['final_cov']:.3f} turns={result['n_turns']} "
              f"({len(results)} sauvegardes)")
    return results


def load_prompt_texts() -> dict[str, str]:
    """Joint prompts.json (50) + prompts_new_100.csv (100) -> {prompt_id: text}."""
    texts = {}
    with open(config.CONFIG_DIR / "prompts_P01-P50.json") as f:
        for p in json.load(f):
            texts[p["id"]] = p["text"]
    extra_path = config.CONFIG_DIR / "prompts_N01-N100.csv"  # [relocated]
    if extra_path.exists():
        extra = pd.read_csv(extra_path, skipinitialspace=True)
        id_col = "prompt_id" if "prompt_id" in extra.columns else extra.columns[0]
        text_col = "text" if "text" in extra.columns else extra.columns[-1]
        for _, r in extra.iterrows():
            texts[r[id_col]] = r[text_col]
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all", help="ALI_v1 | ALI_v2 | ALI_v3 | all")
    args = parser.parse_args()

    prompts_df = load_prompts()
    prompt_texts = load_prompt_texts()
    prompts_df["prompt"] = prompts_df["prompt_id"].map(prompt_texts)
    missing = prompts_df["prompt"].isna().sum()
    if missing:
        print(f"ATTENTION : {missing} prompt_id sans texte trouve (verifier la jointure)")

    configs_to_run = CONFIGS if args.config == "all" else [c for c in CONFIGS if c["name"] == args.config]

    for cfg in configs_to_run:
        print(f"--- Running {cfg['name']} (c1={cfg['c1']}, c4={cfg['c4']}) ---")
        out_path = config.RECOMPUTED_DIR / f"results_{cfg['name']}.json"  # [relocated]
        results = run_config(cfg, prompts_df, out_path)
        print(f"Termine : {len(results)} resultats dans {out_path}")


if __name__ == "__main__":
    main()
