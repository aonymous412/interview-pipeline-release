"""
Analyse de sensibilite au seuil : ALI_v1 sur les 50 prompts Payments a theta=0.90.

Question posee : le 83.9 % / 3.98 tours de Payments (tab:ali_domain) mesure-t-il une
faiblesse du deploiement FinAgent, ou seulement son seuil d'arret (theta=0.75, contre
0.90 Telos et 0.92 Rad-Assist) ? La boucle s'arretant des que cov(F) >= theta, la
couverture publiee est bornee par le seuil ; ce script relance le meme pipeline, les
memes prompts et le meme scorer avec le seul theta modifie.

Aucun fichier publie n'est touche : THETA est patche en memoire et la sortie va dans
results_ALI_v1_payments_theta090.json.

Usage :
    python3 checkpoints/ali_standalone/run_theta_sensitivity.py [--theta 0.90]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [relocated] code/ on sys.path
from ali import config  # noqa: E402
from ali import ali_pipeline as ali_mod  # noqa: E402
from ali.ali_pipeline import ALIPipeline  # noqa: E402
from ali.run_phase2 import load_prompt_texts  # noqa: E402
from ali.user_simulator import GeminiUserSimulator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, default=0.90)
    parser.add_argument("--deployment", default="finagent",
                        choices=["finagent", "rad_assist", "telos"])
    parser.add_argument("--limit", type=int, default=0, help="0 = les 50 prompts")
    args = parser.parse_args()

    DOMAIN = {"finagent": "Payments", "rad_assist": "Medical", "telos": "Consulting"}
    SLUG = {"finagent": "payments", "rad_assist": "medical", "telos": "consulting"}
    dep, domain = args.deployment, DOMAIN[args.deployment]

    out_path = (Path(__file__).parent
                / f"results_ALI_v1_{SLUG[dep]}_theta{int(args.theta * 100):03d}.json")

    baseline_theta = ali_mod.THETA[dep]
    ali_mod.THETA[dep] = args.theta
    print(f"theta {dep} : {baseline_theta} -> {args.theta} (patch memoire)")

    prompts_df = pd.read_csv(config.BASELINE_CSV, skipinitialspace=True)
    prompts_df = prompts_df[prompts_df["domain"] == domain].copy()
    texts = load_prompt_texts()
    prompts_df["prompt"] = prompts_df["prompt_id"].map(texts)
    if args.limit:
        prompts_df = prompts_df.head(args.limit)

    results: list[dict] = []
    done: set[str] = set()
    if out_path.exists():
        results = json.loads(out_path.read_text())
        done = {r["prompt_id"] for r in results}
        print(f"Reprise : {len(done)} prompts deja traites")

    total = len(prompts_df)
    for i, (_, row) in enumerate(prompts_df.iterrows(), 1):
        if row["prompt_id"] in done:
            continue
        try:
            pipeline = ALIPipeline(deployment=dep, c1_mode="finetuned", c4_mode="gemini")
            simulator = GeminiUserSimulator(initial_prompt=row["prompt"])
            result = pipeline.run(row["prompt"], simulator, prompt_id=row["prompt_id"])
        except Exception as e:  # meme tolerance par prompt que run_phase2
            print(f"  [ERREUR] {row['prompt_id']} : {e} -- prompt saute")
            continue
        result["config"] = f"ALI_v1_theta{args.theta}"
        result["theta"] = args.theta
        result["vagueness"] = row["vagueness"]
        result["domain"] = row["domain"]
        result["deployment"] = dep
        result["baseline_cov"] = float(row["final_cov"])
        results.append(result)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  [{i}/{total}] {row['prompt_id']} cov={result['final_cov']:.3f} "
              f"turns={result['n_turns']} stop={result['stopped_by']}")

    print(f"Termine : {len(results)} resultats dans {out_path}")


if __name__ == "__main__":
    main()
