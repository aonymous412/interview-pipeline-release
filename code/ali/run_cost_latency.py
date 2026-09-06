"""
Mesure du cout et de la latence d'ALI_v1 sur le sous-ensemble apparie n=30.

A LANCER EN LOCAL (acces reseau Gemini + poids C1/C4 requis) :

    python3 checkpoints/ali_standalone/run_cost_latency.py            # les 30
    python3 checkpoints/ali_standalone/run_cost_latency.py --n 2      # essai a blanc

Le sous-ensemble est celui de l'ablation n=30 (`ablation.sample_30`, seed 42,
10 prompts par domaine) : les chiffres sont donc apparies avec ceux de
[[T1 Ablation n30]] et avec la ré-execution d'Opus sur les memes prompts.

Sorties (reprise possible : le fichier est reecrit apres chaque prompt) :
    checkpoints/ali_standalone/cost_latency_ALI_v1.json

Ce que le script NE fait pas : mesurer les baselines. La latence d'Opus sur les
memes 30 prompts se mesure avec run_multiturn_claude.py ; seule la latence est a
re-mesurer, les tokens des baselines etant deja reconstruits et valides par
cost_analysis.py.

Avertissement de lecture : l'horloge murale depend de la charge de l'API et de la
machine. Elle se rapporte comme une mediane indicative, jamais comme une garantie
de performance. Les tokens et le nombre d'appels, eux, sont reproductibles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # [relocated] code/ on sys.path

from ali import config  # noqa: E402
from ali import cost_instrumentation as ci  # noqa: E402
from ali.ablation import sample_30  # noqa: E402
from ali.ali_pipeline import ALIPipeline  # noqa: E402
from ali.run_phase2 import (  # noqa: E402
    DOMAIN_TO_DEPLOYMENT, load_prompt_texts,
)
from ali.user_simulator import GeminiUserSimulator  # noqa: E402

ROOT_EXP2 = Path(__file__).resolve().parents[2] / "experiments/phase1_results/exp2_baseline_extended"

HERE = Path(__file__).parent

# Memes definitions que run_phase2.CONFIGS -- ne pas diverger.
ALI_CONFIGS = {
    "ALI_v1": {"c1": "finetuned", "c4": "gemini"},
    "ALI_v2": {"c1": "gemini", "c4": "gemini"},
    "ALI_v3": {"c1": "finetuned", "c4": "finetuned"},
}


def run_single_pass(prompts, out_path: Path) -> None:
    """Extraction en une passe : un seul appel Gemini, aucune interview.

    Reutilise SINGLE_PASS_SYSTEM du runner d'Exp2 pour que le payload soit
    exactement celui qui a produit les 40.7 % publies.
    """
    import time
    sys.path.insert(0, str(ROOT_EXP2))
    from run_exp2 import SINGLE_PASS_SYSTEM  # noqa: E402

    from ali import gemini_client

    results: list[dict] = []
    done: set[str] = set()
    if out_path.exists():
        results = json.load(open(out_path))
        done = {r["prompt_id"] for r in results}

    for _, row in prompts.iterrows():
        if row["prompt_id"] in done:
            continue
        full = f"{SINGLE_PASS_SYSTEM}\n\nUser request:\n{row['prompt']}"
        ci.pop_episode()
        with ci.component("single_pass"):
            try:
                gemini_client.gemini_call("", full)
            except Exception as e:  # noqa: BLE001
                print(f"  [ERREUR] {row['prompt_id']}: {e}")
                ci.pop_episode()
                continue
        calls = ci.pop_episode()
        agg = ci.summarise(calls)
        # le systeme est vide et tout le texte est dans `full` : on corrige le
        # decompte de caracteres d'entree pour refleter le payload reel
        results.append({"prompt_id": row["prompt_id"], "domain": row["domain"],
                        "vagueness": row["vagueness"], "n_turns": 1,
                        "cost": agg})
        json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
        s = agg["system"]
        print(f"  {row['prompt_id']:6s} | {s['calls']} appel, "
              f"{s['in_tok'] + s['out_tok']:6d} tok, {s['wall_s']:5.1f} s "
              f"({len(results)}/{len(prompts)})")
        time.sleep(1)

    if results:
        n = len(results)
        k = lambda f: sum(f(r) for r in results) / n  # noqa: E731
        print(f"\nSingle-pass, moyenne sur n={n} : "
              f"{k(lambda r: r['cost']['system']['in_tok'] + r['cost']['system']['out_tok']):.0f} tok, "
              f"{k(lambda r: r['cost']['system']['chars']):.0f} car., "
              f"{k(lambda r: r['cost']['system']['wall_s']):.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10,
                    help="prompts par domaine (10 = les 30 de l'ablation)")
    ap.add_argument("--config", default="ALI_v1",
                    choices=[*ALI_CONFIGS, "single_pass"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = HERE / f"cost_latency_{args.config}.json"

    ci.install()

    # Exactement la construction de ablation.py:131-134 -- meme CSV, meme jointure,
    # meme seed : le sous-ensemble est donc identique a celui de l'ablation n=30.
    prompts_df = pd.read_csv(config.BASELINE_CSV, skipinitialspace=True)
    prompts_df["prompt"] = prompts_df["prompt_id"].map(load_prompt_texts())
    prompts = sample_30(prompts_df, n_per_domain=args.n)

    if args.config == "single_pass":
        run_single_pass(prompts, args.out)
        return
    modes = ALI_CONFIGS[args.config]

    results: list[dict] = []
    done: set[str] = set()
    if args.out.exists():
        results = json.load(open(args.out))
        done = {r["prompt_id"] for r in results}
        print(f"Reprise : {len(done)} prompts deja mesures.")

    for _, row in prompts.iterrows():
        if row["prompt_id"] in done:
            continue
        deployment = DOMAIN_TO_DEPLOYMENT[row["domain"]]
        try:
            pipeline = ALIPipeline(deployment=deployment,
                                   c1_mode=modes["c1"], c4_mode=modes["c4"])
            ci.wrap_pipeline(pipeline)
            simulator = ci.wrap_oracle(GeminiUserSimulator(initial_prompt=row["prompt"]))
            ci.pop_episode()                      # repart d'un compteur propre
            res = pipeline.run(row["prompt"], simulator, prompt_id=row["prompt_id"])
        except Exception as e:  # noqa: BLE001
            print(f"  [ERREUR] {row['prompt_id']}: {e} -- saute")
            ci.pop_episode()
            continue

        calls = ci.pop_episode()
        rec = {
            "prompt_id": row["prompt_id"],
            "domain": row["domain"],
            "vagueness": row["vagueness"],
            "final_cov": res["final_cov"],
            "n_turns": res["n_turns"],
            "stopped_by": res["stopped_by"],
            "cost": ci.summarise(calls),
        }
        results.append(rec)
        json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
        s = rec["cost"]["system"]
        print(f"  {row['prompt_id']:6s} cov={res['final_cov']:.3f} "
              f"turns={res['n_turns']:2d} | {s['calls']:3d} appels, "
              f"{s['in_tok'] + s['out_tok']:6d} tok, {s['wall_s']:6.1f} s "
              f"({len(results)}/{len(prompts)})")

    if not results:
        print("Aucun resultat.")
        return

    n = len(results)
    sysagg = {k: sum(r["cost"]["system"][k] for r in results) / n
              for k in ("calls", "in_tok", "out_tok", "chars", "wall_s")}
    ora = {k: sum(r["cost"]["oracle"][k] for r in results) / n
           for k in ("calls", "in_tok", "out_tok", "chars", "wall_s")}
    print(f"\n{args.config} par interview, moyenne sur n={n} :")
    print(f"  appels LLM : {sysagg['calls']:.2f}   tours : "
          f"{sum(r['n_turns'] for r in results) / n:.2f}")
    print(f"  tokens     : {sysagg['in_tok']:.0f} in + {sysagg['out_tok']:.0f} out "
          f"= {sysagg['in_tok'] + sysagg['out_tok']:.0f}")
    print(f"  caracteres : {sysagg['chars']:.0f}")
    print(f"  horloge    : {sysagg['wall_s']:.1f} s")
    print(f"  [oracle, exclu du cout systeme : {ora['calls']:.2f} appels, "
          f"{ora['in_tok'] + ora['out_tok']:.0f} tok, {ora['wall_s']:.1f} s]")
    print("\nDecomposition par composant (moyenne par interview) :")
    for tag in ("C0", "C1", "C2", "C3", "C4"):
        c = {k: sum(r["cost"]["by_component"][tag][k] for r in results) / n
             for k in ("calls", "in_tok", "out_tok", "wall_s", "wall_s_incl")}
        print(f"  {tag} : {c['calls']:5.2f} appels  "
              f"{c['in_tok'] + c['out_tok']:7.0f} tok  "
              f"{c['wall_s_incl']:6.2f} s")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
