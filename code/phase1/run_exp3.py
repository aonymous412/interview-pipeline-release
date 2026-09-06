"""
Experiment 3: Scoring Validation — Cohen's κ
==============================================
Validates the Gemini 2.5 Flash scorer reliability against human annotation.

Protocol:
  1. Select 30 prompts from Exp2 baseline (10 per domain, stratified by vagueness
     and coverage tier: ~5 clear + ~5 ambiguous per domain)
  2. Re-run Gemini single-pass on each selected prompt → get response text
  3. Re-run Gemini scorer → get per-parameter RESOLVED/MENTIONED/ABSENT labels
  4. Save annotation_sample_30.csv with scorer labels + blank human_status column
  5. Human annotator fills in human_status for each param row
  6. Run compute_kappa.py to calculate Cohen's κ and generate stats

Output files:
  annotation_sample_30.csv     — scorer labels + human_status column (to fill)
  responses_30.json            — full Gemini response texts for reference
  agreement_stats.json         — κ stats (after human annotation + kappa computation)
  disagreement_analysis.md     — analysis of disagreements (after kappa computation)

Author: (anonymised for review)
Date: 2026-06-30
"""

import json
import csv
import sys
import time
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # [relocated]
from scoring import score_response, get_gemini_client, call_gemini

# ─── Constants ────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recomputed"  # [relocated]
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"  # [relocated]
EXP2_CSV = Path(__file__).resolve().parents[2] / "results" / "baselines" / "baseline_extended_n150.csv"  # [relocated]

SINGLE_PASS_SYSTEM = (
    "You are a project requirements analyst. Extract all requirements "
    "from the following description and structure them into a complete, "
    "detailed specification. Be thorough — cover every aspect needed "
    "to implement this project/configure this system."
)

# ─── Load full prompt list ─────────────────────────────────────────────────────

def load_all_prompts() -> dict:
    """Load all 150 prompts. Returns dict id→prompt_data."""
    with open(CONFIG_DIR / "prompts_P01-P50.json") as f:
        orig = json.load(f)

    # Import new prompts from exp2 script
    sys.path.insert(0, str(Path(__file__).parent))  # [relocated]
    from run_exp2 import NEW_PROMPTS

    all_prompts = orig + NEW_PROMPTS
    return {p["id"]: p for p in all_prompts}


def load_exp2_results() -> list:
    """Load coverage results from Exp2."""
    results = []
    with open(EXP2_CSV, newline='') as f:
        for row in csv.DictReader(f):
            row["final_cov"] = float(row["final_cov"])
            results.append(row)
    return results


def select_30_prompts(results: list) -> list:
    """
    Select 30 prompts (10 per domain) with stratified sampling:
      - 5 'clear' cases: cov < 25% (clearly low) or cov > 60% (clearly high)
      - 5 'ambiguous' cases: 25% ≤ cov ≤ 60% (harder to annotate)

    Also ensure vagueness diversity within each group.
    Seed=42 for reproducibility.
    """
    rng = random.Random(SEED)
    selected = []

    for domain in ["Consulting", "Medical", "Payments"]:
        domain_rows = [r for r in results if r["domain"] == domain]

        clear_low = [r for r in domain_rows if r["final_cov"] < 0.25]
        clear_high = [r for r in domain_rows if r["final_cov"] > 0.60]
        ambiguous = [r for r in domain_rows if 0.25 <= r["final_cov"] <= 0.60]

        # Build clear pool (low + high combined), shuffle
        clear_pool = clear_low + clear_high
        rng.shuffle(clear_pool)
        rng.shuffle(ambiguous)

        # Take up to 5 clear and 5 ambiguous
        picked_clear = clear_pool[:5]
        picked_ambig = ambiguous[:5]

        # If we don't have enough in one category, backfill from the other
        shortfall_clear = 5 - len(picked_clear)
        if shortfall_clear > 0:
            picked_ambig_extra = ambiguous[5:5 + shortfall_clear]
            picked_clear += picked_ambig_extra
        shortfall_ambig = 5 - len(picked_ambig)
        if shortfall_ambig > 0:
            picked_clear_extra = clear_pool[5:5 + shortfall_ambig]
            picked_ambig += picked_clear_extra

        domain_selected = picked_clear[:5] + picked_ambig[:5]
        assert len(domain_selected) == 10, f"Expected 10 for {domain}, got {len(domain_selected)}"

        for r in domain_selected:
            r["annotation_tier"] = "clear" if (r["final_cov"] < 0.25 or r["final_cov"] > 0.60) else "ambiguous"
            selected.append(r)

    return selected


def get_gemini_response(prompt_text: str, gemini_client) -> str:
    """Get Gemini single-pass response text."""
    full_prompt = f"{SINGLE_PASS_SYSTEM}\n\nUser request:\n{prompt_text}"
    try:
        text = call_gemini(full_prompt, client=gemini_client, temperature=0.0, max_tokens=3000)
        time.sleep(5)
        return text or ""
    except Exception as e:
        print(f"    [ERROR] Gemini response failed: {e}")
        time.sleep(15)
        return ""


def get_scorer_annotations(response_text: str, domain: str, prompt_text: str,
                            gemini_client) -> dict:
    """Get per-parameter Gemini scorer annotations."""
    try:
        time.sleep(4)
        result = score_response(response_text, domain, prompt_text,
                                gemini_client=gemini_client, scorer="gemini")
        return result
    except Exception as e:
        print(f"    [SCORER ERROR] {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENT 3: Scoring Validation — Cohen's κ")
    print("Selecting 30 prompts (10/domain, 5 clear + 5 ambiguous)")
    print("=" * 60 + "\n")

    gemini_client = get_gemini_client()
    prompt_map = load_all_prompts()
    results = load_exp2_results()

    # Select 30 prompts
    selected = select_30_prompts(results)
    print(f"Selected {len(selected)} prompts:")
    for r in selected:
        print(f"  {r['prompt_id']} [{r['domain']}] [{r['vagueness']}] "
              f"cov={r['final_cov']:.1%} [{r['annotation_tier']}]")

    # Load existing annotation file (resume support)
    ann_path = OUTPUT_DIR / "annotation_sample_30.csv"
    done_ids = set()
    existing_rows = []
    if ann_path.exists() and not args.no_resume:
        with open(ann_path, newline='') as f:
            for row in csv.DictReader(f):
                if row.get("scorer_status"):
                    done_ids.add(row["prompt_id"])
                existing_rows.append(row)
        print(f"\nResume: {len(done_ids)} prompts already annotated by scorer.")

    # Load or init responses cache
    responses_path = OUTPUT_DIR / "responses_30.json"
    responses_cache = {}
    if responses_path.exists() and not args.no_resume:
        with open(responses_path) as f:
            responses_cache = json.load(f)

    # Process each selected prompt
    all_annotation_rows = []
    for i, sel in enumerate(selected):
        pid = sel["prompt_id"]
        if pid in done_ids:
            print(f"\n  [{i+1}/30] {pid} — already done, skipping")
            continue

        prompt_data = prompt_map[pid]
        domain = sel["domain"]
        print(f"\n  [{i+1}/30] {pid} [{domain}] [{sel['vagueness']}] "
              f"cov={sel['final_cov']:.1%} [{sel['annotation_tier']}]")
        print(f"    \"{prompt_data['text'][:80]}{'...' if len(prompt_data['text'])>80 else ''}\"")

        # Get Gemini response (or use cached)
        if pid in responses_cache:
            response_text = responses_cache[pid]
            print(f"    [CACHED] response ({len(response_text)} chars)")
        else:
            print(f"    Getting Gemini response...")
            response_text = get_gemini_response(prompt_data["text"], gemini_client)
            responses_cache[pid] = response_text
            # Save cache after each response
            with open(responses_path, 'w') as f:
                json.dump(responses_cache, f, indent=2, ensure_ascii=False)

        if not response_text:
            print(f"    [SKIP] Empty response")
            continue

        # Get scorer annotations
        print(f"    Scoring per-parameter...")
        score_result = get_scorer_annotations(response_text, domain, prompt_data["text"],
                                               gemini_client)

        if score_result is None:
            print(f"    [SKIP] Scoring failed")
            continue

        cov = score_result["coverage"]
        params = score_result["parameters"]
        print(f"    → cov={cov:.1%} | "
              f"resolved={len(score_result['params_resolved'])} | "
              f"mentioned={len(score_result['params_mentioned'])} | "
              f"absent={len(score_result['params_absent'])}")

        # Add rows for each parameter
        for param_name, param_info in params.items():
            all_annotation_rows.append({
                "prompt_id": pid,
                "domain": domain,
                "vagueness": sel["vagueness"],
                "annotation_tier": sel["annotation_tier"],
                "coverage_scorer": round(cov, 4),
                "param_name": param_name,
                "param_weight": param_info["weight"],
                "scorer_status": param_info["status"],
                "scorer_evidence": param_info["evidence"][:100],
                "human_status": "",  # TO BE FILLED BY HUMAN ANNOTATOR
                "human_notes": "",   # optional
            })

        # Save annotation CSV incrementally
        all_rows_to_save = [r for r in existing_rows
                            if not any(nr["prompt_id"] == r.get("prompt_id", "") and
                                       nr["param_name"] == r.get("param_name", "")
                                       for nr in all_annotation_rows)]
        all_rows_to_save += all_annotation_rows

        with open(ann_path, 'w', newline='') as f:
            fieldnames = [
                "prompt_id", "domain", "vagueness", "annotation_tier",
                "coverage_scorer", "param_name", "param_weight",
                "scorer_status", "scorer_evidence",
                "human_status", "human_notes"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows_to_save)

        print(f"    → Saved {len(all_rows_to_save)} annotation rows")
        time.sleep(2)

    # Final save
    if all_annotation_rows:
        final_rows = [r for r in existing_rows
                      if not any(nr["prompt_id"] == r.get("prompt_id", "") and
                                 nr["param_name"] == r.get("param_name", "")
                                 for nr in all_annotation_rows)]
        final_rows += all_annotation_rows

        with open(ann_path, 'w', newline='') as f:
            fieldnames = [
                "prompt_id", "domain", "vagueness", "annotation_tier",
                "coverage_scorer", "param_name", "param_weight",
                "scorer_status", "scorer_evidence",
                "human_status", "human_notes"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)

    # Summary
    print(f"\n{'='*60}")
    print("STEP 1 COMPLETE — Scorer annotations generated")
    print(f"{'='*60}")
    total_params = sum(1 for r in (all_annotation_rows or existing_rows) if r.get("scorer_status"))
    print(f"\nFile: {ann_path}")
    print(f"Total parameter annotations: {total_params}")
    print(f"Prompts covered: {len(set(r.get('prompt_id') for r in (all_annotation_rows or existing_rows)))}/30")
    print(f"\n>>> NEXT STEP: Open annotation_sample_30.csv and fill in the")
    print(f"    'human_status' column for each row.")
    print(f"    Valid values: RESOLVED / MENTIONED / ABSENT")
    print(f"    Reference: scorer_evidence column shows what the scorer saw.")
    print(f"\n    Then run: python3 compute_kappa.py")


if __name__ == "__main__":
    main()
