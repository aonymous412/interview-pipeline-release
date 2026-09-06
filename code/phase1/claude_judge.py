"""
Exp3 extension — Claude as a second judge (κ triangulation)
===========================================================
Re-scores the 36 already-human-annotated single-pass responses
(responses_30.json / annotation_sample_30.csv, 516 param rows) with a
Claude model as the judge, using the EXACT same scoring prompt as the
Gemini scorer (score_response in config/scoring.py), then computes:

  - κ (Claude judge vs human)   overall + per domain + per class
  - κ (Gemini judge vs human)   recomputed on the same rows (sanity ref:
                                0.749 overall, 0.539 Payments)
  - raw agreement Claude vs Gemini

Motivation: the Payments domain fails its pre-registered κ threshold
(0.539 < 0.65) and three correction attempts failed. If a judge from a
different model family agrees better with humans on Payments, that is a
scorer fix; if it also fails, that documents a taxonomy-level ambiguity
rather than a Gemini-specific bias.

Requires: ANTHROPIC_API_KEY in env (the night orchestrator exports it
from experiments/.env `api_claude`).

Usage:
  python claude_judge.py                     # full 36 prompts
  python claude_judge.py --max-cost 1.0 --model claude-sonnet-5
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic

OUTPUT_DIR = Path(__file__).parent
CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"  # [relocated]
EXP2_DIR = Path(__file__).resolve().parents[2] / "data"  # [relocated]

sys.path.insert(0, str(CONFIG_DIR))
from scoring import TAXONOMIES, _extract_json_from_text  # noqa: E402

sys.path.insert(0, str(OUTPUT_DIR))
from compute_kappa import cohen_kappa  # noqa: E402

# USD per 1M tokens (input, output) — Anthropic pricing, July 2026
PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

VALID = {"RESOLVED", "MENTIONED", "ABSENT"}


def load_prompt_texts() -> dict:
    """id -> prompt text: the 150-prompt Exp2 set + the 6 detailed
    Payments prompts (V01-V06) added by extend_payments.py."""
    with open(CONFIG_DIR / "prompts_P01-P50.json") as f:
        prompts = {p["id"]: p["text"] for p in json.load(f)}
    with open(EXP2_DIR / "prompts_N01-N100.csv", newline="") as f:
        for row in csv.DictReader(f):
            prompts[row["id"]] = row["text"]
    from extend_payments import NEW_DETAILED_PAYMENTS
    for p in NEW_DETAILED_PAYMENTS:
        prompts[p["id"]] = p["text"]
    return prompts


def load_annotation_rows() -> list:
    rows = []
    with open(Path(__file__).resolve().parents[2] / "data" / "annotations" / "human_annotations_516.csv", newline="") as f:  # [relocated]
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        for raw in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in raw.items()}
            if row.get("human_status", "").upper() in VALID:
                row["human_status"] = row["human_status"].upper()
                row["scorer_status"] = row["scorer_status"].upper()
                rows.append(row)
    return rows


def build_scoring_prompt(domain: str, prompt_text: str, response_text: str) -> str:
    """Byte-identical to score_response() in config/scoring.py."""
    params = TAXONOMIES[domain]["parameters"]
    param_descriptions = "\n".join([
        f"- {name} (weight={info['weight']}): {info['description']}"
        for name, info in params.items()
    ])
    return f"""You are a precise annotation assistant for a research project.

TASK: Classify each parameter as RESOLVED, MENTIONED, or ABSENT.

DEFINITIONS:
- RESOLVED: A CONCRETE value is given (e.g., "$5,000", "women aged 25-40", "stage IIB")
- MENTIONED: Discussed but no concrete value committed
- ABSENT: Not addressed at all

DOMAIN: {domain}
ORIGINAL PROMPT: {prompt_text}

PARAMETERS:
{param_descriptions}

RESPONSE TO SCORE:
---
{response_text[:5000]}
---

Return ONLY a JSON object. Keep evidence very short (max 10 words).
Example: {{"param_name": {{"status": "RESOLVED", "evidence": "value found"}}}}"""


def claude_call(client, model, prompt, usage_acc, n_attempts=5):
    delays = [15, 60, 120, 300]
    kwargs = dict(
        model=model,
        max_tokens=4000,
        temperature=0.0,  # match the Gemini judge (temperature=0.0)
        messages=[{"role": "user", "content": prompt}],
    )
    for attempt in range(n_attempts):
        try:
            r = client.messages.create(**kwargs)
            usage_acc[0] += r.usage.input_tokens
            usage_acc[1] += r.usage.output_tokens
            return "\n".join(b.text for b in r.content if b.type == "text")
        except anthropic.BadRequestError as e:
            # Some model generations reject the temperature parameter.
            if "temperature" in str(e).lower() and "temperature" in kwargs:
                print("    [NOTE] temperature rejected by API — retrying without")
                kwargs.pop("temperature")
                continue
            raise
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:
            if attempt == n_attempts - 1:
                raise
            d = delays[min(attempt, len(delays) - 1)]
            print(f"    [RETRY judge] {type(e).__name__}, waiting {d}s")
            time.sleep(d)


def kappa_report(rows, a_field, b_field) -> dict:
    """κ overall + per domain + per class between two status fields."""
    out = {}
    ys = [(r[a_field], r[b_field]) for r in rows]
    out["overall"] = round(cohen_kappa([a for a, _ in ys], [b for _, b in ys]), 4)
    out["n"] = len(ys)
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append((r[a_field], r[b_field]))
    out["by_domain"] = {
        d: {"kappa": round(cohen_kappa([a for a, _ in v], [b for _, b in v]), 4),
            "n": len(v)}
        for d, v in sorted(by_dom.items())
    }
    # Per-class one-vs-rest κ
    labels = sorted(VALID)
    out["by_class"] = {}
    for lab in labels:
        y1 = [("X" if a == lab else "O") for a, _ in ys]
        y2 = [("X" if b == lab else "O") for _, b in ys]
        out["by_class"][lab] = round(cohen_kappa(y1, y2), 4)
    out["raw_agreement"] = round(
        sum(1 for a, b in ys if a == b) / len(ys), 4)
    return out


def main():
    parser = argparse.ArgumentParser(description="Claude as second judge (Exp3 ext)")
    parser.add_argument("--model", default="claude-sonnet-5", choices=list(PRICING))
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    with open(OUTPUT_DIR / "responses_30.json") as f:
        responses = json.load(f)
    prompt_texts = load_prompt_texts()
    ann_rows = load_annotation_rows()

    # domain per prompt_id from the annotation CSV
    domain_of = {r["prompt_id"]: r["domain"] for r in ann_rows}
    prompt_ids = sorted(set(r["prompt_id"] for r in ann_rows))
    print(f"Prompts to judge: {len(prompt_ids)} | annotated param rows: {len(ann_rows)}")

    # Resume support
    judged_path = OUTPUT_DIR / "claude_judge_labels.json"
    judged = {}
    if judged_path.exists() and not args.no_resume:
        with open(judged_path) as f:
            judged = json.load(f)
        print(f"Resume: {len(judged)} prompts already judged")

    client = anthropic.Anthropic()
    usage = [0, 0]
    in_p, out_p = PRICING[args.model]

    def cost():
        return (usage[0] * in_p + usage[1] * out_p) / 1e6

    for i, pid in enumerate(prompt_ids):
        if pid in judged:
            continue
        if cost() >= args.max_cost:
            print(f"[BUDGET] ${cost():.2f} >= cap ${args.max_cost:.2f} — "
                  f"stopping cleanly (resume supported).")
            break
        domain = domain_of[pid]
        prompt = build_scoring_prompt(domain, prompt_texts[pid], responses[pid])
        print(f"[{i+1}/{len(prompt_ids)}] {pid} ({domain}) ... ", end="", flush=True)
        raw = claude_call(client, args.model, prompt, usage)
        ann = _extract_json_from_text(raw)
        if ann is None:
            print("PARSE FAIL")
            judged[pid] = {"_parse_failed": True}
        else:
            judged[pid] = {
                k: str(v.get("status", "ABSENT")).upper()
                for k, v in ann.items() if isinstance(v, dict)
            }
            print(f"ok (${cost():.2f})")
        with open(judged_path, "w") as f:
            json.dump(judged, f, indent=1)

    # Attach Claude labels to annotation rows
    complete = []
    missing = 0
    for r in ann_rows:
        labels = judged.get(r["prompt_id"])
        if labels is None or labels.get("_parse_failed"):
            missing += 1
            continue
        st = labels.get(r["param_name"], "ABSENT")
        if st not in VALID:
            st = "ABSENT"  # param missing from judge output = not addressed
        r2 = dict(r)
        r2["claude_status"] = st
        complete.append(r2)

    print(f"\nRows with Claude labels: {len(complete)} (skipped: {missing})")

    stats = {
        "judge_model": args.model,
        "n_prompts_judged": len([p for p in judged if not judged[p].get("_parse_failed")]),
        "n_rows": len(complete),
        "claude_vs_human": kappa_report(complete, "claude_status", "human_status"),
        "gemini_vs_human": kappa_report(complete, "scorer_status", "human_status"),
        "claude_vs_gemini": kappa_report(complete, "claude_status", "scorer_status"),
        "cost_estimate": {"input_tokens": usage[0], "output_tokens": usage[1],
                          "estimated_usd": round(cost(), 2)},
        "reference": "Gemini scorer pre-registered thresholds: overall κ>=0.65; "
                     "known values 0.749 overall / 0.539 Payments",
    }
    with open(OUTPUT_DIR / "claude_judge_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(OUTPUT_DIR / "claude_judge_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "domain", "param_name", "param_weight",
                    "human_status", "gemini_status", "claude_status"])
        for r in complete:
            w.writerow([r["prompt_id"], r["domain"], r["param_name"],
                        r["param_weight"], r["human_status"],
                        r["scorer_status"], r["claude_status"]])

    print("\n===== CLAUDE JUDGE RESULTS =====")
    for k in ("claude_vs_human", "gemini_vs_human", "claude_vs_gemini"):
        s = stats[k]
        doms = " | ".join(f"{d}: {v['kappa']}" for d, v in s["by_domain"].items())
        print(f"{k}: κ={s['overall']} (n={s['n']}) | {doms}")
    print(f"Cost: ${cost():.2f}")
    print(f"→ {OUTPUT_DIR / 'claude_judge_stats.json'}")


if __name__ == "__main__":
    main()
