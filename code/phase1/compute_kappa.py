"""
Exp3 — Step 2: Compute Cohen's κ from filled annotation CSV.

Run this AFTER filling in human_status in annotation_sample_30.csv.

Usage:
    python3 compute_kappa.py

Outputs:
    agreement_stats.json      — κ overall + per domain + per class
    disagreement_analysis.md  — qualitative analysis of disagreements
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recomputed"  # [relocated] writes
OUTPUT_DIR.mkdir(exist_ok=True)
ANN_CSV = Path(__file__).resolve().parents[2] / "data" / "annotations" / "human_annotations_516.csv"  # [relocated]

VALID_STATUSES = {"RESOLVED", "MENTIONED", "ABSENT"}


def cohen_kappa(y1: list, y2: list, labels=None) -> float:
    """Compute Cohen's κ for two lists of categorical labels."""
    if labels is None:
        labels = sorted(set(y1) | set(y2))
    n = len(y1)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Confusion matrix
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        if a in label_to_idx and b in label_to_idx:
            cm[label_to_idx[a]][label_to_idx[b]] += 1

    # Observed agreement
    p_o = np.trace(cm) / n

    # Expected agreement
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = np.dot(row_sums, col_sums) / (n ** 2)

    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1 - p_e))


def load_annotations() -> list:
    """Load annotation CSV, validate human_status values."""
    rows = []
    missing = 0
    with open(ANN_CSV, newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        # Normalize header keys (strip whitespace) in case the file was
        # reformatted/aligned by an editor or linter.
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for raw_row in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                   for k, v in raw_row.items()}
            human = row.get("human_status", "").strip().upper()
            if not human:
                missing += 1
                continue
            if human not in VALID_STATUSES:
                print(f"  [WARN] Invalid human_status '{human}' for "
                      f"{row['prompt_id']}/{row['param_name']} — skipped")
                continue
            row["human_status"] = human
            row["scorer_status"] = row["scorer_status"].strip().upper()
            rows.append(row)

    if missing:
        print(f"  [WARN] {missing} rows have no human_status — excluded from κ computation")
    return rows


def main():
    print("=" * 60)
    print("EXP3: Computing Cohen's κ")
    print("=" * 60 + "\n")

    if not ANN_CSV.exists():
        print(f"ERROR: {ANN_CSV} not found. Run run_exp3.py first.")
        sys.exit(1)

    rows = load_annotations()
    if not rows:
        print("No annotated rows found. Fill in human_status in the CSV first.")
        sys.exit(1)

    scorer_labels = [r["scorer_status"] for r in rows]
    human_labels = [r["human_status"] for r in rows]

    # Overall κ
    kappa_overall = cohen_kappa(scorer_labels, human_labels, labels=list(VALID_STATUSES))
    n_agree = sum(1 for s, h in zip(scorer_labels, human_labels) if s == h)
    p_agree = n_agree / len(rows)

    print(f"Overall results ({len(rows)} parameter annotations)")
    print(f"  Agreement: {p_agree:.1%} ({n_agree}/{len(rows)})")
    print(f"  Cohen's κ: {kappa_overall:.4f}")
    passed = bool(kappa_overall >= 0.65)
    print(f"  Threshold κ ≥ 0.65: {'PASSED ✓' if passed else 'FAILED ✗'}")

    # κ by domain
    kappa_by_domain = {}
    for domain in ["Consulting", "Medical", "Payments"]:
        d_rows = [r for r in rows if r["domain"] == domain]
        if not d_rows:
            continue
        s = [r["scorer_status"] for r in d_rows]
        h = [r["human_status"] for r in d_rows]
        k = cohen_kappa(s, h, labels=list(VALID_STATUSES))
        n_agr = sum(1 for a, b in zip(s, h) if a == b)
        kappa_by_domain[domain] = {
            "n": len(d_rows), "kappa": round(k, 4),
            "agreement": round(n_agr / len(d_rows), 4),
            "passed": bool(k >= 0.65)
        }
        print(f"  {domain}: κ={k:.4f} (n={len(d_rows)}, agree={n_agr/len(d_rows):.1%})")

    # κ by annotation tier
    kappa_by_tier = {}
    for tier in ["clear", "ambiguous"]:
        t_rows = [r for r in rows if r.get("annotation_tier") == tier]
        if not t_rows:
            continue
        s = [r["scorer_status"] for r in t_rows]
        h = [r["human_status"] for r in t_rows]
        k = cohen_kappa(s, h, labels=list(VALID_STATUSES))
        n_agr = sum(1 for a, b in zip(s, h) if a == b)
        kappa_by_tier[tier] = {
            "n": len(t_rows), "kappa": round(k, 4),
            "agreement": round(n_agr / len(t_rows), 4)
        }
        print(f"  {tier}: κ={k:.4f} (n={len(t_rows)}, agree={n_agr/len(t_rows):.1%})")

    # Per-class agreement
    per_class = {}
    for label in VALID_STATUSES:
        tp = sum(1 for s, h in zip(scorer_labels, human_labels) if s == label and h == label)
        fp = sum(1 for s, h in zip(scorer_labels, human_labels) if s == label and h != label)
        fn = sum(1 for s, h in zip(scorer_labels, human_labels) if s != label and h == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        print(f"  {label}: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

    # Disagreement analysis
    disagreements = [r for r in rows if r["scorer_status"] != r["human_status"]]
    disagree_types = defaultdict(int)
    for r in disagreements:
        disagree_types[f"{r['scorer_status']}→{r['human_status']}"] += 1

    # Save stats JSON
    stats = {
        "n_annotations": len(rows),
        "n_prompts": len(set(r["prompt_id"] for r in rows)),
        "kappa_overall": round(kappa_overall, 4),
        "agreement_overall": round(p_agree, 4),
        "threshold": 0.65,
        "passed_threshold": passed,
        "kappa_by_domain": kappa_by_domain,
        "kappa_by_tier": kappa_by_tier,
        "per_class_metrics": per_class,
        "n_disagreements": len(disagreements),
        "disagreement_types": dict(disagree_types),
        "disagreement_rate": round(len(disagreements) / len(rows), 4)
    }

    stats_path = OUTPUT_DIR / "agreement_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  → Stats: {stats_path}")

    # Disagreement analysis markdown
    md_lines = [
        "# Exp3 — Disagreement Analysis",
        "",
        f"**Cohen's κ overall: {kappa_overall:.4f}** "
        f"({'PASSED ✓' if passed else 'FAILED ✗'} — threshold κ ≥ 0.65)",
        f"- Total annotations: {len(rows)}",
        f"- Agreements: {n_agree} ({p_agree:.1%})",
        f"- Disagreements: {len(disagreements)} ({1-p_agree:.1%})",
        "",
        "## Disagreement Type Breakdown",
        "",
        "| Type | Count | Rate |",
        "|------|-------|------|",
    ]
    for dtype, count in sorted(disagree_types.items(), key=lambda x: -x[1]):
        md_lines.append(f"| {dtype} | {count} | {count/len(rows):.1%} |")

    md_lines += ["", "## Examples of Disagreements", ""]
    for r in disagreements[:20]:  # show up to 20
        md_lines.append(
            f"- **{r['prompt_id']} / {r['param_name']}** "
            f"[{r['domain']}, {r['vagueness']}]: "
            f"Scorer={r['scorer_status']}, Human={r['human_status']}  "
            f"  Evidence: _{r['scorer_evidence'][:80]}_"
        )

    md_lines += [
        "",
        "## Analysis by Domain",
        "",
        "| Domain | n | κ | Agreement | Passed? |",
        "|--------|---|---|-----------|---------|",
    ]
    for domain, d in kappa_by_domain.items():
        md_lines.append(
            f"| {domain} | {d['n']} | {d['kappa']:.3f} | "
            f"{d['agreement']:.1%} | {'✓' if d['passed'] else '✗'} |"
        )

    md_lines += [
        "",
        "## Analysis by Annotation Tier",
        "",
        "| Tier | n | κ | Agreement |",
        "|------|---|---|-----------|",
    ]
    for tier, t in kappa_by_tier.items():
        md_lines.append(f"| {tier} | {t['n']} | {t['kappa']:.3f} | {t['agreement']:.1%} |")

    md_lines += [
        "",
        "## Interpretation",
        "",
        "- κ < 0.40: Poor agreement — scorer not reliable",
        "- 0.40 ≤ κ < 0.60: Moderate — acceptable with caveats",
        "- 0.60 ≤ κ < 0.80: Substantial — suitable for research use",
        "- κ ≥ 0.80: Almost perfect — publication-ready",
        "",
        "**Key observations:**",
        "- RESOLVED vs MENTIONED confusion is expected: the boundary requires ",
        "  interpreting whether a value is 'concrete enough'",
        "- ABSENT misses are more problematic (scorer seeing things not there)",
        "- Domain-specific analysis helps identify taxonomy weaknesses",
    ]

    md_path = OUTPUT_DIR / "disagreement_analysis.md"
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines))
    print(f"  → Analysis: {md_path}")

    print(f"\n{'='*60}")
    print(f"FINAL: κ = {kappa_overall:.4f} — {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
