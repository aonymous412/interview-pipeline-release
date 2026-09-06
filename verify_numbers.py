#!/usr/bin/env python3
"""Recompute every headline number of the paper from the stored run records.

    python verify_numbers.py

Needs only numpy. No API key, no model download, no re-run: each result file
stores the resolved frame and the coverage trajectory of every run, so all
reported means, bootstrap intervals, turn counts and stopping statistics are
recomputable offline. Printed alongside each value is what the paper claims,
so a mismatch is visible without cross-referencing the PDF.

The first section is an integrity pass rather than a comparison against the
paper. It recomputes each run's coverage from its stored frame and the published
weight vector, and checks the trajectories and stopping labels against the
thresholds. A reviewer who doubts the headline gap can start there: it establishes
that the coverage values are arithmetic on the frames, not figures asserted by the
pipeline, before any mean over them is reported.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parent
ALI = ROOT / "results" / "ali"
BASE = ROOT / "results" / "baselines"
SCORE = ROOT / "results" / "scoring"

B, SEED = 10_000, 42
ok = True


def load(p):
    d = json.loads(pathlib.Path(p).read_text())
    if isinstance(d, dict):
        d = d.get("results", list(d.values()))
    return d


def ci(x):
    """Percentile bootstrap, B=10,000, seed 42 — the paper's canonical method."""
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def check(label, got, want, tol=0.05):
    global ok
    hit = abs(got - want) <= tol
    ok &= hit
    print(f"  {'OK ' if hit else 'XX '} {label:<44} {got:7.2f}   paper: {want}")


def integrity(label, passed, detail=""):
    global ok
    ok &= passed
    print(f"  {'OK ' if passed else 'XX '} {label:<44} {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Record integrity. Run first, because every number below is a statistic over
# these records: if a coverage value did not follow from the frame and the
# weights, no mean computed from it would mean anything. Each run stores the
# resolved frame, so coverage is not taken on trust -- it is recomputed here from
# F_resolved and the published weight vector, independently of the pipeline.
# ─────────────────────────────────────────────────────────────────────────────
print("=== Record integrity (coverage recomputed from the frame) ===")
TAX = json.loads((ROOT / "data" / "taxonomies.json").read_text())
DOM = {"telos": "Consulting", "rad_assist": "Medical", "finagent": "Payments"}
THETA = {"telos": 0.90, "rad_assist": 0.90, "finagent": 0.75}

for d in ("Consulting", "Medical", "Payments"):
    t = TAX[d]
    s = sum(p["weight"] for p in t["parameters"].values())
    integrity(f"{d}: sum(w) == total_weight",
              s == t["total_weight"], f"{s} == {t['total_weight']}")

worst, mism, stray, mono, tail, cap, theta_v, n_tot = 0.0, 0, 0, 0, 0, 0, 0, 0
for name in ("ALI_v1", "ALI_v2", "ALI_v3"):
    for x in load(ALI / f"results_{name}.json"):
        n_tot += 1
        t = TAX[DOM[x["deployment"]]]
        P, tw = t["parameters"], t["total_weight"]
        stray += sum(1 for k in x["F_resolved"] if k not in P)
        got = sum(P[k]["weight"] for k in x["F_resolved"] if k in P) / tw
        e = abs(got - x["final_cov"])
        worst = max(worst, e)
        mism += e > 1e-9
        c = x["cov_by_turn"]
        mono += any(b < a - 1e-9 for a, b in zip(c, c[1:]))
        tail += abs(c[-1] - x["final_cov"]) > 1e-9 or len(c) != x["n_turns"] + 1
        if x["stopped_by"] == "coverage_reached":
            theta_v += x["final_cov"] < THETA[x["deployment"]] - 1e-9
        else:
            cap += x["n_turns"] != 15

integrity(f"cov == sum(w_resolved)/sum(w), {n_tot} runs", mism == 0, f"max err {worst:.1e}")
integrity("no frame key outside the parameter space", stray == 0)
integrity("trajectories non-decreasing", mono == 0)
integrity("cov_by_turn[-1] == final_cov, len == turns+1", tail == 0)
integrity("every 'coverage_reached' run has cov >= theta_d", theta_v == 0)
integrity("every other run stopped at the 15-turn cap", cap == 0)

print("\n=== Table: all evaluated systems (coverage %) ===")
for name, want in [("ALI_v1", 89.0), ("ALI_v2", 89.4), ("ALI_v3", 51.6)]:
    r = load(ALI / f"results_{name}.json")
    cov = [x["final_cov"] for x in r]
    lo, hi = ci(cov)
    check(f"{name} (n={len(r)}) [{lo*100:.1f}-{hi*100:.1f}]", np.mean(cov) * 100, want)

for f, want in [("claude-opus-4-8", 58.4), ("claude-haiku-4-5", 57.1), ("claude-sonnet-5", 59.9),
                ("gemini-2.5-flash", 52.8), ("gemma3_4b", 42.2), ("llama3.1_8b", 47.5)]:
    p = BASE / f"results_{f}_multiturn_full.json"
    if not p.exists():
        continue
    r = load(p)
    cov = [x["final_cov"] for x in r]
    lo, hi = ci(cov)
    check(f"{f} (n={len(r)}) [{lo*100:.1f}-{hi*100:.1f}]", np.mean(cov) * 100, want)

sp = json.loads((BASE / "baseline_extended_stats.json").read_text())
check("single-pass Gemini 2.5 Flash (n=150)", sp["overall"]["mean"] * 100, 40.7)

print("\n=== Headline gap ===")
ali = np.mean([x["final_cov"] for x in load(ALI / "results_ALI_v1.json")])
opus = np.mean([x["final_cov"] for x in load(BASE / "results_claude-opus-4-8_multiturn_full.json")])
check("ALI_v1 - Claude Opus 4.8 (pp)", (ali - opus) * 100, 30.6, tol=0.15)

print("\n=== ALI_v1 by domain ===")
r = load(ALI / "results_ALI_v1.json")
for dep, label, want, wturns in [("telos", "Consulting", 92.7, 8.52),
                                 ("rad_assist", "Medical", 90.6, 8.28),
                                 ("finagent", "Payments", 83.9, 3.98)]:
    s = [x for x in r if x["deployment"] == dep]
    if not s:
        continue
    check(f"{label} coverage", np.mean([x["final_cov"] for x in s]) * 100, want)
    check(f"{label} turns", np.mean([x["n_turns"] for x in s]), wturns, tol=0.02)

print("\n=== Stopping rule ===")
for name, want in [("ALI_v1", 97.3), ("ALI_v2", 96.7), ("ALI_v3", 30.0)]:
    r = load(ALI / f"results_{name}.json")
    n = sum(1 for x in r if x["stopped_by"] == "coverage_reached")
    check(f"{name} stopped by cov>=theta (%)", 100 * n / len(r), want, tol=0.1)

print("\n=== 8-turn truncation control ===")
r = load(ALI / "results_ALI_v1.json")
# cov_by_turn[0] is coverage after C0 pre-extraction, so index k is coverage after k turns
tr = [x["cov_by_turn"][min(8, len(x["cov_by_turn"]) - 1)] for x in r]
lo, hi = ci(tr)
check(f"ALI_v1 truncated at 8 turns [{lo*100:.1f}-{hi*100:.1f}]", np.mean(tr) * 100, 84.4)
check("turns under truncation", np.mean([min(x["n_turns"], 8) for x in r]), 5.95, tol=0.02)

print("\n=== Ablation (n=30, paired) ===")
for v, want in [("full", 90.3), ("no_c2", 89.9), ("no_priority", 90.2),
                ("c1_random", 90.2), ("c3_tier1_only", 90.4)]:
    p = ALI / f"results_ablation_{v}.json"
    if p.exists():
        check(v, np.mean([x["final_cov"] for x in load(p)]) * 100, want)

print("\n=== Scorer validation ===")
a = json.loads((SCORE / "agreement_stats.json").read_text())
check(f"Cohen's kappa (n={a['n_annotations']} annotations)", a["kappa_overall"], 0.749, tol=0.001)
for d, want in [("Consulting", 0.758), ("Medical", 0.968), ("Payments", 0.539)]:
    check(f"  kappa {d}", a["kappa_by_domain"][d]["kappa"], want, tol=0.001)

print("\n" + ("ALL CHECKS PASSED" if ok else "MISMATCHES ABOVE — see lines marked XX"))
sys.exit(0 if ok else 1)
