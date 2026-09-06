"""
Phase 3 — Volet 5: Frontier Baseline (Claude via Anthropic API)
================================================================
Runs the same multi-turn interview protocol as Experiment 1
(run_multiturn.py) with a Claude frontier model as the interviewer,
on the FULL 150-prompt set of Exp2 (50 from config/prompts.json +
100 from exp2_baseline_extended/prompts_new_100.csv).

Protocol identity with Exp1 is preserved by importing the system prompts,
constants and phase-runner directly from run_multiturn.py:
  - Same INTERVIEWER_SYSTEM / ORACLE_SYSTEM prompts
  - Same turn-pressure reminders after turns 5 and 8 (as in the Gemini
    interviewer variant), MAX_TURNS = 15
  - Same scorer: gemini-2.5-flash with retry/backoff

Deviations from Exp1, all documented for the paper:
  - n=150 (Exp2 prompt set) instead of n=50
  - Oracle is claude-haiku-4-5 via API instead of gemma3:4b via Ollama
    (no local dependency; the scorer remains Gemini, so the evaluator
    is still from a different model family than the interviewer)
  - No temperature parameter (removed on Opus 4.7+; prior runs used 0.7)
  - Adaptive thinking enabled on the interviewer

Budget safety: token usage for both models is accumulated from
response.usage and the run aborts cleanly (results saved per-prompt,
resume supported) when estimated spend reaches --max-cost.

Requires: ANTHROPIC_API_KEY in env; Gemini key in experiments/.env
(read by scoring.get_gemini_client).

Usage:
  python run_multiturn_claude.py --mode pilot          # 5 prompts
  python run_multiturn_claude.py --mode full           # 150 prompts
  python run_multiturn_claude.py --mode full --effort medium
"""

import csv
import json
import time
from pathlib import Path

import anthropic

from run_multiturn import (
    DOMAIN_TASKS,
    INTERVIEWER_SYSTEM,
    MAX_TURNS,
    ORACLE_SYSTEM,
    SEED,
    _check_done,
    compute_summary,
    run_phase,
)
from scoring import get_gemini_client, score_conversation

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recomputed"  # [relocated]
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"  # [relocated]
EXP2_DIR = Path(__file__).resolve().parents[2] / "data"  # [relocated]

ORACLE_MODEL = "claude-haiku-4-5"

# USD per 1M tokens (input, output) — Anthropic pricing, July 2026
PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def load_prompts_150() -> list:
    """The Exp2 prompt set: 50 originals (JSON) + 100 extension (CSV)."""
    with open(CONFIG_DIR / "prompts_P01-P50.json") as f:
        prompts = json.load(f)
    with open(EXP2_DIR / "prompts_N01-N100.csv", newline="") as f:
        for row in csv.DictReader(f):
            prompts.append({
                "id": row["id"],
                "domain": row["domain"],
                "vagueness": row["vagueness"],
                "text": row["text"],
            })
    assert len(prompts) == 150, f"Expected 150 prompts, got {len(prompts)}"
    return prompts


class CostTracker:
    """Accumulates token usage per model and enforces a USD cap."""

    def __init__(self, max_cost_usd: float):
        self.max_cost_usd = max_cost_usd
        self.usage = {}  # model -> [input_tokens, output_tokens]

    def add(self, model: str, usage) -> None:
        u = self.usage.setdefault(model, [0, 0])
        u[0] += usage.input_tokens
        u[1] += usage.output_tokens

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for model, (tin, tout) in self.usage.items():
            in_p, out_p = PRICING.get(model, (5.00, 25.00))
            total += (tin * in_p + tout * out_p) / 1e6
        return total

    def check(self) -> None:
        if self.cost_usd >= self.max_cost_usd:
            print(f"\n[BUDGET] Estimated spend ${self.cost_usd:.2f} >= "
                  f"cap ${self.max_cost_usd:.2f} — stopping cleanly. "
                  f"Completed prompts are saved; rerun to resume.")
            raise SystemExit(1)

    def report(self) -> str:
        parts = [f"{m}: in={u[0]:,} out={u[1]:,}" for m, u in self.usage.items()]
        return f"{' | '.join(parts)} | est. ${self.cost_usd:.2f}"

    def to_dict(self) -> dict:
        return {
            "per_model_tokens": {m: {"input": u[0], "output": u[1]}
                                 for m, u in self.usage.items()},
            "estimated_usd": round(self.cost_usd, 2),
        }


def _claude_call_with_retry(client, cost: CostTracker, label: str,
                            n_attempts: int = 5, **kwargs):
    """messages.create with backoff on rate limits / overloads / 5xx.

    The SDK already retries twice internally; this outer loop covers longer
    outages so the run survives unattended overnight.
    """
    delays = [15, 60, 120, 300]
    for attempt in range(n_attempts):
        try:
            response = client.messages.create(**kwargs)
            cost.add(kwargs["model"], response.usage)
            return response
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:
            if attempt == n_attempts - 1:
                raise
            d = delays[min(attempt, len(delays) - 1)]
            print(f"    [RETRY {label}] {type(e).__name__}, waiting {d}s "
                  f"(attempt {attempt + 1}/{n_attempts})")
            time.sleep(d)


def _score_with_retry(conversation, domain, prompt_text, gemini_client,
                      n_attempts: int = 4):
    """Gemini scorer with backoff. No Ollama fallback (no local models)."""
    delays = [15, 45, 90]
    for attempt in range(n_attempts):
        try:
            return score_conversation(
                conversation, domain, prompt_text,
                gemini_client=gemini_client, scorer="gemini"
            ), False
        except Exception as e:
            if attempt == n_attempts - 1:
                print(f"    [SCORING FAILED after {n_attempts} attempts] {e}")
                return {"coverage": 0.0, "params_resolved": [],
                        "params_mentioned": [], "params_absent": []}, True
            d = delays[min(attempt, len(delays) - 1)]
            print(f"    [RETRY scorer] {e} — waiting {d}s")
            time.sleep(d)


def run_multiturn_claude(prompt_data: dict, anthropic_client, cost: CostTracker,
                         model: str = "claude-opus-4-8",
                         effort: str = None,
                         thinking: str = "adaptive",
                         gemini_client=None) -> dict:
    """Multi-turn interview: Claude interviewer + Claude Haiku oracle."""
    cost.check()

    domain = prompt_data["domain"]
    domain_task = DOMAIN_TASKS[domain]
    interviewer_system = INTERVIEWER_SYSTEM.format(domain_task=domain_task)
    oracle_system = ORACLE_SYSTEM.format(
        domain_task=domain_task, original_prompt=prompt_data["text"]
    )

    interviewer_kwargs = {
        "model": model,
        "max_tokens": 2000,  # headroom for adaptive thinking + reply
        "system": interviewer_system,
    }
    if thinking == "adaptive":
        interviewer_kwargs["thinking"] = {"type": "adaptive"}
    if effort:
        interviewer_kwargs["output_config"] = {"effort": effort}

    # API-side message history (keeps thinking blocks intact between turns)
    api_msgs = []
    oracle_msgs = []
    # Scored conversation log (text only, same shape as other variants)
    conversation = [{"role": "user", "content": prompt_data["text"]}]
    stopped_by = "max_turns"
    current_user_msg = prompt_data["text"]

    for turn in range(MAX_TURNS):
        # Same turn-pressure as the Gemini interviewer variant of Exp1
        if turn >= 8:
            msg_to_send = (
                f"{current_user_msg}\n\n"
                f"[REMINDER: You have asked {turn} questions. "
                f"You MUST output [DONE] now — do not ask another question.]"
            )
        elif turn >= 5:
            msg_to_send = (
                f"{current_user_msg}\n\n"
                f"[REMINDER: You have asked {turn} questions. "
                f"Wrap up and output [DONE] soon.]"
            )
        else:
            msg_to_send = current_user_msg

        api_msgs.append({"role": "user", "content": msg_to_send})

        try:
            response = _claude_call_with_retry(
                anthropic_client, cost, "interviewer",
                messages=api_msgs, **interviewer_kwargs,
            )
        except Exception as e:
            print(f"    [ERROR] Interviewer turn {turn}: {e}")
            stopped_by = f"error_turn_{turn}"
            break

        # Echo full content back next turn (thinking blocks unchanged)
        api_msgs.append({"role": "assistant", "content": response.content})

        interviewer_text = "\n".join(
            b.text for b in response.content if b.type == "text"
        ).strip()
        conversation.append({"role": "assistant", "content": interviewer_text})

        if _check_done(interviewer_text):
            stopped_by = "DONE"
            break

        cost.check()

        # Oracle turn — Claude Haiku plays the user
        oracle_msgs.append({
            "role": "user",
            "content": f"The interviewer asks:\n{interviewer_text}\n\nRespond as the user."
        })
        try:
            oracle_resp = _claude_call_with_retry(
                anthropic_client, cost, "oracle",
                model=ORACLE_MODEL,
                max_tokens=300,
                system=oracle_system,
                messages=oracle_msgs,
            )
        except Exception as e:
            print(f"    [ERROR] Oracle turn {turn}: {e}")
            stopped_by = f"oracle_error_turn_{turn}"
            break

        oracle_text = "\n".join(
            b.text for b in oracle_resp.content if b.type == "text"
        ).strip()
        conversation.append({"role": "user", "content": oracle_text})
        oracle_msgs.append({"role": "assistant", "content": oracle_text})
        current_user_msg = oracle_text

    # Score with the same scorer as every other system
    time.sleep(4)  # Gemini free-tier rate limit
    final_score, scoring_failed = _score_with_retry(
        conversation, domain, prompt_data["text"], gemini_client
    )

    n_turns = len([m for m in conversation if m["role"] == "assistant"])
    print(f"    [{cost.report()}]")

    return {
        "prompt_id": prompt_data["id"],
        "domain": domain,
        "vagueness": prompt_data["vagueness"],
        "model": model,
        "model_type": "anthropic_api",
        "n_turns": n_turns,
        "final_cov": final_score["coverage"],
        "stopped_by": stopped_by,
        "scoring_failed": scoring_failed,
        "params_resolved": final_score.get("params_resolved", []),
        "params_mentioned": final_score.get("params_mentioned", []),
        "params_absent": final_score.get("params_absent", []),
        "n_resolved": len(final_score.get("params_resolved", [])),
        "n_mentioned": len(final_score.get("params_mentioned", [])),
        "n_absent": len(final_score.get("params_absent", [])),
        "conversation": conversation,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3 Volet 5: Claude frontier multi-turn baseline (n=150)"
    )
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot=5 prompts, full=150 prompts (Exp2 set)")
    parser.add_argument("--n", type=int, default=None,
                        help="Override number of prompts")
    parser.add_argument("--model", default="claude-opus-4-8",
                        choices=list(PRICING),
                        help="Claude model to use as interviewer (using the "
                             "oracle model as interviewer is allowed for the "
                             "scaling-curve runs; the scorer stays Gemini)")
    parser.add_argument("--effort", default=None,
                        choices=["low", "medium", "high"],
                        help="Optional output_config.effort for the interviewer")
    parser.add_argument("--thinking", default="adaptive",
                        choices=["adaptive", "off"],
                        help="Interviewer thinking mode (off = no thinking "
                             "param, for models that reject adaptive thinking)")
    parser.add_argument("--max-cost", type=float, default=18.0,
                        help="Abort when estimated spend reaches this many USD")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing results")
    args = parser.parse_args()

    all_prompts = load_prompts_150()
    n = args.n or (5 if args.mode == "pilot" else 150)
    prompts = all_prompts[:n]

    label = f"{args.model}_multiturn"
    print(f"{'='*60}")
    print(f"FRONTIER BASELINE: {args.model} multi-turn interviewer")
    print(f"Mode: {args.mode} | Prompts: {len(prompts)}/150 | Seed: {SEED} | "
          f"Budget cap: ${args.max_cost:.2f}"
          + (f" | effort={args.effort}" if args.effort else ""))
    print(f"Oracle: {ORACLE_MODEL} (API) | Scorer: gemini-2.5-flash")
    print(f"{'='*60}\n")

    anthropic_client = anthropic.Anthropic()
    gemini_client = get_gemini_client()
    cost = CostTracker(args.max_cost)

    budget_stopped = False
    try:
        results = run_phase(
            label, prompts,
            run_multiturn_claude, OUTPUT_DIR,
            resume=not args.no_resume,
            anthropic_client=anthropic_client,
            cost=cost,
            model=args.model,
            effort=args.effort,
            thinking=args.thinking,
            gemini_client=gemini_client,
        )
    except SystemExit:
        # Budget cap hit mid-run: per-prompt results are already saved.
        # Reload them so the summary (and the actual spend) still gets
        # written — the night orchestrator relies on it for accounting.
        budget_stopped = True
        json_path = OUTPUT_DIR / f"results_{label}_full.json"
        results = json.load(open(json_path)) if json_path.exists() else []

    summary = compute_summary({label: results})
    summary["budget_stopped"] = budget_stopped
    summary["interviewer_api"] = "anthropic"
    summary["oracle"] = f"{ORACLE_MODEL} (Anthropic API)"
    summary["thinking"] = args.thinking + (f", effort={args.effort}" if args.effort else "")
    summary["prompt_set"] = "Exp2 150-prompt set (50 json + 100 csv)"
    summary["n_scoring_failed"] = sum(1 for r in results if r.get("scoring_failed"))
    summary["cost_estimate"] = cost.to_dict()
    summary_path = OUTPUT_DIR / f"results_{label}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → Summary: {summary_path}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    for lbl, data in summary.get("models", {}).items():
        o = data["overall"]
        print(f"\n{lbl}:")
        print(f"  Overall: {o['mean_cov']*100:.1f}% ± {o['std_cov']*100:.1f}% | "
              f"Turns: {o['mean_turns']:.1f} | Done: {o['done_rate']*100:.0f}%")
        for domain, d in data.get("by_domain", {}).items():
            print(f"  {domain}: {d['mean_cov']*100:.1f}% ± {d['std_cov']*100:.1f}%")
    print(f"\nReference points: gemini-2.5-flash multi-turn = 52.8% (n=50) | "
          f"single-pass = 40.7% (n=150) | ALI_v1 = 89.0% (n=150)")
    print(f"Total {cost.report()}")


if __name__ == "__main__":
    main()
