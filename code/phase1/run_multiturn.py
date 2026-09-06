"""
Experiment 1: Multi-Turn LLM Comparison
========================================
Compares multi-turn conversational LLMs against the single-pass baseline
and ALI simulation targets.

Models tested:
  - llama3.1:8b (via Ollama, local) — multi-turn interviewer
  - gemma3:4b   (via Ollama, local) — multi-turn interviewer
  - gemini-2.5-flash (via API, free tier) — multi-turn interviewer

Oracle simulator:
  - For llama3.1:8b interviewer → gemma3:4b oracle
  - For gemma3:4b interviewer   → llama3.1:8b oracle
  - For gemini interviewer       → gemma3:4b oracle (Ollama local)

Scorer: gemini-2.5-flash (Gemini API, free tier) with Ollama fallback

Protocol:
  - 50 prompts × each model
  - Multi-turn: LLM interviews, oracle answers, up to MAX_TURNS turns
  - Coverage scored by cov(F) at conversation end
  - Resume mode: skip prompts already in existing CSV

Author: (anonymised for review)
Date: 2026-06-28 (updated 2026-06-29)
Seed: 42
"""

import json
import csv
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # [relocated]
from scoring import (
    score_conversation, score_response, get_gemini_client,
    call_ollama_chat, call_gemini
)
from google import genai

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_TURNS = 15
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "recomputed"  # [relocated]
OUTPUT_DIR.mkdir(exist_ok=True)

DOMAIN_TASKS = {
    "Consulting": "scope and build a software project",
    "Medical": "generate a complete clinical oncology report",
    "Payments": "configure an autonomous financial trading/payment agent safely"
}

# Stronger [DONE] instruction — model MUST stop after 5-8 questions
INTERVIEWER_SYSTEM = """You are an expert requirements interviewer. Your task is to gather the information needed to {domain_task} from the user.

MANDATORY RULES — FOLLOW EXACTLY:
1. Ask EXACTLY ONE focused question per message — never a list of questions
2. Do NOT ask for information already provided by the user
3. Prioritize the most critical missing information first
4. YOU MUST OUTPUT [DONE] AFTER 5 TO 8 QUESTIONS — this is not optional
5. When you have gathered sufficient key requirements, write [DONE] on its own line, exactly like this:
   [DONE]
6. After [DONE], write a short bullet-point summary of what was gathered

CRITICAL: You are required to stop and output [DONE] after 5-8 questions maximum.
Do NOT ask more than 8 questions under any circumstances. Quality over quantity."""

ORACLE_SYSTEM = """You are a user who needs to {domain_task}. Your original request was:

"{original_prompt}"

RULES:
1. Answer the interviewer's questions based on what a realistic user with this project/situation would know
2. Give natural, concise answers (1-3 sentences)
3. If asked about something you haven't thought about, give a reasonable answer
4. If asked something very technical, say "I'm not sure" or give a vague preference
5. Stay consistent with previous answers
6. Don't volunteer extra information beyond what was asked
7. Be realistic — not every user knows exact budgets, timelines, or technical details"""


def _check_done(text: str) -> bool:
    """Robust detection of conversation termination signal."""
    # Primary: exact token as instructed
    if "[DONE]" in text:
        return True
    # Secondary: common variations the model might produce
    text_up = text.upper()
    markers = ["[ DONE ]", "[DONE]\n", "\nDONE\n", "**[DONE]**", "**DONE**"]
    return any(m in text_up for m in markers)


def load_existing_results(csv_path: Path) -> set:
    """Return set of prompt_ids already recorded in an existing CSV."""
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add(row["prompt_id"])
    return done


def run_multiturn_ollama(prompt_data: dict, model: str = "llama3.1:8b",
                         oracle_model: str = "gemma3:4b",
                         gemini_client=None,
                         min_turns: int = 4) -> dict:
    """Run multi-turn conversation using Ollama local models.

    Supports any Ollama interviewer model. Oracle model should differ from
    the interviewer to avoid self-evaluation bias.

    min_turns: minimum number of interviewer turns before [DONE] is accepted.
    Prevents small models (e.g. gemma3:4b) from terminating too early.
    """
    domain = prompt_data["domain"]
    domain_task = DOMAIN_TASKS[domain]

    interviewer_msgs = [
        {"role": "system", "content": INTERVIEWER_SYSTEM.format(domain_task=domain_task)},
        {"role": "user", "content": prompt_data["text"]}
    ]

    oracle_msgs = [
        {"role": "system", "content": ORACLE_SYSTEM.format(
            domain_task=domain_task, original_prompt=prompt_data["text"]
        )}
    ]

    conversation = [{"role": "user", "content": prompt_data["text"]}]
    stopped_by = "max_turns"

    for turn in range(MAX_TURNS):
        # Interviewer turn
        try:
            interviewer_text = call_ollama_chat(
                interviewer_msgs, model=model, temperature=0.7, max_tokens=500
            )
        except Exception as e:
            print(f"    [ERROR] Interviewer turn {turn}: {e}")
            stopped_by = f"error_turn_{turn}"
            break

        conversation.append({"role": "assistant", "content": interviewer_text})
        interviewer_msgs.append({"role": "assistant", "content": interviewer_text})

        # Only accept [DONE] after min_turns to prevent premature termination
        if _check_done(interviewer_text) and turn >= min_turns - 1:
            stopped_by = "DONE"
            break

        # Oracle turn — up to 3 attempts before giving up
        oracle_msgs.append({
            "role": "user",
            "content": f"The interviewer asks:\n{interviewer_text}\n\nRespond as the user."
        })

        oracle_text = None
        for attempt in range(3):
            try:
                oracle_text = call_ollama_chat(
                    oracle_msgs, model=oracle_model, temperature=0.7, max_tokens=300
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [ERROR] Oracle turn {turn} (attempt {attempt+1}): {e}")
                    stopped_by = f"oracle_error_turn_{turn}"
                else:
                    time.sleep(2)

        if oracle_text is None:
            break

        conversation.append({"role": "user", "content": oracle_text})
        interviewer_msgs.append({"role": "user", "content": oracle_text})
        oracle_msgs.append({"role": "assistant", "content": oracle_text})

    # Score the final conversation
    try:
        final_score = score_conversation(
            conversation, domain, prompt_data["text"],
            gemini_client=gemini_client, scorer="gemini"
        )
    except Exception as e:
        print(f"    [SCORING ERROR] {e}, trying ollama fallback...")
        try:
            final_score = score_conversation(
                conversation, domain, prompt_data["text"], scorer="ollama"
            )
        except Exception:
            final_score = {"coverage": 0.0, "params_resolved": [],
                           "params_mentioned": [], "params_absent": []}

    n_turns = len([m for m in conversation if m["role"] == "assistant"])

    return {
        "prompt_id": prompt_data["id"],
        "domain": domain,
        "vagueness": prompt_data["vagueness"],
        "model": model,
        "model_type": "ollama_local",
        "n_turns": n_turns,
        "final_cov": final_score["coverage"],
        "stopped_by": stopped_by,
        "params_resolved": final_score.get("params_resolved", []),
        "params_mentioned": final_score.get("params_mentioned", []),
        "params_absent": final_score.get("params_absent", []),
        "n_resolved": len(final_score.get("params_resolved", [])),
        "n_mentioned": len(final_score.get("params_mentioned", [])),
        "n_absent": len(final_score.get("params_absent", [])),
        "conversation": conversation
    }


def run_multiturn_gemini(prompt_data: dict, gemini_client,
                          model: str = "gemini-2.5-flash",
                          oracle_model: str = "gemma3:4b") -> dict:
    """Run multi-turn conversation with Gemini as interviewer using stateful Chat API.

    Fix (2026-06-29): replaced the stateless prompt-rebuild approach with
    client.chats.create() so Gemini maintains proper conversational state
    and follows the system instruction reliably. Also added turn-pressure
    after turn 5 to enforce [DONE] termination.
    """
    domain = prompt_data["domain"]
    domain_task = DOMAIN_TASKS[domain]

    system_instruction = INTERVIEWER_SYSTEM.format(domain_task=domain_task)

    oracle_msgs = [
        {"role": "system", "content": ORACLE_SYSTEM.format(
            domain_task=domain_task, original_prompt=prompt_data["text"]
        )}
    ]

    # Stateful Gemini chat session — system instruction sent once, state maintained
    chat = gemini_client.chats.create(
        model=model,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            max_output_tokens=500,
        )
    )

    conversation = [{"role": "user", "content": prompt_data["text"]}]
    stopped_by = "max_turns"

    # First message is the user's initial prompt
    current_user_msg = prompt_data["text"]

    for turn in range(MAX_TURNS):
        # Add pressure after turn 5 to enforce [DONE]
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

        try:
            response = chat.send_message(msg_to_send)
            interviewer_text = response.text
            time.sleep(4)  # Rate limit: 15 RPM free tier
        except Exception as e:
            print(f"    [ERROR] Gemini turn {turn}: {e}")
            stopped_by = f"error_turn_{turn}"
            time.sleep(10)
            break

        conversation.append({"role": "assistant", "content": interviewer_text})

        if _check_done(interviewer_text):
            stopped_by = "DONE"
            break

        # Oracle turn — up to 3 attempts
        oracle_msgs.append({
            "role": "user",
            "content": f"The interviewer asks:\n{interviewer_text}\n\nRespond as the user."
        })

        oracle_text = None
        for attempt in range(3):
            try:
                oracle_text = call_ollama_chat(
                    oracle_msgs, model=oracle_model, temperature=0.7, max_tokens=300
                )
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [ERROR] Oracle turn {turn} (attempt {attempt+1}): {e}")
                    stopped_by = f"oracle_error_turn_{turn}"
                else:
                    time.sleep(2)

        if oracle_text is None:
            break

        conversation.append({"role": "user", "content": oracle_text})
        oracle_msgs.append({"role": "assistant", "content": oracle_text})
        current_user_msg = oracle_text  # Next message to Gemini is the oracle response

    # Score
    try:
        time.sleep(4)
        final_score = score_conversation(
            conversation, domain, prompt_data["text"],
            gemini_client=gemini_client, scorer="gemini"
        )
    except Exception as e:
        print(f"    [SCORING ERROR] {e}, trying ollama fallback...")
        try:
            final_score = score_conversation(
                conversation, domain, prompt_data["text"], scorer="ollama"
            )
        except Exception:
            final_score = {"coverage": 0.0, "params_resolved": [],
                           "params_mentioned": [], "params_absent": []}

    n_turns = len([m for m in conversation if m["role"] == "assistant"])

    return {
        "prompt_id": prompt_data["id"],
        "domain": domain,
        "vagueness": prompt_data["vagueness"],
        "model": model,
        "model_type": "gemini_api",
        "n_turns": n_turns,
        "final_cov": final_score["coverage"],
        "stopped_by": stopped_by,
        "params_resolved": final_score.get("params_resolved", []),
        "params_mentioned": final_score.get("params_mentioned", []),
        "params_absent": final_score.get("params_absent", []),
        "n_resolved": len(final_score.get("params_resolved", [])),
        "n_mentioned": len(final_score.get("params_mentioned", [])),
        "n_absent": len(final_score.get("params_absent", [])),
        "conversation": conversation
    }


def run_single_pass_gemini(prompt_data: dict, gemini_client,
                            model: str = "gemini-2.5-flash") -> dict:
    """Run single-pass extraction (reproducing §8 baseline)."""
    domain = prompt_data["domain"]

    system_prompt = (
        "You are a project requirements analyst. Extract all requirements "
        "from the following description and structure them into a complete, "
        "detailed specification. Be thorough — cover every aspect needed "
        "to implement this project/configure this system."
    )

    try:
        response_text = call_gemini(
            f"{system_prompt}\n\nUser request:\n{prompt_data['text']}",
            client=gemini_client, model=model, temperature=0.0, max_tokens=4000
        )
        time.sleep(4)
    except Exception as e:
        print(f"    [ERROR] Single-pass failed: {e}")
        return None

    try:
        time.sleep(4)
        score = score_response(
            response_text, domain, prompt_data["text"],
            gemini_client=gemini_client, scorer="gemini"
        )
    except Exception as e:
        print(f"    [SCORING ERROR] {e}")
        score = {"coverage": 0.0, "params_resolved": [],
                 "params_mentioned": [], "params_absent": []}

    return {
        "prompt_id": prompt_data["id"],
        "domain": domain,
        "vagueness": prompt_data["vagueness"],
        "model": model,
        "model_type": "gemini_single_pass",
        "n_turns": 1,
        "final_cov": score["coverage"],
        "stopped_by": "single_pass",
        "params_resolved": score.get("params_resolved", []),
        "params_mentioned": score.get("params_mentioned", []),
        "params_absent": score.get("params_absent", []),
        "n_resolved": len(score.get("params_resolved", [])),
        "n_mentioned": len(score.get("params_mentioned", [])),
        "n_absent": len(score.get("params_absent", [])),
        "response_text": response_text
    }


def save_results(results: list, label: str, output_dir: Path):
    """Save results to CSV and full JSON."""
    safe = label.replace("/", "_").replace(" ", "_").replace(":", "_")

    csv_path = output_dir / f"results_{safe}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "domain", "vagueness", "model", "model_type",
            "n_turns", "final_cov", "stopped_by",
            "n_resolved", "n_mentioned", "n_absent"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    print(f"  → CSV: {csv_path}")

    json_path = output_dir / f"results_{safe}_full.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  → JSON: {json_path}")


def compute_summary(all_results: dict) -> dict:
    """Compute summary statistics across all models."""
    import numpy as np

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "max_turns": MAX_TURNS,
        "scorer": "gemini-2.5-flash",
        "oracle": "gemma3:4b (Ollama) for llama/gemini; llama3.1:8b (Ollama) for gemma",
        "models": {}
    }

    for label, results in all_results.items():
        valid = [r for r in results if r.get("final_cov") is not None]
        if not valid:
            continue
        covs = np.array([r["final_cov"] for r in valid])
        turns = np.array([r["n_turns"] for r in valid])

        model_data = {
            "n_total": len(results),
            "n_valid": len(valid),
            "overall": {
                "mean_cov": round(float(np.mean(covs)), 4),
                "std_cov": round(float(np.std(covs)), 4),
                "median_cov": round(float(np.median(covs)), 4),
                "min_cov": round(float(np.min(covs)), 4),
                "max_cov": round(float(np.max(covs)), 4),
                "mean_turns": round(float(np.mean(turns)), 1),
                "done_rate": round(
                    sum(1 for r in valid if r["stopped_by"] == "DONE") / len(valid), 3
                ),
            },
            "by_domain": {},
            "by_vagueness": {}
        }

        for domain in ["Consulting", "Medical", "Payments"]:
            dr = [r for r in valid if r["domain"] == domain]
            if dr:
                dc = np.array([r["final_cov"] for r in dr])
                dt = np.array([r["n_turns"] for r in dr])
                model_data["by_domain"][domain] = {
                    "n": len(dr),
                    "mean_cov": round(float(np.mean(dc)), 4),
                    "std_cov": round(float(np.std(dc)), 4),
                    "mean_turns": round(float(np.mean(dt)), 1),
                }

        for vag in ["Ultra-vague", "Medium", "Detailed"]:
            vr = [r for r in valid if r["vagueness"] == vag]
            if vr:
                vc = np.array([r["final_cov"] for r in vr])
                model_data["by_vagueness"][vag] = {
                    "n": len(vr),
                    "mean_cov": round(float(np.mean(vc)), 4),
                    "std_cov": round(float(np.std(vc)), 4),
                }

        summary["models"][label] = model_data

    return summary


def generate_latex_table(summary: dict, output_path: Path):
    """Generate comparison LaTeX table for the paper."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Coverage comparison: multi-turn local LLMs vs.\ multi-turn Gemini vs.\ ALI (simulation target). "
        r"All local models run via Ollama. Scorer: Gemini 2.5 Flash.}",
        r"\label{tab:multiturn_comparison}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{System} & \textbf{Type} & \textbf{Mean cov.} & \textbf{Std} "
        r"& \textbf{Turns} & \textbf{Done\%} \\",
        r"\midrule",
    ]

    for label, data in summary.get("models", {}).items():
        o = data["overall"]
        safe_label = label.replace("_", r"\_")
        model_type = "multi-turn" if o["mean_turns"] > 1.5 else "single-pass"
        lines.append(
            f"  {safe_label} & {model_type} & {o['mean_cov']*100:.1f}\\% & "
            f"{o['std_cov']*100:.1f}\\% & {o['mean_turns']:.1f} & "
            f"{o['done_rate']*100:.0f}\\% \\\\"
        )

    lines.extend([
        r"\midrule",
        r"  ALI pipeline & interview & $\sim$87\% & -- & $\sim$7 & 100\% \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  → LaTeX: {output_path}")


# ─── Phase runners ────────────────────────────────────────────────────────────

def run_phase(label: str, prompts: list, runner_fn, output_dir: Path,
              resume: bool = True, **runner_kwargs) -> list:
    """Generic phase runner with optional resume support.

    Loads existing results from CSV, skips already-done prompts,
    appends new results, and saves after each prompt.
    """
    safe = label.replace("/", "_").replace(" ", "_").replace(":", "_")
    csv_path = output_dir / f"results_{safe}.csv"
    json_path = output_dir / f"results_{safe}_full.json"

    # Load existing results if resuming
    existing_results = []
    done_ids = set()
    if resume and csv_path.exists():
        done_ids = load_existing_results(csv_path)
        if json_path.exists():
            with open(json_path) as f:
                existing_results = json.load(f)
        print(f"  Resume: {len(done_ids)} prompts already done, skipping.")

    results = list(existing_results)
    pending = [p for p in prompts if p["id"] not in done_ids]

    for i, p in enumerate(pending):
        idx = len(results) + 1
        total = len(results) + len(pending)
        print(f"\n  [{idx}/{total}] {p['id']} | {p['domain']} | {p['vagueness']}")
        print(f"    \"{p['text'][:70]}{'...' if len(p['text']) > 70 else ''}\"")

        try:
            result = runner_fn(p, **runner_kwargs)
            results.append(result)
            print(f"    → cov={result['final_cov']:.1%} | "
                  f"turns={result['n_turns']} | stop={result['stopped_by']}")
        except Exception as e:
            print(f"    [FATAL] {e}")
            traceback.print_exc()
            results.append({
                "prompt_id": p["id"], "domain": p["domain"],
                "vagueness": p["vagueness"], "model": label,
                "model_type": "unknown", "n_turns": 0,
                "final_cov": 0.0, "stopped_by": f"fatal: {str(e)[:80]}",
                "params_resolved": [], "params_mentioned": [], "params_absent": [],
                "n_resolved": 0, "n_mentioned": 0, "n_absent": 0,
                "conversation": []
            })

        save_results(results, label, output_dir)
        time.sleep(2)  # Gemini scorer rate limit buffer

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Experiment 1: Multi-Turn LLM Comparison"
    )
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="pilot=5 prompts, full=50 prompts")
    parser.add_argument("--n", type=int, default=None,
                        help="Override number of prompts")
    parser.add_argument("--models", nargs="+",
                        choices=["llama", "gemma", "gemini", "single-pass"],
                        default=["llama", "gemma", "gemini"],
                        help="Which model phases to run")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing results")
    args = parser.parse_args()

    resume = not args.no_resume

    # Load prompts
    config_dir = Path(__file__).resolve().parents[2] / "data"  # [relocated]
    with open(config_dir / "prompts_P01-P50.json") as f:
        all_prompts = json.load(f)

    n = args.n or (5 if args.mode == "pilot" else 50)
    prompts = all_prompts[:n]

    print(f"{'='*60}")
    print(f"EXPERIMENT 1: Multi-Turn LLM Comparison")
    print(f"Mode: {args.mode} | Prompts: {len(prompts)} | Resume: {resume}")
    print(f"Models: {', '.join(args.models)}")
    print(f"{'='*60}\n")

    gemini_client = get_gemini_client()
    all_results = {}

    # ── Phase A: llama3.1:8b multi-turn ──────────────────────────────────
    if "llama" in args.models:
        print(f"\n{'─'*50}")
        print(f"Phase A: llama3.1:8b multi-turn (Ollama local)")
        print(f"  Interviewer: llama3.1:8b | Oracle: gemma3:4b")
        print(f"{'─'*50}")

        llama_results = run_phase(
            "llama3.1_8b_multiturn", prompts,
            run_multiturn_ollama, OUTPUT_DIR,
            resume=resume,
            model="llama3.1:8b",
            oracle_model="gemma3:4b",
            gemini_client=gemini_client
        )
        all_results["llama3.1_8b_multiturn"] = llama_results

    # ── Phase B: gemma3:4b multi-turn ────────────────────────────────────
    if "gemma" in args.models:
        print(f"\n{'─'*50}")
        print(f"Phase B: gemma3:4b multi-turn (Ollama local)")
        print(f"  Interviewer: gemma3:4b | Oracle: llama3.1:8b")
        print(f"{'─'*50}")

        gemma_results = run_phase(
            "gemma3_4b_multiturn", prompts,
            run_multiturn_ollama, OUTPUT_DIR,
            resume=resume,
            model="gemma3:4b",
            oracle_model="llama3.1:8b",
            gemini_client=gemini_client,
            min_turns=4
        )
        all_results["gemma3_4b_multiturn"] = gemma_results

    # ── Phase C: Gemini multi-turn ────────────────────────────────────────
    if "gemini" in args.models:
        print(f"\n{'─'*50}")
        print(f"Phase C: gemini-2.5-flash multi-turn (API, stateful chat)")
        print(f"  Interviewer: gemini-2.5-flash | Oracle: gemma3:4b")
        print(f"{'─'*50}")

        gemini_mt_results = run_phase(
            "gemini-2.5-flash_multiturn", prompts,
            run_multiturn_gemini, OUTPUT_DIR,
            resume=resume,
            gemini_client=gemini_client,
            model="gemini-2.5-flash",
            oracle_model="gemma3:4b"
        )
        all_results["gemini-2.5-flash_multiturn"] = gemini_mt_results

    # ── Phase D: Single-pass baseline ─────────────────────────────────────
    if "single-pass" in args.models:
        print(f"\n{'─'*50}")
        print(f"Phase D: gemini-2.5-flash single-pass (baseline reproduction)")
        print(f"{'─'*50}")

        sp_results = run_phase(
            "gemini-2.5-flash_single_pass", prompts,
            run_single_pass_gemini, OUTPUT_DIR,
            resume=resume,
            gemini_client=gemini_client,
            model="gemini-2.5-flash"
        )
        all_results["gemini-2.5-flash_single_pass"] = sp_results

    if not all_results:
        print("No models selected. Exiting.")
        return

    # ── Summary + LaTeX ───────────────────────────────────────────────────
    summary = compute_summary(all_results)

    summary_path = OUTPUT_DIR / "results_multiturn_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → Summary: {summary_path}")

    generate_latex_table(summary, OUTPUT_DIR / "table_comparison_latex.tex")

    # Save run config
    run_config = {
        "experiment": "Phase 1 — Experiment 1: Multi-Turn LLM Comparison",
        "date": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "max_turns": MAX_TURNS,
        "n_prompts": len(prompts),
        "mode": args.mode,
        "models_tested": list(all_results.keys()),
        "interviewer_models": {
            "llama": "llama3.1:8b (Ollama local)",
            "gemma": "gemma3:4b (Ollama local)",
            "gemini": "gemini-2.5-flash (Google API, free tier, stateful chat)"
        },
        "oracle_models": {
            "for_llama_interviewer": "gemma3:4b (Ollama local)",
            "for_gemma_interviewer": "llama3.1:8b (Ollama local)",
            "for_gemini_interviewer": "gemma3:4b (Ollama local)"
        },
        "scorer_model": "gemini-2.5-flash (Google API, free tier) with Ollama fallback",
        "scoring_method": "LLM-as-judge: parameter classification (RESOLVED/MENTIONED/ABSENT)",
        "coverage_formula": "cov(F) = sum(w_d(p) for resolved p) / sum(w_d(p) for all p in P_d)",
        "gemini_fix": "2026-06-29: switched to stateful client.chats.create() API + turn pressure after turn 5"
    }
    with open(OUTPUT_DIR / "run_config.json", 'w') as f:
        json.dump(run_config, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    ali_targets = {"Consulting": 0.87, "Medical": 0.92, "Payments": 0.75}
    paper_baseline = {"Consulting": 0.887, "Medical": 0.720, "Payments": 0.643}

    for label, data in summary.get("models", {}).items():
        o = data["overall"]
        print(f"\n{label}:")
        print(f"  Overall: {o['mean_cov']*100:.1f}% ± {o['std_cov']*100:.1f}% | "
              f"Turns: {o['mean_turns']:.1f} | Done: {o['done_rate']*100:.0f}%")
        for domain, d in data.get("by_domain", {}).items():
            ali = ali_targets.get(domain, 0)
            baseline = paper_baseline.get(domain, 0)
            print(f"  {domain}: {d['mean_cov']*100:.1f}% ± {d['std_cov']*100:.1f}% "
                  f"(§8 baseline: {baseline*100:.1f}%, ALI target: {ali*100:.0f}%)")

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
