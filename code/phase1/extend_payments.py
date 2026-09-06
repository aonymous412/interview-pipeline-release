"""
Exp3 — Payments domain extension.

Two jobs:
  1. Recover the 9 Payments prompts whose rows were lost from
     annotation_sample_30.csv (P44, P37, P39, N83, N93, N70, N72, N74, N87).
     Their Gemini responses are cached in responses_30.json, so we only
     need to re-run the scorer (no fresh API call for the response).
  2. Add 6 new, more detailed Payments prompts (matching the richness of
     the Medical "Detailed" prompts) to broaden the Payments sample,
     get Gemini responses + scorer annotations for them.

All new rows are appended to annotation_sample_30.csv with human_status
and human_notes left BLANK for manual annotation. Existing rows
(Consulting, Medical, P48) are left untouched.
"""

import json
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from scoring import score_response, get_gemini_client, call_gemini

OUTPUT_DIR = Path(__file__).parent
CONFIG_DIR = Path(__file__).parent.parent / "config"
ANN_CSV = OUTPUT_DIR / "annotation_sample_30.csv"
RESPONSES_JSON = OUTPUT_DIR / "responses_30.json"

SINGLE_PASS_SYSTEM = (
    "You are a project requirements analyst. Extract all requirements "
    "from the following description and structure them into a complete, "
    "detailed specification. Be thorough — cover every aspect needed "
    "to implement this project/configure this system."
)

# IDs whose rows were lost — recover from cached responses (no new API call for text)
MISSING_IDS = ["P44", "P37", "P39", "N83", "N93", "N70", "N72", "N74", "N87"]

# 6 new, richly detailed Payments prompts (same density as Medical "Detailed" set)
NEW_DETAILED_PAYMENTS = [
    {"id": "V01", "domain": "Payments", "vagueness": "Detailed",
     "text": "I run a market-making desk for a mid-cap altcoin (XYZ/USDT) on a tier-2 "
             "exchange. I need an automated quoting system: post bid/ask quotes 0.3% "
             "around mid-price, refresh every 2 seconds, max inventory ±$50,000 notional. "
             "If inventory exceeds ±$40,000, widen the spread to 0.6% and skew quotes to "
             "rebalance. Hard kill-switch if realized PnL drops below -$5,000 in a single "
             "day. Capital: $200,000 total, $150,000 working capital. Exchange: Binance via "
             "REST + WebSocket API. Logging: every quote update and fill to TimescaleDB. "
             "Alert via PagerDuty if the bot disconnects for more than 30 seconds."},
    {"id": "V02", "domain": "Payments", "vagueness": "Detailed",
     "text": "I'm building an escrow payment system for a freelance marketplace. Client "
             "funds a job ($500-$50,000 range) into escrow via Stripe. Funds release to "
             "freelancer in 3 milestones (30%/40%/30%) upon client approval, with auto-"
             "release after 7 days of client inaction (with a 48h dispute window). Platform "
             "fee: 5% of total, taken at final release. Dispute resolution: funds frozen, "
             "manual review by support team within 72 hours. Refunds: full refund if no "
             "milestone approved, partial pro-rata otherwise. Currency: USD only at launch, "
             "multi-currency in v2. Need full transaction ledger for accounting reconciliation."},
    {"id": "V03", "domain": "Payments", "vagueness": "Detailed",
     "text": "I need a systematic options selling strategy on SPX. Sell 30-delta cash-"
             "secured puts, 30-45 DTE, on the first trading day of each week. Position "
             "size: max 10% of portfolio notional per trade. Close at 50% max profit or "
             "21 DTE, whichever first. Roll down and out if tested (breached strike) and "
             "still profitable to do so; otherwise accept assignment. Portfolio: $500,000, "
             "max 3 concurrent positions. No earnings-week entries. Broker: Tastytrade API. "
             "Daily Greeks report (delta, theta, vega exposure) emailed at market close."},
    {"id": "V04", "domain": "Payments", "vagueness": "Detailed",
     "text": "I want to automate quarterly profit distribution for my 4-person trading "
             "partnership. Total trading account: $800,000, contributions tracked per "
             "partner (35%/25%/25%/15% capital weighting). Distribute 50% of realized "
             "quarterly net profit (after a 2% management fee retained for operating costs) "
             "pro-rata to each partner's bank account via ACH. Losses carried forward, no "
             "distribution in loss quarters. Tax: generate K-1-equivalent statements per "
             "partner. Approval: requires 3-of-4 partner sign-off via DocuSign before any "
             "transfer executes. Full audit trail retained for 7 years."},
    {"id": "V05", "domain": "Payments", "vagueness": "Detailed",
     "text": "I'm setting up an automated stablecoin carry-trade strategy. Borrow USDC on "
             "Aave at variable rate (currently ~4%), convert to a higher-yield stablecoin "
             "position (e.g., sDAI at ~8%), maintain LTV below 70% (target 60%), auto-"
             "deleverage if LTV exceeds 75%. Capital: 500,000 USDC initial, leveraged up to "
             "2x. Monitor borrow rate spread hourly; unwind entirely if spread compresses "
             "below 1.5% for 24 consecutive hours. Gas budget: max $200/month on "
             "transactions. Multi-sig (3-of-5) required for any unwind above $100,000. "
             "Daily report: net APY, LTV, liquidation buffer."},
    {"id": "V06", "domain": "Payments", "vagueness": "Detailed",
     "text": "I need a cross-border remittance product for migrant workers sending money "
             "from the UAE to India, Philippines, and Pakistan. Transfer limits: $50-"
             "$10,000 per transaction, $20,000/month per sender (AML threshold). FX: "
             "live mid-market rate + 1.5% margin, locked for 10 minutes per quote. Payout "
             "methods: bank transfer (2-day settlement), mobile wallet (instant, e.g. "
             "GCash, Easypaisa), cash pickup via partner network. KYC: Emirates ID "
             "verification for sender, recipient ID verification above $1,000. Compliance: "
             "UAE Central Bank licensing, sanctions screening (OFAC, UN) on every "
             "transaction. Settlement: pre-funded nostro accounts in each corridor."},
]


def load_prompt_map():
    with open(CONFIG_DIR / "prompts.json") as f:
        orig = json.load(f)
    sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_baseline_extended"))
    from run_exp2 import NEW_PROMPTS
    all_prompts = orig + NEW_PROMPTS
    return {p["id"]: p for p in all_prompts}


def get_gemini_response(prompt_text, gemini_client):
    full_prompt = f"{SINGLE_PASS_SYSTEM}\n\nUser request:\n{prompt_text}"
    try:
        text = call_gemini(full_prompt, client=gemini_client, temperature=0.0, max_tokens=3000)
        time.sleep(5)
        return text or ""
    except Exception as e:
        print(f"    [ERROR] {e}")
        time.sleep(15)
        return ""


def main():
    print("=" * 60)
    print("EXP3 — Payments extension: recover 9 + add 6 new prompts")
    print("=" * 60 + "\n")

    gemini_client = get_gemini_client()
    prompt_map = load_prompt_map()

    with open(RESPONSES_JSON) as f:
        responses_cache = json.load(f)

    new_rows = []

    # Step 1: recover the 9 missing existing Payments prompts (cached responses)
    print("Step 1 — Recovering 9 lost Payments rows (using cached responses)\n")
    for i, pid in enumerate(MISSING_IDS):
        prompt_data = prompt_map[pid]
        response_text = responses_cache.get(pid, "")
        if not response_text:
            print(f"  [{i+1}/9] {pid} — no cached response, skipping")
            continue
        print(f"  [{i+1}/9] {pid}: \"{prompt_data['text'][:60]}...\"")
        print(f"    Scoring (cached response, {len(response_text)} chars)...")
        time.sleep(4)
        score = score_response(response_text, "Payments", prompt_data["text"],
                               gemini_client=gemini_client, scorer="gemini")
        cov = score["coverage"]
        print(f"    → cov={cov:.1%}")
        for param_name, param_info in score["parameters"].items():
            new_rows.append({
                "prompt_id": pid, "domain": "Payments", "vagueness": prompt_data["vagueness"],
                "annotation_tier": "recovered",
                "coverage_scorer": round(cov, 4),
                "param_name": param_name, "param_weight": param_info["weight"],
                "scorer_status": param_info["status"],
                "scorer_evidence": param_info["evidence"][:100],
                "human_status": "", "human_notes": ""
            })
        time.sleep(2)

    # Step 2: 6 new detailed Payments prompts
    print("\nStep 2 — Adding 6 new detailed Payments prompts\n")
    for i, p in enumerate(NEW_DETAILED_PAYMENTS):
        print(f"  [{i+1}/6] {p['id']}: \"{p['text'][:60]}...\"")
        print(f"    Getting Gemini response...")
        response_text = get_gemini_response(p["text"], gemini_client)
        if not response_text:
            print(f"    [SKIP] empty response")
            continue
        responses_cache[p["id"]] = response_text
        with open(RESPONSES_JSON, 'w') as f:
            json.dump(responses_cache, f, indent=2, ensure_ascii=False)

        print(f"    Scoring...")
        time.sleep(4)
        score = score_response(response_text, "Payments", p["text"],
                               gemini_client=gemini_client, scorer="gemini")
        cov = score["coverage"]
        print(f"    → cov={cov:.1%}")
        for param_name, param_info in score["parameters"].items():
            new_rows.append({
                "prompt_id": p["id"], "domain": "Payments", "vagueness": "Detailed",
                "annotation_tier": "new_detailed",
                "coverage_scorer": round(cov, 4),
                "param_name": param_name, "param_weight": param_info["weight"],
                "scorer_status": param_info["status"],
                "scorer_evidence": param_info["evidence"][:100],
                "human_status": "", "human_notes": ""
            })
        time.sleep(2)

    # Step 3: append to annotation_sample_30.csv (preserve existing rows)
    print(f"\nStep 3 — Appending {len(new_rows)} new rows to annotation_sample_30.csv\n")

    existing_rows = []
    with open(ANN_CSV, newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        for raw in reader:
            existing_rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                                  for k, v in raw.items()})

    all_rows = existing_rows + new_rows

    fieldnames = [
        "prompt_id", "domain", "vagueness", "annotation_tier",
        "coverage_scorer", "param_name", "param_weight",
        "scorer_status", "scorer_evidence",
        "human_status", "human_notes"
    ]
    with open(ANN_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Total rows in CSV now: {len(all_rows)}")
    print(f"  Existing (untouched): {len(existing_rows)}")
    print(f"  New (need human_status): {len(new_rows)}")
    print(f"\nPayments prompts now in sample: "
          f"{sorted(set(r['prompt_id'] for r in all_rows if r['domain']=='Payments'))}")


if __name__ == "__main__":
    main()
