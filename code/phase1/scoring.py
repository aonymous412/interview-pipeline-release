"""
Coverage Scoring Engine for Phase 1 Experiments.
Implements cov(F) = Σ w_d(p) for p ∈ dom(F) / Σ w_d(P_d)

Uses Gemini 2.5 Flash (free tier) as the scorer LLM to classify each parameter as:
  - RESOLVED: concrete value extracted (contributes to coverage)
  - MENTIONED: discussed generically but no concrete value
  - ABSENT: not addressed at all

Fallback: Ollama local model (gemma3:4b) if Gemini rate-limited.

Author: (anonymised for review)
Date: 2026-06-28
"""

import json
import os
import re
import time
import requests
from pathlib import Path
from google import genai

# Load config
CONFIG_DIR = Path(__file__).resolve().parents[2] / "data"  # [relocated]
with open(CONFIG_DIR / "taxonomies.json") as f:
    TAXONOMIES = json.load(f)

# Remove _meta key
TAXONOMIES = {k: v for k, v in TAXONOMIES.items() if not k.startswith("_")}


def get_gemini_client():
    """Load Gemini client from .env file."""
    env_path = Path(__file__).resolve().parents[2] / ".env"  # [relocated]
    with open(env_path) as f:
        for line in f:
            if "gemini" in line.lower() and "=" in line:
                api_key = line.split("=", 1)[1].strip()
                return genai.Client(api_key=api_key)
    raise ValueError("Gemini API key not found in .env")


def call_ollama(prompt: str, model: str = "gemma3:4b", temperature: float = 0.0,
                max_tokens: int = 4000) -> str:
    """Call Ollama local API."""
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["response"]


def call_ollama_chat(messages: list, model: str = "llama3.1:8b",
                     temperature: float = 0.7, max_tokens: int = 500) -> str:
    """Call Ollama chat API (multi-turn)."""
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_gemini(prompt: str, client=None, model: str = "gemini-2.5-flash",
                temperature: float = 0.0, max_tokens: int = 4000) -> str:
    """Call Gemini API."""
    if client is None:
        client = get_gemini_client()
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )
    return response.text


def _extract_json_from_text(text: str) -> dict:
    """Extract JSON object from text that may contain markdown fences, extra text, or truncation."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code fences
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try to repair truncated JSON from code fence
            repaired = _repair_truncated_json(candidate)
            if repaired is not None:
                return repaired
    
    # Try finding first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(candidate)
            if repaired is not None:
                return repaired
    
    # Last resort: find { and try to repair from there
    if start != -1:
        candidate = text[start:]
        repaired = _repair_truncated_json(candidate)
        if repaired is not None:
            return repaired
    
    return None


def _repair_truncated_json(text: str) -> dict:
    """Attempt to repair truncated JSON by closing open braces/quotes."""
    # Remove trailing incomplete key-value pairs
    # Find the last complete entry (ends with })
    last_complete = text.rfind('}')
    if last_complete == -1:
        return None
    
    # Count open/close braces up to last_complete
    candidate = text[:last_complete + 1]
    
    # Close any remaining open braces
    open_braces = candidate.count('{') - candidate.count('}')
    if open_braces > 0:
        candidate += '}' * open_braces
    
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    
    # More aggressive: find entries line by line
    # Look for pattern "param_name": {"status": "X", "evidence": "Y"}
    results = {}
    for match in re.finditer(
        r'"(\w+)"\s*:\s*\{\s*"status"\s*:\s*"(RESOLVED|MENTIONED|ABSENT)"'
        r'(?:\s*,\s*"evidence"\s*:\s*"([^"]*)")?',
        text
    ):
        param_name = match.group(1)
        status = match.group(2)
        evidence = match.group(3) or ""
        results[param_name] = {"status": status, "evidence": evidence}
    
    if results:
        return results
    
    return None


def score_response(response_text: str, domain: str, prompt_text: str,
                   gemini_client=None, scorer: str = "gemini",
                   verbose: bool = False) -> dict:
    """
    Score an LLM response against the domain taxonomy.
    
    Args:
        scorer: "gemini" (default, uses Gemini 2.5 Flash) or "ollama" (uses gemma3:4b)
    
    Returns:
        dict with keys:
            - coverage: float in [0,1]
            - total_weight: int
            - resolved_weight: int  
            - parameters: dict mapping param_name -> {status, weight, evidence}
    """
    taxonomy = TAXONOMIES[domain]
    params = taxonomy["parameters"]
    
    # Build the scoring prompt
    param_descriptions = "\n".join([
        f"- {name} (weight={info['weight']}): {info['description']}"
        for name, info in params.items()
    ])
    
    scoring_prompt = f"""You are a precise annotation assistant for a research project.

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

    # Call the scorer
    raw = None
    try:
        if scorer == "gemini":
            raw = call_gemini(scoring_prompt, client=gemini_client,
                            temperature=0.0, max_tokens=4000)
        else:
            raw = call_ollama(scoring_prompt, model="gemma3:4b",
                            temperature=0.0, max_tokens=4000)
        
        annotations = _extract_json_from_text(raw)
        if annotations is None:
            raise ValueError(f"Could not parse JSON from scorer output: {raw[:200]}")
            
    except Exception as e:
        print(f"  [SCORING ERROR] {e}")
        # Fallback: try the other scorer
        if scorer == "gemini":
            try:
                print("  [FALLBACK] Trying Ollama scorer...")
                raw = call_ollama(scoring_prompt, model="gemma3:4b",
                                temperature=0.0, max_tokens=4000)
                annotations = _extract_json_from_text(raw)
                if annotations is None:
                    raise ValueError("Ollama fallback also failed to produce JSON")
            except Exception as e2:
                print(f"  [SCORING FALLBACK FAILED] {e2}")
                annotations = {name: {"status": "ABSENT", "evidence": "scoring error"}
                              for name in params}
        else:
            annotations = {name: {"status": "ABSENT", "evidence": "scoring error"}
                          for name in params}
    
    # Compute coverage
    total_weight = sum(info["weight"] for info in params.values())
    resolved_weight = 0
    param_results = {}
    
    for name, info in params.items():
        ann = annotations.get(name, {"status": "ABSENT", "evidence": "not evaluated"})
        status = ann.get("status", "ABSENT").upper()
        if status not in ("RESOLVED", "MENTIONED", "ABSENT"):
            status = "ABSENT"
        
        if status == "RESOLVED":
            resolved_weight += info["weight"]
        
        param_results[name] = {
            "status": status,
            "weight": info["weight"],
            "evidence": ann.get("evidence", "")
        }
    
    coverage = resolved_weight / total_weight if total_weight > 0 else 0.0
    
    if verbose:
        print(f"  Coverage: {coverage:.1%} ({resolved_weight}/{total_weight})")
        for name, r in param_results.items():
            if r["status"] == "RESOLVED":
                print(f"    ✓ {name} (w={r['weight']}): {r['evidence'][:60]}")
            elif r["status"] == "MENTIONED":
                print(f"    ~ {name} (w={r['weight']}): {r['evidence'][:60]}")
    
    return {
        "coverage": round(coverage, 4),
        "total_weight": total_weight,
        "resolved_weight": resolved_weight,
        "parameters": param_results,
        "params_resolved": [n for n, r in param_results.items() if r["status"] == "RESOLVED"],
        "params_mentioned": [n for n, r in param_results.items() if r["status"] == "MENTIONED"],
        "params_absent": [n for n, r in param_results.items() if r["status"] == "ABSENT"]
    }


def score_conversation(conversation_history: list, domain: str, prompt_text: str,
                       gemini_client=None, scorer: str = "gemini") -> dict:
    """
    Score a multi-turn conversation by building a full transcript and scoring it.
    """
    full_transcript = "\n\n".join([
        f"[{msg['role'].upper()}] {msg['content']}"
        for msg in conversation_history
    ])
    
    return score_response(full_transcript, domain, prompt_text, gemini_client, scorer)


if __name__ == "__main__":
    print("Testing scoring engine...")
    
    # Test with Gemini
    gemini_client = get_gemini_client()
    result = score_response(
        "You should build a React website with a clean design for selling jewelry online. "
        "Target audience would be women aged 25-45. Budget should be around $5,000. "
        "You'll need product pages, a shopping cart, and Stripe for payments.",
        "Consulting",
        "I want to create an online store for handmade jewelry.",
        gemini_client=gemini_client,
        scorer="gemini",
        verbose=True
    )
    print(f"\nGemini scorer test — Coverage: {result['coverage']:.1%}")
    print(f"  Resolved: {result['params_resolved']}")
    print(f"  Mentioned: {result['params_mentioned']}")
    
    # Test with Ollama
    print("\n--- Ollama scorer test ---")
    result2 = score_response(
        "You should build a React website with a clean design for selling jewelry online. "
        "Target audience would be women aged 25-45. Budget should be around $5,000. "
        "You'll need product pages, a shopping cart, and Stripe for payments.",
        "Consulting",
        "I want to create an online store for handmade jewelry.",
        scorer="ollama",
        verbose=True
    )
    print(f"\nOllama scorer test — Coverage: {result2['coverage']:.1%}")
