"""
Phase 2 - Section 2.3 / Piste A : evaluation de l'extraction Rad-Assist (GGUF).

Rad-Assist n'est PAS un systeme conversationnel ALI -- c'est un extracteur
one-shot (Mistral 7B + GRPO, cf. checkpoints/rad_assist/EXTRACTION_REPORT_MODEL.md).
Son evaluation se fait separement de cov(F) : precision/rappel par variable
clinique sur un holdout MIMIC-CXR, table separee dans le papier ("extraction
accuracy", pas "coverage").

PREREQUIS NON SATISFAITS AU MOMENT DE L'ECRITURE DE CE SCRIPT (2026-07-06) :
  1. `checkpoints/rad_assist/reports_clean.csv` (holdout MIMIC-CXR nettoye,
     ~90 Mo) n'a pas ete telecharge localement -- il est reste sur le Google
     Drive (MyDrive/RadAssist/reports_clean.csv). A telecharger avant de lancer
     ce script.
  2. Aucun fichier de verite terrain structuree (valeurs des 8 CLINICAL_VARIABLES
     par rapport) n'a ete trouve, ni sur le Drive ni dans le repo. Sans verite
     terrain, on ne peut PAS calculer une vraie precision/rappel -- seulement
     un taux de "JSON valide" (sanity check de format), ce que fait ce script
     en attendant.

  Construire la verite terrain necessiterait soit (a) un ré-appariement avec
  les colonnes structurees originales de simhadrisadaram/mimic-cxr-dataset
  (si elles existent au-dela du texte libre), soit (b) une relecture manuelle
  d'un sous-ensemble de rapports -- les deux sont hors scope de cette passe et
  a planifier separement (voir phase2_report.md, section limitations).

Usage (une fois reports_clean.csv present) :
    ollama create radassist -f checkpoints/rad_assist/model/Modelfile   # deja fait
    python3 checkpoints/ali_standalone/radassist_extraction_eval.py --n 50
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_CSV = REPO_ROOT / "checkpoints" / "rad_assist" / "reports_clean.csv"

CLINICAL_VARIABLES = [
    "tumor_size_mm", "sum_of_diameters_mm", "lesion_count", "non_target_status",
    "new_lesion_flag", "performance_status", "tumor_density_hu", "enhancement_pattern",
]

EXTRACTION_PROMPT = (
    "Extract ALL clinically relevant findings from this radiology report as a "
    "JSON array of (variable, value) pairs. Output ONLY valid JSON.\n\nReport:\n{report}"
)


OLLAMA_URL = "http://localhost:11434/api/generate"


def run_radassist(report_text: str, timeout: int = 60) -> str:
    # API HTTP plutot que `ollama run` : la CLI injecte des sequences de
    # controle terminal (ESC[nD/ESC[K) dans stdout, ce qui casse le parsing JSON.
    payload = json.dumps({
        "model": "radassist",
        "prompt": EXTRACTION_PROMPT.format(report=report_text),
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))["response"].strip()


def is_valid_json_array(text: str) -> bool:
    try:
        parsed = json.loads(text)
        return isinstance(parsed, list) and all(
            isinstance(e, dict) and "variable" in e and "value" in e for e in parsed
        )
    except (json.JSONDecodeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="nombre de rapports holdout")
    parser.add_argument("--timeout", type=int, default=180,
                        help="timeout par rapport en secondes (inference CPU : compter 30-90s/rapport)")
    args = parser.parse_args()

    if not REPORTS_CSV.exists():
        print(
            f"MANQUANT : {REPORTS_CSV}\n"
            "Telecharger MyDrive/RadAssist/reports_clean.csv depuis Google Drive "
            "avant de lancer cette evaluation (90 Mo, ~50-100k rapports)."
        )
        return

    df = pd.read_csv(REPORTS_CSV)
    text_col = "report" if "report" in df.columns else df.columns[0]
    sample = df[text_col].dropna().sample(n=min(args.n, len(df)), random_state=42)

    n_valid = 0
    results = []
    for i, report in enumerate(sample):
        try:
            output = run_radassist(report, timeout=args.timeout)
        except (TimeoutError, OSError):
            output = ""
        valid = is_valid_json_array(output)
        n_valid += valid
        results.append({"report_excerpt": report[:200], "output": output, "valid_json": valid})
        print(f"[{i + 1}/{len(sample)}] valid_json={valid}")

    report = {
        "n": len(sample),
        "valid_json_rate": round(n_valid / len(sample), 4) if len(sample) else None,
        "note": "Taux de format valide uniquement -- PAS de precision/rappel par "
                "variable clinique (aucune verite terrain structuree disponible, "
                "voir docstring de ce script).",
    }
    out_dir = Path(__file__).resolve().parents[2] / "recomputed"  # [relocated]
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "radassist_extraction_eval_report.json", "w") as f:
        json.dump({"summary": report, "results": results}, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
