"""
Client Gemini partage par C0/C3/user_simulator et les fallbacks C1/C4.

Reprend la convention du projet (voir experiments/phase1_results/config/scoring.py) :
la cle est lue depuis un fichier .env a la racine du repo, sur une ligne
contenant "gemini" et "=", ex:
    gemini=AIzaSy...

NOTE POUR L'AGENT LOCAL : ce module ne peut PAS etre teste depuis le sandbox
Cowork (generativelanguage.googleapis.com est bloque par le proxy reseau du
sandbox). Il est ecrit et pret a l'usage, mais non execute. Verifier la
connectivite avec :
    python3 -c "from ali_standalone.gemini_client import get_client; \
                 print(get_client().models.list())"
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from google import genai

from . import config


def _read_key_from_env(env_path: Path) -> str:
    if not env_path.exists():
        raise FileNotFoundError(
            f"Aucun fichier .env trouve a {env_path}. "
            f"Ajouter une ligne 'gemini=VOTRE_CLE_API' a la racine du repo."
        )
    with open(env_path) as f:
        for line in f:
            if "gemini" in line.lower() and "=" in line:
                return line.split("=", 1)[1].strip()
    raise ValueError(f"Pas de ligne 'gemini=...' trouvee dans {env_path}")


_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = _read_key_from_env(config.ENV_FILE)
        _client = genai.Client(api_key=api_key)
    return _client


def gemini_call(system_prompt: str, user_prompt: str, model: str = "gemini-2.5-flash",
                temperature: float = 0.0, max_retries: int = 3) -> str:
    """Appel generique Gemini avec retry exponentiel simple (rate limits)."""
    client = get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config={"system_instruction": system_prompt, "temperature": temperature},
            )
            return resp.text or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


def parse_json_safe(text: str) -> dict:
    """Extrait un dict JSON d'une reponse LLM potentiellement entouree de texte/markdown."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
