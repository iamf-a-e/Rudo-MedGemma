"""Tiny client the Rudo bot uses to call the MedGemma server. Never raises: returns None on any failure
so the caller can fall back to Gemini."""
import logging
import os

import requests

MEDGEMMA_URL = os.environ.get("MEDGEMMA_URL", "").rstrip("/")      # e.g. https://my-gpu-host:8000
MEDGEMMA_TOKEN = os.environ.get("MEDGEMMA_TOKEN")
MEDGEMMA_TIMEOUT = float(os.environ.get("MEDGEMMA_TIMEOUT", "25"))  # seconds before falling back


def medgemma_enabled() -> bool:
    return bool(MEDGEMMA_URL)


def pairs_from_history(entries, max_turns=2):
    """Turn the bot's stored conversation [{"role": "user"|"bot", "message": str}, ...] into
    [{"user":..., "model":...}] pairs. The newest entry is the current question, so it is dropped."""
    entries = (entries or [])[:-1]
    pairs, pending = [], None
    for e in entries:
        msg = (e.get("message") or "").strip()
        if not msg:
            continue
        if e.get("role") == "user":
            pending = msg
        elif e.get("role") == "bot" and pending and len(msg) >= 20:   # skip short acks
            pairs.append({"user": pending, "model": msg})
            pending = None
    return pairs[-max_turns:]


def ask_medgemma(question, history=None):
    if not medgemma_enabled():
        return None
    headers = {"Authorization": f"Bearer {MEDGEMMA_TOKEN}"} if MEDGEMMA_TOKEN else {}
    try:
        r = requests.post(f"{MEDGEMMA_URL}/answer", headers=headers, timeout=MEDGEMMA_TIMEOUT,
                          json={"question": question, "history": history or []})
        r.raise_for_status()
        text = (r.json().get("answer") or "").strip()
        return text or None
    except Exception as e:
        logging.warning(f"[medgemma] call failed, falling back: {type(e).__name__}: {e}")
        return None
