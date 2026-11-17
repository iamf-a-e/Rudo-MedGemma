
import logging
import os

import requests

_base = os.environ.get("MEDGEMMA_URL", "").rstrip("/")
LANGID_URL = os.environ.get("LANGID_URL") or (f"{_base}/detect" if _base else "")
LANGID_TOKEN = os.environ.get("MEDGEMMA_TOKEN")
LANGID_TIMEOUT = float(os.environ.get("LANGID_TIMEOUT", "4"))

_SUPPORTED = {"english", "shona", "ndebele", "chinyanja", "bemba", "tonga", "lozi"}


def detect_remote(message):
    """Return a Rudo language name, or None if disabled, unsure or failed."""
    if not LANGID_URL or not (message or "").strip():
        return None
    headers = {"Authorization": f"Bearer {LANGID_TOKEN}"} if LANGID_TOKEN else {}
    try:
        r = requests.post(LANGID_URL, json={"text": message}, headers=headers, timeout=LANGID_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        lang = data.get("language")
        logging.info(f"[langid] {lang} ({data.get('confidence')})")
        return lang if lang in _SUPPORTED else None
    except Exception as e:
        logging.warning(f"[langid] call failed: {type(e).__name__}: {e}")
        return None
