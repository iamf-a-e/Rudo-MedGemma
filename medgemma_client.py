
import json
import logging
import os

import requests

MEDGEMMA_URL = os.environ.get("MEDGEMMA_URL", "").strip().rstrip("/")
MEDGEMMA_TOKEN = os.environ.get("MEDGEMMA_TOKEN", "")
MEDGEMMA_TIMEOUT = float(os.environ.get("MEDGEMMA_TIMEOUT", "90"))
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 300


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


def _headers():
    # The Space's own access check uses the `token` argument in the body, not this header.
    # Authorization here is only for private Spaces (a Hugging Face token).
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h


def _read_sse_result(resp):
    """Read a Gradio server-sent-events stream and return the first output value."""
    event = None
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if event == "complete":
                data = json.loads(payload)
                return data[0] if isinstance(data, list) and data else ""
            if event == "error":
                raise RuntimeError(f"Space returned an error: {payload}")
    raise RuntimeError("Space stream ended without a result")


def ask_medgemma(question, history=None):
    if not medgemma_enabled():
        return None

    body = {"data": [
        question,
        json.dumps(history or []),
        MAX_NEW_TOKENS,
        MEDGEMMA_TOKEN,
    ]}

    try:
        # Gradio 5 serves /gradio_api/call/answer, Gradio 4 serves /call/answer
        for prefix in ("/gradio_api", ""):
            call_url = f"{MEDGEMMA_URL}{prefix}/call/answer"
            r = requests.post(call_url, headers=_headers(), json=body, timeout=30)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            event_id = r.json()["event_id"]

            with requests.get(f"{call_url}/{event_id}", headers=_headers(),
                              stream=True, timeout=MEDGEMMA_TIMEOUT) as stream:
                stream.raise_for_status()
                text = (_read_sse_result(stream) or "").strip()
            return text or None

        logging.warning(f"[medgemma] no Gradio API route found at {MEDGEMMA_URL}")
        return None
    except Exception as e:
        logging.warning(f"[medgemma] call failed, falling back: {type(e).__name__}: {e}")
        return None
