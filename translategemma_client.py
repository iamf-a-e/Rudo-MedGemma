"""
translategemma_client.py

Thin HTTP client for a self-hosted TranslateGemma endpoint.

Server contract (you host this next to MedGemma on a GPU box):
    POST TRANSLATEGEMMA_URL
    body:     {"text": "...", "source_lang_code": "sn", "target_lang_code": "en"}
    response: {"translation": "..."}

Env vars:
    TRANSLATEGEMMA_URL        endpoint URL (if unset, translation is disabled)
    TRANSLATEGEMMA_TOKEN      optional bearer token
    TRANSLATEGEMMA_TIMEOUT    seconds per request (default 25)
    TRANSLATEGEMMA_MAX_CHARS  max characters per request, longer text is chunked (default 1200)
    TRANSLATE_LANGS           comma separated list of Rudo languages that must be pivoted through
                              English (default: all six non-English languages). Remove a language
                              from this list once your fine-tuned MedGemma handles it natively.
"""

import logging
import os
import re

import requests

TG_URL = os.environ.get("TRANSLATEGEMMA_URL")
TG_TOKEN = os.environ.get("TRANSLATEGEMMA_TOKEN", "")
TG_TIMEOUT = float(os.environ.get("TRANSLATEGEMMA_TIMEOUT", "25"))
TG_MAX_CHARS = int(os.environ.get("TRANSLATEGEMMA_MAX_CHARS", "1200"))

# Verify each code against the allow list in the model's chat_template.jinja.
LANG_CODES = {
    "english": "en",
    "shona": "sn",
    "ndebele": "nd",
    "chinyanja": "ny",
    "bemba": "bem",
    "tonga": "toi",
    "lozi": "loz",
}

_DEFAULT_PIVOT = "shona,ndebele,chinyanja,bemba,tonga,lozi"
PIVOT_LANGS = {
    x.strip().lower()
    for x in os.environ.get("TRANSLATE_LANGS", _DEFAULT_PIVOT).split(",")
    if x.strip()
}

# Terms that code-switching speakers keep in English. They are swapped for opaque
# tokens before translation and restored afterwards. Longest terms first.
PROTECTED_TERMS = ["cervical cancer", "Dawa Health", "DawaMom", "antenatal", "ultrasound", "HPV"]


def translategemma_enabled() -> bool:
    return bool(TG_URL)


def needs_translation(lang: str) -> bool:
    """True when this language must be pivoted through English."""
    return lang in PIVOT_LANGS


def _mask(text: str):
    mapping = {}
    for i, term in enumerate(PROTECTED_TERMS):
        if re.search(re.escape(term), text, flags=re.I):
            token = f"XQ{i}Z"
            text = re.sub(re.escape(term), token, text, flags=re.I)
            mapping[token] = term
    return text, mapping


def _unmask(text: str, mapping: dict) -> str:
    for token, term in mapping.items():
        text = text.replace(token, term)
    return text


def _chunks(text: str, limit: int):
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = line
    if current:
        chunks.append(current)
    return chunks


def _translate_chunk(text: str, src_code: str, tgt_code: str) -> str:
    headers = {"Content-Type": "application/json"}
    if TG_TOKEN:
        headers["Authorization"] = f"Bearer {TG_TOKEN}"
    resp = requests.post(
        TG_URL,
        json={"text": text, "source_lang_code": src_code, "target_lang_code": tgt_code},
        headers=headers,
        timeout=TG_TIMEOUT,
    )
    resp.raise_for_status()
    out = resp.json().get("translation")
    if not out or not out.strip():
        raise ValueError("TranslateGemma returned an empty translation")
    return out.strip()


def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate `text` from Rudo language name `src` to `tgt` (for example "shona" -> "english").
    Raises on any failure so the caller can decide what to do.
    """
    if not text or not text.strip() or src == tgt:
        return text
    if not TG_URL:
        raise RuntimeError("TRANSLATEGEMMA_URL is not set")

    src_code = LANG_CODES[src]
    tgt_code = LANG_CODES[tgt]

    masked, mapping = _mask(text)
    parts = []
    for chunk in _chunks(masked, TG_MAX_CHARS):
        if not chunk.strip():
            parts.append("")
            continue
        parts.append(_translate_chunk(chunk, src_code, tgt_code))

    result = _unmask("\n".join(parts), mapping)
    logging.info(f"[translate] {src}->{tgt} ok ({len(text)} chars in, {len(result)} chars out)")
    return result
