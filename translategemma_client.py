"""
translategemma_client.py

Translation layer for the Rudo bot.

Engines:
    TranslateGemma  self-hosted HTTP endpoint (languages in TG_LANGS)
    Gemini          direct translation for languages not in TG_LANGS (for example Bemba, Tonga,
                    Lozi), as a fallback when TranslateGemma fails, and as a reviewer of
                    TranslateGemma output

Server contract for TranslateGemma (you host this next to MedGemma on a GPU box):
    POST TRANSLATEGEMMA_URL
    body:     {"text": "...", "source_lang_code": "sn", "target_lang_code": "en"}
    response: {"translation": "..."}

Env vars:
    TRANSLATEGEMMA_URL        endpoint URL (if unset, translation is disabled)
    TRANSLATEGEMMA_TOKEN      optional bearer token
    TRANSLATEGEMMA_TIMEOUT    seconds per request (default 25)
    TRANSLATEGEMMA_MAX_CHARS  max characters per request, longer text is chunked (default 1200)
    TRANSLATE_LANGS           Rudo language NAMES that must be pivoted through English
                              (default: all six non-English languages)
    TG_LANGS                  Rudo language NAMES that go to TranslateGemma first
                              (default: shona,chinyanja,ndebele). Other pivoted languages go
                              straight to Gemini.
    VERIFY_ALL                "1" = Gemini reviews every TranslateGemma output (safest, slower).
                              "0" = only outputs flagged by the heuristics are reviewed.
    GEN_API                   Gemini key
    GEN_MODEL                 Gemini model name (default gemini-2.5-flash)

Use language names in the env vars, for example TRANSLATE_LANGS=shona,chinyanja,ndebele,bemba,tonga,lozi
(not codes like sn, ny).
"""

import json
import logging
import os
import re

import google.generativeai as genai
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

_DEFAULT_TG = "shona,chinyanja,ndebele"
TG_LANGS = {
    x.strip().lower()
    for x in os.environ.get("TG_LANGS", _DEFAULT_TG).split(",")
    if x.strip()
}

VERIFY_ALL = os.environ.get("VERIFY_ALL", "0") == "1"

# Terms that code-switching speakers keep in English. They are swapped for opaque
# tokens before translation and restored afterwards. Longest terms first.
PROTECTED_TERMS = ["cervical cancer", "Dawa Health", "DawaMom", "antenatal", "ultrasound", "HPV"]

# Names used in Gemini prompts.
GEM_NAMES = {
    "english": "English",
    "shona": "Shona",
    "chinyanja": "Nyanja (Chichewa)",
    "ndebele": "Northern Ndebele (Zimbabwe)",
    "bemba": "Bemba",
    "tonga": "Tonga (Chitonga, Zambia)",
    "lozi": "Lozi (Silozi)",
}

# Thinking tokens count against max_output_tokens on Gemini 2.5, so keep this generous.
# finish_reason=2 (MAX_TOKENS) in the logs means this budget ran out.
_GEN_CFG = {"temperature": 0, "max_output_tokens": 4096}

# Medical wording can trip the default safety filters.
_SAFETY = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

_gem_model = None


def _gem():
    """Create the Gemini model lazily so a missing key does not break import."""
    global _gem_model
    if _gem_model is None:
        key = os.environ.get("GEN_API")
        if not key:
            raise RuntimeError("GEN_API is not set")
        genai.configure(api_key=key)
        _gem_model = genai.GenerativeModel(
            os.environ.get("GEN_MODEL", "gemini-2.5-flash"),
            safety_settings=_SAFETY,
        )
    return _gem_model


def translategemma_enabled() -> bool:
    return bool(TG_URL)


def needs_translation(lang: str) -> bool:
    """True when this language must be pivoted through English."""
    return lang in PIVOT_LANGS


# ─────────────────────────────────────────────
#  MASKING AND CHUNKING
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  TRANSLATEGEMMA
# ─────────────────────────────────────────────

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


def _tg_translate(text: str, src: str, tgt: str) -> str:
    """TranslateGemma translation with masking and chunking. Raises on any failure."""
    src_code = LANG_CODES[src]
    tgt_code = LANG_CODES[tgt]

    masked, mapping = _mask(text)
    parts = []
    for chunk in _chunks(masked, TG_MAX_CHARS):
        if not chunk.strip():
            parts.append("")
            continue
        parts.append(_translate_chunk(chunk, src_code, tgt_code))

    return _unmask("\n".join(parts), mapping)


# ─────────────────────────────────────────────
#  GEMINI (translation fallback and reviewer)
# ─────────────────────────────────────────────

def _finished_ok(response) -> bool:
    """True when Gemini stopped normally, so the text is not truncated or blocked."""
    if not response.candidates:
        return False
    reason = response.candidates[0].finish_reason
    return getattr(reason, "value", reason) == 1  # 1 == STOP


def _gemini_translate(text: str, src: str, tgt: str) -> str:
    """Gemini translation with masking. Raises on any failure, including truncation."""
    masked, mapping = _mask(text)
    prompt = (
        f"Translate the following text from {GEM_NAMES[src]} to {GEM_NAMES[tgt]}. "
        "It is part of a maternal health chat. Keep the medical meaning exact, "
        "keep numbers, names, units and tokens like XQ0Z unchanged, and do not add or remove advice. "
        "Return only the translation.\n\n" + masked
    )
    r = _gem().generate_content(prompt, generation_config=_GEN_CFG)
    if not _finished_ok(r):
        raise ValueError("Gemini translation did not finish normally (blocked or truncated)")
    out = (r.text or "").strip()
    if not out:
        raise ValueError("Gemini returned an empty translation")
    return _unmask(out, mapping)


def _gemini_review(src_text: str, src: str, candidate: str, tgt: str):
    """
    Returns (ok, corrected_translation). corrected_translation is only meaningful when ok is False.
    If the review itself fails, returns (True, "") so the TranslateGemma output is kept.
    """
    prompt = (
        f"Original ({GEM_NAMES[src]}):\n{src_text}\n\n"
        f"Proposed translation ({GEM_NAMES[tgt]}):\n{candidate}\n\n"
        "This is a maternal health chat. Check that the translation preserves the exact "
        "medical meaning, including danger signs, numbers and instructions, and adds nothing. "
        'Reply with JSON only: {"ok": true or false, "translation": "<corrected translation if not ok, else empty>"}'
    )
    try:
        r = _gem().generate_content(
            prompt, generation_config={**_GEN_CFG, "response_mime_type": "application/json"}
        )
        if not _finished_ok(r):
            logging.warning("[review] Gemini review did not finish normally, keeping TranslateGemma output")
            return True, ""
        data = json.loads(r.text)
        return bool(data.get("ok")), (data.get("translation") or "").strip()
    except Exception:
        logging.exception("[review] Gemini review failed, keeping TranslateGemma output")
        return True, ""


def _looks_unsure(src_text: str, out: str) -> bool:
    """Cheap heuristics that decide when a TranslateGemma output gets a second opinion."""
    if not out or not out.strip():
        return True
    if out.strip() == src_text.strip() and len(src_text) > 20:
        return True                                   # came back untranslated
    ratio = len(out) / max(len(src_text), 1)
    if ratio < 0.4 or ratio > 3.0:
        return True                                   # suspicious length
    if set(re.findall(r"\d+", src_text)) != set(re.findall(r"\d+", out)):
        return True                                   # numbers changed
    return False


# ─────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate `text` from Rudo language name `src` to `tgt` (for example "shona" -> "english").
    Raises on any failure so the caller (_retry in the bot) can decide what to do.

    Routing:
        non-English side not in TG_LANGS -> Gemini
        otherwise                        -> TranslateGemma, Gemini fallback if it fails,
                                            Gemini review when VERIFY_ALL or the output looks unsure
    """
    if not text or not text.strip() or src == tgt:
        return text
    if not TG_URL:
        raise RuntimeError("TRANSLATEGEMMA_URL is not set")

    lang = tgt if src == "english" else src          # the non-English side

    if lang not in TG_LANGS:
        result = _gemini_translate(text, src, tgt)
        logging.info(f"[translate] {src}->{tgt} via Gemini ok ({len(text)} chars in, {len(result)} out)")
        return result

    try:
        out = _tg_translate(text, src, tgt)
    except Exception:
        logging.exception(f"[translate] TranslateGemma failed for {src}->{tgt}, using Gemini")
        return _gemini_translate(text, src, tgt)

    if VERIFY_ALL or _looks_unsure(text, out):
        ok, fixed = _gemini_review(text, src, out, tgt)
        if not ok and fixed:
            logging.warning("[translate] Gemini overrode TG [%s->%s]: TG=%r GEM=%r", src, tgt, out, fixed)
            return fixed

    logging.info(f"[translate] {src}->{tgt} ok ({len(text)} chars in, {len(out)} chars out)")
    return out
