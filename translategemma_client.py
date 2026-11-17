

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

def _lang_set(raw: str) -> set:
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


_DEFAULT_TG = "shona,chinyanja,ndebele"
# local -> English: TranslateGemma first (legacy TG_LANGS is honoured as the default).
TG_IN_LANGS = _lang_set(os.environ.get("TG_IN_LANGS", os.environ.get("TG_LANGS", _DEFAULT_TG)))
# English -> local: Gemini by default, because this is the text people read as medical advice.
TG_OUT_LANGS = _lang_set(os.environ.get("TG_OUT_LANGS", ""))
TG_LANGS = TG_IN_LANGS | TG_OUT_LANGS

VERIFY_ALL = os.environ.get("VERIFY_ALL", "0") == "1"
VERIFY_OUT = os.environ.get("VERIFY_OUT", "1") == "1"
TRANSLATE_DEBUG = os.environ.get("TRANSLATE_DEBUG", "0") == "1"

GEM_TRANSLATE_TEMP = float(os.environ.get("GEM_TRANSLATE_TEMP", "0.3"))
GEM_RETRY_TEMP = 0.2

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

# Per-language enforcement lines, the same ones Rudo uses in _get_lang_enforce.
LANG_ENFORCE = {
    "shona":     "Pindura muchiShona chete. Usashandise Chirungu.",
    "ndebele":   "Phendula ngesiNdebele kuphela. Ungasebenzisi isiNgisi.",
    "chinyanja": "Yankhani mu Chichewa/Chinyanja basi. Osagwiritsa ntchito Chingerezi.",
    "lozi":      "Arabela ka Silozi feela. U se ke wa sebelisa Siingelesi.",
    "bemba":     "Yasuka mu Cibemba fye. Wibonfya icingeleshi.",
    "tonga":     "Mupandule mu Chitonga buyo. Mutabelezyi Ciingelezi.",
}

# Same English tell-words and threshold as Rudo's looks_like_english_leak.
_ENGLISH_TELLS = {
    "the", "is", "are", "you", "your", "how", "many", "months", "week", "weeks",
    "should", "would", "could", "will", "does", "doctor", "please", "thank",
    "and", "with", "this", "that", "have", "has", "when", "what", "can",
}
_LEAK_THRESHOLD = 0.15

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


def _english_leak_ratio(text: str) -> float:
    """Share of words that are common English tell-words. 0.0 for very short text."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 6:
        return 0.0
    return sum(1 for w in words if w in _ENGLISH_TELLS) / len(words)


def _looks_like_english_leak(text: str, tgt: str) -> bool:
    """Only meaningful when the target is a local language."""
    if tgt == "english" or not text:
        return False
    return _english_leak_ratio(text) > _LEAK_THRESHOLD


def _gemini_generate(prompt: str, temperature: float) -> str:
    """One Gemini call. Raises if blocked, truncated or empty."""
    cfg = {**_GEN_CFG, "temperature": temperature}
    r = _gem().generate_content(prompt, generation_config=cfg)
    if not _finished_ok(r):
        raise ValueError("Gemini translation did not finish normally (blocked or truncated)")
    out = (r.text or "").strip()
    if not out:
        raise ValueError("Gemini returned an empty translation")
    return out


def _gemini_translate(text: str, src: str, tgt: str) -> str:
    """
    Gemini translation written the way Rudo's ask_gemini works: native-language
    enforcement before and after the prompt, English-leak detection, and one stricter
    retry. Raises on any failure, including truncation.
    """
    masked, mapping = _mask(text)

    if tgt == "english":
        enforce = "Respond in English only."
    else:
        enforce = LANG_ENFORCE.get(tgt, f"Respond in {GEM_NAMES[tgt]} only.")

    base_prompt = (
        f"{enforce}\n\n"
        "You are a professional medical translator for a maternal health chat on WhatsApp. "
        f"Translate the following message from {GEM_NAMES[src]} into clear, simple, natural "
        f"{GEM_NAMES[tgt]}, the way a local health worker would say it. "
        "Keep the medical meaning exact, including danger signs, numbers, units and names. "
        "Do not add or remove advice. Keep tokens like XQ0Z unchanged. "
        "Do not start with phrases like 'Sure' or 'Here is the translation'. "
        "Return only the translation.\n\n"
        f"Message:\n{masked}\n\n"
        f"{enforce}"
    )

    out = _gemini_generate(base_prompt, temperature=GEM_TRANSLATE_TEMP)

    if _looks_like_english_leak(out, tgt):
        logging.warning(f"[gemini_translate] English leak detected for {tgt}, retrying with stricter prompt")
        strict_prompt = (
            f"CRITICAL: Your entire reply must be written only in {GEM_NAMES[tgt]}. "
            "Do not include any English sentences or phrases, even if the original contains "
            "English words (except protected tokens like XQ0Z).\n\n" + base_prompt
        )
        try:
            retry = _gemini_generate(strict_prompt, temperature=GEM_RETRY_TEMP)
            # Keep whichever attempt has less English in it.
            if _english_leak_ratio(retry) < _english_leak_ratio(out):
                out = retry
        except Exception:
            logging.exception("[gemini_translate] strict retry failed, keeping first attempt")

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


def _dbg(engine: str, src: str, tgt: str, text: str, out: str):
    """Full text of one hop, so you can see exactly which step degrades the message."""
    if TRANSLATE_DEBUG:
        logging.info("[translate-debug] %s %s->%s\n  IN : %r\n  OUT: %r", engine, src, tgt, text, out)


# ─────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def translate(text: str, src: str, tgt: str) -> str:
    """
    Translate `text` from Rudo language name `src` to `tgt` (for example "shona" -> "english").
    Raises on any failure so the caller (_retry in the bot) can decide what to do.

    Routing (per direction):
        local -> English: language in TG_IN_LANGS  -> TranslateGemma, else Gemini
        English -> local: language in TG_OUT_LANGS -> TranslateGemma, else Gemini
                          (TG_OUT_LANGS is empty by default, so outbound is Gemini)
        When TranslateGemma is used: Gemini fallback if it fails or TRANSLATEGEMMA_URL is unset,
        and Gemini review when:
            VERIFY_ALL is on, or
            VERIFY_OUT is on and the direction is English -> local, or
            the heuristics flag the output as unsure
    """
    if not text or not text.strip() or src == tgt:
        return text

    outbound = src == "english"
    lang = tgt if outbound else src                  # the non-English side
    use_tg = lang in (TG_OUT_LANGS if outbound else TG_IN_LANGS)

    if use_tg and not TG_URL:
        logging.warning("[translate] TRANSLATEGEMMA_URL is not set, using Gemini")
        use_tg = False

    if not use_tg:
        result = _gemini_translate(text, src, tgt)
        _dbg("gemini", src, tgt, text, result)
        logging.info(f"[translate] {src}->{tgt} via Gemini ok ({len(text)} chars in, {len(result)} out)")
        return result

    try:
        out = _tg_translate(text, src, tgt)
    except Exception:
        logging.exception(f"[translate] TranslateGemma failed for {src}->{tgt}, using Gemini")
        result = _gemini_translate(text, src, tgt)
        _dbg("gemini-fallback", src, tgt, text, result)
        return result
    _dbg("translategemma", src, tgt, text, out)

    if VERIFY_ALL or (VERIFY_OUT and outbound) or _looks_unsure(text, out):
        ok, fixed = _gemini_review(text, src, out, tgt)
        logging.info(f"[translate] review {src}->{tgt}: ok={ok} corrected={'yes' if (not ok and fixed) else 'no'}")
        if not ok and fixed:
            logging.warning("[translate] Gemini overrode TG [%s->%s]: TG=%r GEM=%r", src, tgt, out, fixed)
            _dbg("gemini-review-override", src, tgt, text, fixed)
            return fixed

    logging.info(f"[translate] {src}->{tgt} ok ({len(text)} chars in, {len(out)} chars out)")
    return out
