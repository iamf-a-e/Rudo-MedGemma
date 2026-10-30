"""Safety guardrails around the MedGemma answers, so they behave like the rest of the Rudo bot.

1. Danger-sign check on the user's message  -> urgent "see a doctor" line goes FIRST.
2. "See a doctor" safety footer on every normal answer (same idea as the Gemini disclaimer).
3. Output hygiene for a small fine-tuned model: strip template/control tokens, drop repetition
   loops and empty output, trim a cut-off last sentence.

The referral lines below are copied verbatim from the bot's existing "doctor" menu option in main.py,
so no new medical wording is introduced in any language.
"""
import logging
import os
import re

SEE_DOCTOR = {
    "shona": "Enda kuchiremba kana uine kurwadziwa kwakanyanya, kubuda ropa kwakawanda, kana fivha yepamusoro.",
    "ndebele": "Iya kudokotela uma unobuhlungu obukhulu, ukuphuma kwegazi okukhulu, noma imfiva ephezulu.",
    "chinyanja": "Pitani kudokotala ngati muli ndi kupweteka kwakukulu, kutuluka magazi ambiri, kapena malungo apamwamba.",
    "lozi": "Bona ngaka kapili ha u ka ba ni buhlungu bo boholo, kuelwa mali a mangi, kamba mufufutso o mutuna.",
    "tonga": "Bona dokotela cakufwambana, kutuluka magazi amanji, naa malungo apamwamba.",
    "bemba": "Kamubona musilisi mukufwambaana kuti mwamvwa kucisa kapati, kuswa bulowa bunji, naa kupya kapati mubili.",
}
SEE_DOCTOR_DEFAULT = "See a doctor immediately if you experience severe pain, heavy bleeding, or high fever."

# STARTER LIST: English terms plus the local blood/bleeding words that already appear in this repo.
# Clinicians who speak each language should extend this (convulsions, baby not moving, waters breaking...).
# A false positive only adds the referral line in front of the answer, so lean towards including terms.
DANGER_PATTERNS = [
    r"\bbleed(?:ing|s)?\b", r"\bblood loss\b", r"\bpassing blood\b", r"\bheavy period\b",
    r"\bhaemorrhag\w*", r"\bhemorrhag\w*", r"\bmiscarr\w*",
    r"\bconvuls\w*", r"\bseizure\w*", r"\bfits\b", r"\bunconscious\b", r"\bfaint(?:ed|ing)?\b",
    r"\bpassed out\b", r"\bsevere (?:headache|pain|abdominal pain|stomach pain)\b",
    r"\bblurr?(?:ed|y) vision\b", r"\bswollen (?:face|hands)\b",
    r"\b(?:baby|fetus|foetus) (?:is |has )?(?:not|stopped|isn't|no) (?:moving|movement)\b",
    r"\bno (?:fetal |foetal |baby )?movement\b", r"\bwater(?:s)? (?:has |have )?broke\b",
    r"\bleaking (?:fluid|water)\b", r"\b(?:high|very high) fever\b",
    r"\b(?:can'?t|cannot|difficulty|hard to) breath\w*", r"\blabou?r pains?\b", r"\bsuicid\w*",
    # local-language words already used in the bot's own referral lines
    r"\bropa\b", r"\bmagazi\b", r"\bigazi\b", r"\bbulowa\b", r"\bkuelwa mali\b",
]
_DANGER_RE = re.compile("|".join(DANGER_PATTERNS), re.I)

_CONTROL_TOKENS = re.compile(r"<\s*/?\s*(?:start_of_turn|end_of_turn|bos|eos|pad)\s*>|<unused\d+>", re.I)
FOOTER_ON = os.environ.get("MEDGEMMA_FOOTER", "1") != "0"


def see_doctor_line(lang: str) -> str:
    return SEE_DOCTOR.get(lang, SEE_DOCTOR_DEFAULT)


def is_danger_sign(text: str) -> bool:
    return bool(text and _DANGER_RE.search(text))


def _has_repetition_loop(text: str) -> bool:
    parts = [p.strip().lower() for p in re.split(r"[.!?\n]+", text) if len(p.strip()) >= 4]
    return any(parts.count(p) >= 3 for p in set(parts))


def clean_model_output(text):
    """Return a safe-to-send string, or None if the output should be discarded."""
    if not text:
        return None
    text = _CONTROL_TOKENS.sub("", text)
    text = text.replace("product_image", "").replace("unable_to_solve_query", "").replace("agent_request", "")
    text = re.sub(r"^\s*model\s*\n", "", text, flags=re.I).strip()
    if len(text) < 15 or _has_repetition_loop(text):
        return None
    if text[-1] not in ".!?)\"'”:" and not text.endswith("\n"):      # generation hit max_new_tokens mid-sentence
        cut = max(text.rfind(c) for c in ".!?")
        if cut > len(text) * 0.5:
            text = text[: cut + 1]
    return text


def compose_reply(question: str, lang: str, model_text):
    """Combine guardrails and the (already cleaned) model answer.
    Returns None only when there is nothing to send and no danger sign (caller falls back to Gemini)."""
    line = see_doctor_line(lang)
    if is_danger_sign(question):
        logging.warning(f"[guardrails] danger sign detected, lang={lang}")
        return f"⚠️ {line}" + (f"\n\n{model_text}" if model_text else "")
    if not model_text:
        return None
    return f"{model_text}\n\n{line}" if FOOTER_ON else model_text
