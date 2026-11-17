
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


SELF_HARM_SUPPORT = {
    # "shona": "...",
    # "ndebele": "...",
    # "chinyanja": "...",
    # "lozi": "...",
    # "tonga": "...",
    # "bemba": "...",
}
SELF_HARM_DEFAULT = (
    "I'm really sorry you're going through this. You don't have to face it alone. "
    "Please go to your nearest clinic or hospital now, or tell someone you trust so they can be "
    "with you. If you are in immediate danger, ask someone nearby to take you for help right away. "
    "You can also type *agent* to speak to a person."
)

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
    r"\b(?:can'?t|cannot|difficulty|hard to) breath\w*", r"\blabou?r pains?\b",
    # local-language words already used in the bot's own referral lines
    r"\bropa\b", r"\bmagazi\b", r"\bigazi\b", r"\bbulowa\b", r"\bkuelwa mali\b",
]
_DANGER_RE = re.compile("|".join(DANGER_PATTERNS), re.I)

# STARTER LIST: self-harm wording. Same rule as above: clinicians who speak each language
# should extend this. A false positive here only sends the supportive message, so lean inclusive.
SELF_HARM_PATTERNS = [
    r"\bsuicid\w*", r"\bkill (?:myself|me)\b", r"\bend (?:my|it all|my life)\b",
    r"\bwant(?:ed)? to die\b", r"\bdon'?t want to (?:live|be alive)\b",
    r"\bhurt(?:ing)? myself\b", r"\bharm(?:ing)? myself\b", r"\bself[- ]?harm\b",
    r"\btake my (?:own )?life\b", r"\bbetter off dead\b",
]
_SELF_HARM_RE = re.compile("|".join(SELF_HARM_PATTERNS), re.I)

_CONTROL_TOKENS = re.compile(r"<\s*/?\s*(?:start_of_turn|end_of_turn|bos|eos|pad)\s*>|<unused\d+>", re.I)
FOOTER_ON = os.environ.get("MEDGEMMA_FOOTER", "1") != "0"
MIN_ANSWER_CHARS = 8


def see_doctor_line(lang: str) -> str:
    return SEE_DOCTOR.get(lang, SEE_DOCTOR_DEFAULT)


def self_harm_message(lang: str) -> str:
    return SELF_HARM_SUPPORT.get(lang, SELF_HARM_DEFAULT)


def is_danger_sign(text: str) -> bool:
    return bool(text and _DANGER_RE.search(text))


def is_self_harm(text: str) -> bool:
    return bool(text and _SELF_HARM_RE.search(text))


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
    if len(text) < MIN_ANSWER_CHARS or _has_repetition_loop(text):
        return None
    if text[-1] not in ".!?)\"'”:" and not text.endswith("\n"):      # generation hit max_new_tokens mid-sentence
        cut = max(text.rfind(c) for c in ".!?")
        if cut > len(text) * 0.5:
            text = text[: cut + 1]
    return text


def compose_reply(question: str, lang: str, model_text):
    """Combine guardrails and the (already cleaned) model answer.
    Returns None only when there is nothing to send and no danger sign (caller shows its fallback)."""
    # Self-harm wording takes priority: send the supportive message, never the model's answer.
    if is_self_harm(question):
        logging.warning(f"[guardrails] self-harm wording detected, lang={lang}")
        return f"💛 {self_harm_message(lang)}"

    line = see_doctor_line(lang)
    if is_danger_sign(question):
        logging.warning(f"[guardrails] danger sign detected, lang={lang}")
        return f"⚠️ {line}" + (f"\n\n{model_text}" if model_text else "")
    if not model_text:
        return None
    return f"{model_text}\n\n{line}" if FOOTER_ON else model_text
