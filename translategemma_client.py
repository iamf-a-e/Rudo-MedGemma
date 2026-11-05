
import os, logging, requests

TG_URL = os.environ.get("TRANSLATEGEMMA_URL")
TG_TOKEN = os.environ.get("TRANSLATEGEMMA_TOKEN")
TG_TIMEOUT = float(os.environ.get("TRANSLATEGEMMA_TIMEOUT", "20"))

LANG_CODES = {  # verify against chat_template.jinja
    "english": "en", "shona": "sn", "ndebele": "nd",
    "chinyanja": "ny", "bemba": "bem", "tonga": "toi", "lozi": "loz",
}
PIVOT_LANGS = {"shona", "ndebele", "chinyanja", "bemba", "tonga", "lozi"}

# Terms that code-switching speakers keep in English; protect them
PROTECTED_TERMS = ["HPV", "cervical cancer", "ultrasound", "antenatal", "Dawa"]

def translategemma_enabled() -> bool:
    return bool(TG_URL)

def _mask(text):
    mapping = {}
    for i, term in enumerate(PROTECTED_TERMS):
        if term.lower() in text.lower():
            token = f"XQ{i}Z"
            text = re.sub(re.escape(term), token, text, flags=re.I)
            mapping[token] = term
    return text, mapping

def _unmask(text, mapping):
    for token, term in mapping.items():
        text = text.replace(token, term)
    return text

def translate(text: str, src: str, tgt: str) -> str:
    if not text.strip() or src == tgt:
        return text
    masked, mapping = _mask(text)
    r = requests.post(
        TG_URL,
        json={"text": masked,
              "source_lang_code": LANG_CODES[src],
              "target_lang_code": LANG_CODES[tgt]},
        headers={"Authorization": f"Bearer {TG_TOKEN}"},
        timeout=TG_TIMEOUT,
    )
    r.raise_for_status()
    return _unmask(r.json()["translation"], mapping)
