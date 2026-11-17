
import hmac
import logging
import os
import re
import threading

from flask import jsonify, request

NLLB_ID = os.environ.get("NLLB_ID", "facebook/nllb-200-distilled-1.3B")
API_TOKEN = os.environ.get("MEDGEMMA_TOKEN")
DETECT_MIN_CONF = float(os.environ.get("DETECT_MIN_CONF", "0.5"))
MAX_CHARS = 4000

# codes used by translategemma_client.LANG_CODES -> NLLB codes
NLLB_CODES = {"en": "eng_Latn", "ny": "nya_Latn", "bem": "bem_Latn", "sn": "sna_Latn",
              "nd": "zul_Latn"}   # NLLB has no Ndebele; Zulu is a close approximation

# GlotLID labels -> Rudo language names. Zulu / S. Ndebele are folded into Ndebele.
GLOT_TO_RUDO = {
    "eng_Latn": "english", "sna_Latn": "shona", "nya_Latn": "chinyanja", "bem_Latn": "bemba",
    "loz_Latn": "lozi", "toi_Latn": "tonga", "tog_Latn": "tonga",
    "nde_Latn": "ndebele", "nbl_Latn": "ndebele", "zul_Latn": "ndebele",
}

_state = {}
_load_lock = threading.Lock()


def _get_nllb():
    with _load_lock:
        if "nllb" not in _state:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(NLLB_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID, torch_dtype=torch.float16).to("cuda").eval()
            _state["nllb"] = (tok, model)
            logging.info("NLLB-200 loaded")
        return _state["nllb"]


def _get_glot():
    with _load_lock:
        if "glot" not in _state:
            import fasttext
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("cis-lmu/glotlid", "model.bin", token=os.environ.get("HF_TOKEN"))
            _state["glot"] = fasttext.load_model(path)
            logging.info("GlotLID loaded")
        return _state["glot"]


def _latin_ok(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    ok = sum(ord(c) < 0x250 or 0x1E00 <= ord(c) <= 0x1EFF for c in letters)
    return ok / len(letters) >= 0.9


def _translate(text, src, tgt, gpu_lock):
    import torch
    tok, model = _get_nllb()
    forced = tok.convert_tokens_to_ids(NLLB_CODES[tgt])
    lines = []
    for line in text.split("\n"):
        if not line.strip():
            lines.append(line)
            continue
        sents = [s for s in re.split(r"(?<=[.!?])\s+", line.strip()) if s]
        out = []
        for i in range(0, len(sents), 8):
            tok.src_lang = NLLB_CODES[src]
            enc = tok(sents[i:i + 8], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            with gpu_lock, torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=forced, num_beams=4,
                                     max_new_tokens=256, max_length=None, no_repeat_ngram_size=4)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        lines.append(" ".join(out))
    return "\n".join(lines)


def _authorized():
    sent = request.headers.get("Authorization", "")
    return bool(API_TOKEN) and hmac.compare_digest(sent, f"Bearer {API_TOKEN}")


def register(app, gpu_lock=None):
    if not API_TOKEN:
        raise RuntimeError("Set MEDGEMMA_TOKEN: these endpoints must not be open to the internet.")
    gpu_lock = gpu_lock or threading.Lock()

    @app.post("/translate")
    def translate_route():
        if not _authorized():
            return jsonify(error="unauthorized"), 401
        d = request.get_json(silent=True) or {}
        text = (d.get("text") or "").strip()
        src, tgt = d.get("source_lang_code"), d.get("target_lang_code")
        if not text:
            return jsonify(error="text is required"), 400
        if len(text) > MAX_CHARS:
            return jsonify(error="text too long"), 413
        if src not in NLLB_CODES or tgt not in NLLB_CODES:
            return jsonify(error="unsupported_language", supported=sorted(NLLB_CODES)), 422
        if src == tgt:
            return jsonify(translation=text, engine="none", approximate=False)
        out = _translate(text, src, tgt, gpu_lock)
        if not out.strip() or not _latin_ok(out):
            return jsonify(error="wrong_script_or_empty_output"), 422
        return jsonify(translation=out, engine="nllb-200", approximate=("nd" in (src, tgt)))

    @app.post("/detect")
    def detect_route():
        if not _authorized():
            return jsonify(error="unauthorized"), 401
        text = " ".join(((request.get_json(silent=True) or {}).get("text") or "").split())[:1000]
        if not text:
            return jsonify(language=None, confidence=0.0, scores={})
        labels, probs = _get_glot().predict(text, k=40)
        scores = {}
        for label, p in zip(labels, probs):
            lang = GLOT_TO_RUDO.get(label.replace("__label__", ""))
            if lang:
                scores[lang] = scores.get(lang, 0.0) + float(p)
        if not scores:
            return jsonify(language=None, confidence=0.0, scores={})
        best = max(scores, key=scores.get)
        return jsonify(language=best if scores[best] >= DETECT_MIN_CONF else None,
                       confidence=round(scores[best], 4),
                       scores={k: round(v, 4) for k, v in scores.items()})
