"""GPU inference server for the fine-tuned MedGemma maternal model.

Run this on a machine with a GPU (Colab T4 for a demo, RunPod / Modal / a VM for real use).
The Rudo WhatsApp bot (Flask on Vercel) calls POST /answer over HTTPS.

    pip install -q unsloth flask
    export HF_TOKEN=hf_xxx                 # read token, MedGemma terms accepted
    export MEDGEMMA_TOKEN=some-long-secret # the bot sends this as a Bearer token
    export ADAPTER=tarirozw/medgemma-maternal-lora   # or a local path
    python medgemma_server.py
"""
import os
import threading

from flask import Flask, jsonify, request

# Must match the system prompt used in training (build_dataset.SYSTEM).
SYSTEM = ("You are a maternal health assistant. Answer in the same language and mixing of "
          "languages the person uses. Give clear, supportive guidance, and tell the person to "
          "go to a clinic or hospital right away if there are danger signs.")

ADAPTER = os.environ.get("ADAPTER", "tarirozw/medgemma-maternal-lora")
API_TOKEN = os.environ.get("MEDGEMMA_TOKEN")
MAX_HISTORY_TURNS = 2          # training was single-turn, so keep context short

app = Flask(__name__)
_lock = threading.Lock()       # one GPU, one generation at a time
_state = {}


def get_model():
    if "model" not in _state:
        from unsloth import FastModel
        model, tokenizer = FastModel.from_pretrained(
            model_name=ADAPTER, max_seq_length=2048, load_in_4bit=True)
        FastModel.for_inference(model)
        _state["model"], _state["tokenizer"] = model, tokenizer
    return _state["model"], _state["tokenizer"]


def build_prompt(question, history=None):
    """Gemma turn format, system text folded into the first user turn (same as training).
    history: list of {"user": str, "model": str}."""
    history = (history or [])[-MAX_HISTORY_TURNS:]
    parts = []
    for i, h in enumerate(history):
        u = f"{SYSTEM}\n\n{h['user']}" if i == 0 else h["user"]
        parts.append(f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{h['model']}<end_of_turn>\n")
    q = question if history else f"{SYSTEM}\n\n{question}"
    parts.append(f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n")
    return "".join(parts)


@app.get("/health")
def health():
    return jsonify(status="ok", adapter=ADAPTER)


@app.post("/answer")
def answer():
    if API_TOKEN and request.headers.get("Authorization") != f"Bearer {API_TOKEN}":
        return jsonify(error="unauthorized"), 401
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="question is required"), 400
    max_new = min(int(data.get("max_new_tokens", 300)), 400)

    import torch
    model, tokenizer = get_model()
    prompt = build_prompt(question[:1500], data.get("history"))
    with _lock, torch.no_grad():
        inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, repetition_penalty=1.1)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return jsonify(answer=text)


if __name__ == "__main__":
    get_model()                # load once at startup so the first user doesn't wait
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
