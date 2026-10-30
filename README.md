# Use the fine-tuned MedGemma inside Rudo

Rudo (Flask on Vercel) has no GPU, so MedGemma runs as a separate service and the bot calls it over HTTPS.
If the service is down, slow (>25 s) or answers in English for a non-English user, the bot falls back to Gemini.

## 1. Start the GPU server
    pip install -q unsloth flask
    export HF_TOKEN=...  MEDGEMMA_TOKEN=<long secret>  ADAPTER=tarirozw/medgemma-maternal-lora
    python medgemma_server.py          # serves POST /answer and GET /health on :8000

In Colab, expose it for a demo with a tunnel (ngrok / cloudflared) and use that https URL below.

## 2. Add to the Rudo repo
    cp medgemma_client.py medgemma_guardrails.py <Rudo-Test>/
    cd <Rudo-Test> && git apply rudo_medgemma.patch     # or copy main.py from this folder

## 3. Set env vars on Vercel
    MEDGEMMA_URL=https://<your-gpu-host>      MEDGEMMA_TOKEN=<same secret>      MEDGEMMA_TIMEOUT=25

Leave MEDGEMMA_URL unset to switch back to Gemini only.

## What is routed where
- ask_gemini (pregnancy questions) -> MedGemma first
- ask_gemini_cancer (cervical cancer questions) -> MedGemma first
- ask_gemini_general stays on Gemini: its prompt carries Dawa's contact details and home-visit rules, which MedGemma was not trained on.

## Safety guardrails (medgemma_guardrails.py)
Applied to every MedGemma answer inside `_try_medgemma`:
- Danger signs in the user's message (heavy bleeding, convulsions, baby not moving, waters breaking, severe headache/pain, fainting, ...)
  put the bot's own "see a doctor immediately" line (the existing text from the doctor menu option, all 6 languages) FIRST, followed by the model answer.
  If MedGemma is down in that situation the user still gets the referral line instead of nothing.
- Every normal answer ends with the same referral line. Set `MEDGEMMA_FOOTER=0` to turn this off.
- Output cleaning: control tokens removed, repetition loops and empty/too-short output discarded (falls back to Gemini),
  cut-off last sentence trimmed, internal tokens (`product_image`, `agent_request`, `unable_to_solve_query`) stripped.
- Already in place from before: English-leak check, 25 s timeout, Gemini fallback, human-agent routing (handled upstream in the bot).

The danger-sign list is a STARTER list: English plus the blood/bleeding words already in your repo. Have a clinician who speaks each language add terms.
