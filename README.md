# Rudo: a multilingual maternal health assistant on WhatsApp

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

Rudo is the WhatsApp assistant for [Dawa Health](https://dawa-health.com/), a maternal and women's health platform serving Zimbabwe and Zambia. Women write to Rudo in the language they speak, including mixed English and local language, and get clear, supportive health guidance back in that language. Danger signs are escalated, and a person can take over the conversation at any time.

Rudo speaks seven languages: English, Shona, Ndebele, Chinyanja, Bemba, Tonga and Lozi.

> Rudo gives general health information. It does not diagnose, prescribe or replace a clinician. See [Clinical safety](#clinical-safety).

## Contents

1. [How it works](#how-it-works)
2. [How the AI agent was built](#how-the-ai-agent-was-built)
3. [The models and where they run](#the-models-and-where-they-run)
4. [Hosting: Hugging Face Spaces, Vercel and Supabase](#hosting-hugging-face-spaces-vercel-and-supabase)
5. [Getting started](#getting-started)
6. [Configuration reference](#configuration-reference)
7. [Repository layout](#repository-layout)
8. [Testing and verification](#testing-and-verification)
9. [Adding or changing a language](#adding-or-changing-a-language)
10. [Clinical safety](#clinical-safety)
11. [Data protection](#data-protection)
12. [Troubleshooting](#troubleshooting)
13. [Contributing](#contributing)
14. [License](#license)

## How it works

```mermaid
flowchart LR
    U["WhatsApp user"] --> M["Meta WhatsApp Cloud API"]
    M -->|"POST /webhook"| V["Rudo bot (Flask on Vercel)"]
    V --> S[("Supabase Postgres")]
    V --> G["Gemini (language detection only)"]
    V -->|"POST /translate"| T["TranslateGemma (Hugging Face Space)"]
    V -->|"POST /answer"| H["MedGemma + LoRA adapter (Hugging Face Space)"]
    V -->|"interactive buttons"| A["Human agents on WhatsApp"]
```

The bot is a Flask app deployed on Vercel. It has no GPU, so the language models run as separate services on Hugging Face Spaces and the bot calls them over HTTPS.

### What happens to one message

1. Meta delivers the message to `POST /webhook`. The bot records the message id so a redelivery never produces a second reply.
2. If the sender is a human agent in an active session, the message is relayed to the user and nothing else happens.
3. The user's state is loaded from Supabase and the message is stored in the chat history.
4. The language is detected: an exact-word fast path first, then a Gemini classifier, then keyword scoring if Gemini is unavailable.
5. The router decides what the message is:
   - first ever message: create a user ID (`DH-<last 4 digits>-<4 letters>`) and send the welcome
   - request for a person: notify the agents with Accept and Reject buttons
   - reset phrase or plain greeting: fixed reply, no model call
   - answer to "Is there anything else I can help you with?": yes, no (offers the shop) or a new question
   - an active shop step: handle product choice, quantity, address
   - anything else: a free-form health question
6. A health question goes through the answer pipeline:

   ```mermaid
   flowchart TD
       Q["User question"] --> P{"Pivot language?"}
       P -->|"yes"| T1["TranslateGemma: language to English"]
       P -->|"no (English)"| M
       T1 --> M["MedGemma answers in English"]
       M --> C{"Reply is English?"}
       C -->|"no"| R["Retry once without history"]
       R --> C2{"English now?"}
       C2 -->|"no"| F["Failure message + offer a human"]
       C -->|"yes"| T2["TranslateGemma: English to language"]
       C2 -->|"yes"| T2
       T2 --> G["Guardrails: danger signs, cleaning, footer"]
       G --> S["Send reply, then ask if anything else is needed"]
   ```

7. The guardrails always run, even when a model or translation step failed. If the message contains a danger sign, the referral line is sent first, so a woman describing heavy bleeding still gets advice to go to a clinic when a service is down.

## How the AI agent was built

Rudo was rebuilt around a fine-tuned medical model. The main design decisions, and why they were made:

### 1. A small fine-tuned medical model instead of a general chatbot

The answer model is [MedGemma 4B](https://huggingface.co/google/medgemma-4b-it) with a LoRA adapter fine-tuned on maternal health content in Zimbabwean and Zambian languages. A 4B model is cheap enough to run on a single modest GPU, and fine-tuning teaches it the vocabulary, tone and referral behaviour Dawa wants. The training system prompt tells it to answer in the language and language mix the person uses, to be clear and supportive, and to send the person to a clinic or hospital immediately if there are danger signs.

Details that matter when you reproduce or retrain:

- The adapter was trained single-turn, so the bot sends at most the last two question and answer pairs as context.
- Training used Gemma's turn format with the system text folded into the first user turn. Inference must use the same format, or the adapter behaves like the base model.
- Inference is greedy (no sampling) with a repetition penalty of 1.1 and a cap of a few hundred new tokens, to keep WhatsApp replies short and stable.
- The adapter is published as [dawahealth/medgemma-maternal-lora-v2](https://huggingface.co/dawahealth/medgemma-maternal-lora-v2). Training data sources, size and the training notebook are summarised here: [TRAINING_DATA_SUMMARY](https://huggingface.co/dawahealth).

### 2. A translation pivot for languages the model is weak in

MedGemma was not pretrained on most of Rudo's languages. For Shona, Ndebele, Chinyanja, Bemba, Tonga and Lozi, the question is translated to English by [TranslateGemma](https://huggingface.co/google/translategemma-4b-it), answered, and translated back. Which languages use the pivot is a setting (`TRANSLATE_LANGS`), because a language the fine-tuned adapter already handles well may do better without it. Always compare both routes on an evaluation set before choosing.

The pivot adds engineering rules, each learned from a failure:

| Rule | Why |
|---|---|
| Only English text reaches MedGemma for pivoted languages | A model with a weak grasp of Shona starts answering in broken Shona if local text is in its context |
| Fixed UI text (welcome, menus, failures) is never stored in the chat history | It would otherwise end up in the model's context |
| Each stored message keeps an English version (`message_en`) | History can be replayed to the model in English without translating again |
| The model reply must look like English before it is translated back | Garbled local-language output would otherwise be "translated" into something further from the truth |
| Terms such as HPV and "cervical cancer" are masked before translation and restored after | Speakers keep these in English and translators tend to mangle them |
| Long text is translated in chunks | Keeps each request within what the translation model handles reliably |

### 3. Language detection by a classifier, not the answer model

Gemini 2.5 Flash is used for one job only: classifying the language of a message. Replies never come from Gemini. If Gemini is down, a keyword and phrase scorer takes over, so detection degrades instead of failing. A one or two word message such as "ok" does not flip a user who has been writing in a local language.

### 4. Safety wrapped around every answer

`medgemma_guardrails.py` runs on every reply:

- danger signs in the user's message (heavy bleeding, convulsions, baby not moving, waters breaking, severe headache or pain, fainting and others) put the referral line first
- every normal answer ends with the same referral line
- control tokens, repetition loops, empty or too short output and internal tokens are removed or discarded
- a cut-off last sentence is trimmed

The danger sign list is a starter list. See [Clinical safety](#clinical-safety).

### 5. A human can always take over

Typing a phrase such as "speak to a person" sends a request to the agents in `AGENTS` with Accept and Reject buttons. The first to accept is connected, messages are relayed both ways, `END CHAT` closes the session and the user can type `CANCEL` while waiting. If nobody accepts within 60 seconds, the user is handed back to Rudo. A scheduled job can call `POST /agent-timeout` to enforce that timeout.

### 6. Commerce and attribution

After an answer Rudo asks whether there is anything else. A "no" offers the product shop (categories, quantity, address) and orders are stored in Supabase. QR codes on posters and flyers open WhatsApp with a prefilled message such as "Hi DawaMom, I got connected from <place>"; the source is stored so Dawa can see which channels bring users in.

### 7. Reliable under WhatsApp's rules

- WhatsApp redelivers a webhook that is not answered quickly. Message ids are stored with a unique constraint so a redelivery is ignored, and the webhook always answers 200.
- A short "thinking" message is sent before slow answers.
- Replies longer than WhatsApp's limit are split on line boundaries.
- User state is reloaded from the database on every message, so a serverless instance never acts on a stale copy.

## The models and where they run

| Service | Model | Role | Approximate GPU memory |
|---|---|---|---|
| MedGemma service | `google/medgemma-4b-it` plus the LoRA adapter | Writes the answer in English | A few GB with 4-bit loading, more in bf16 |
| TranslateGemma service | `google/translategemma-4b-it` | Local language to English and back | About 8 to 9 GB in bf16 |

Both base models are gated on Hugging Face. The account that owns the token must have accepted the terms on each model page. Together the two models fit on one 24 GB GPU, or run on two separate Spaces.

Both services speak plain HTTP with JSON, which is the contract the bot's clients (`medgemma_client.py`, `translategemma_client.py`) rely on:

| Endpoint | Request body | Response |
|---|---|---|
| `GET /health` | none | `{"status": "ok"}` |
| `POST /answer` | `{"question": "...", "history": [{"user": "...", "model": "..."}]}` | `{"answer": "..."}` |
| `POST /translate` | `{"text": "...", "source_lang_code": "sn", "target_lang_code": "en"}` | `{"translation": "..."}` |

Both POST endpoints check an `Authorization: Bearer <token>` header against a secret held in the Space.

### Language codes used with TranslateGemma

| Language | Code |
|---|---|
| English | `en` |
| Shona | `sn` |
| Ndebele | `nd` |
| Chinyanja | `ny` |
| Bemba | `bem` |
| Tonga | `toi` |
| Lozi | `loz` |

Check each code against the model's chat template, and test real sentences per language with a native speaker before relying on it. Support is uneven across low-resource languages, and a language that translates badly should be removed from `TRANSLATE_LANGS`.

## Hosting: Hugging Face Spaces, Vercel and Supabase

### Hugging Face Spaces (Gradio)

The two models run as Gradio Spaces on GPU hardware: [MEDGEMMA_SPACE_URL](https://huggingface.co/spaces) and [TRANSLATEGEMMA_SPACE_URL](https://huggingface.co/spaces).

Set these in each Space under Settings, Variables and secrets:

| Secret | Space | Purpose |
|---|---|---|
| `HF_TOKEN` | both | Hugging Face token with read access, from an account that accepted the model terms. Without it the download fails with a 401 gated repo error |
| `MEDGEMMA_TOKEN` | MedGemma | Bearer token the bot must send |
| `ADAPTER` | MedGemma | Adapter repo, for example `dawahealth/medgemma-maternal-lora-v2` |
| `TRANSLATEGEMMA_TOKEN` | TranslateGemma | Bearer token the bot must send |

Notes for running them:

- A Space needs GPU hardware for these models. Choose it under Settings, Hardware, and check current pricing.
- Idle Spaces can sleep to save cost, and the first request after a sleep waits for the model to load. For production, keep the Space awake or accept a slower first reply.
- Keep the Space public and protect the API with the bearer token that the app checks. As far as I know, a private Space expects a Hugging Face token in the same `Authorization` header, which would clash with your own token.
- The Space URL has the form `https://owner-spacename.hf.space`.

The Gradio UI is only a status page. The bot needs plain REST routes, so `app.py` mounts them next to the Gradio app. A reference pattern for the MedGemma Space (the TranslateGemma Space is the same with `/translate`), where `generate` is your model code:

```python
import os
import gradio as gr
import uvicorn
from fastapi import FastAPI, Header, HTTPException

API_TOKEN = os.environ["MEDGEMMA_TOKEN"]
api = FastAPI()


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/answer")
def answer(body: dict, authorization: str = Header(default="")):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401)
    return {"answer": generate(body["question"], body.get("history", []))}


with gr.Blocks() as demo:
    gr.Markdown("Rudo MedGemma service. The API is at /answer and /health.")

# Routes defined above are matched before the Gradio app mounted at "/"
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

Load the model once at startup, not per request, and serialise generation with a lock so one GPU handles one request at a time.

### Vercel (the bot)

The bot is the Flask app in `main.py`, deployed to Vercel. Vercel receives Meta's webhook and calls the Spaces and Supabase.

- Connect this repository in Vercel and add the environment variables from the [configuration reference](#configuration-reference).
- Your public webhook URL is `https://your-project.vercel.app/webhook`.
- A pivoted reply makes several network calls in a row. Raise the function duration (`maxDuration`) in `vercel.json` or the project's Functions settings, within your plan's limit, and keep the per-call timeouts low (see the table below).
- Redeploy after changing environment variables. Running deployments do not pick them up.

### Supabase (the database)

Supabase Postgres stores everything the bot persists. The schema is in `supabase_schema.sql`:

| Table | Holds |
|---|---|
| `users` | One row per WhatsApp user: language, user ID, step, full state |
| `messages` | Chat history, with `message_en` for pivoted languages |
| `referrals` | Where each user came from |
| `orders` | Shop orders |
| `kv` | Short-lived keys with expiry: agent requests and sessions, message de-duplication |

Row level security is enabled on every table with no policies. The bot uses a secret key (`sb_secret_...`, or the legacy `service_role` key), which bypasses it. Never use the publishable key here, and never expose the secret key in a browser, an app or a public repo. To clear expired `kv` rows automatically, enable `pg_cron` and use the schedule in the schema file.

## Getting started

### Prerequisites

- A Meta developer account with a WhatsApp Business app and a phone number ([WhatsApp Cloud API](https://developers.facebook.com/docs/whatsapp/cloud-api))
- A Hugging Face account that has accepted the terms for [MedGemma](https://huggingface.co/google/medgemma-4b-it) and [TranslateGemma](https://huggingface.co/google/translategemma-4b-it), with a read token
- A Supabase project
- A Vercel account
- A Google AI Studio key for Gemini (language detection)
- Python 3.10 or newer

### 1. Clone and install

```bash
git clone https://github.com/Dawa-Health/rudo_medgemma.git
cd rudo_medgemma
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create the database

In the Supabase SQL Editor, run `supabase_schema.sql`. Then open Settings, API Keys and copy the project URL and a secret key.

### 3. Start the model services

Create the two Spaces (or run the same code on any GPU machine), set their secrets as in [Hugging Face Spaces](#hugging-face-spaces-gradio), and check each one:

```bash
curl https://your-owner-medgemma.hf.space/health

curl -X POST https://your-owner-medgemma.hf.space/answer \
  -H "Authorization: Bearer your-medgemma-token" \
  -H "Content-Type: application/json" \
  -d '{"question":"What foods should I eat in the first trimester?","history":[]}'

curl -X POST https://your-owner-translate.hf.space/translate \
  -H "Authorization: Bearer your-translate-token" \
  -H "Content-Type: application/json" \
  -d '{"text":"Ndine pamuviri","source_lang_code":"sn","target_lang_code":"en"}'
```

### 4. Configure the bot

Export the variables for local development (on Vercel, add them in Project Settings, Environment Variables):

```bash
export WA_TOKEN="your-whatsapp-access-token"
export VERIFY_TOKEN="a-long-random-string"
export GEN_API="your-gemini-key"
export SUPABASE_URL="https://your-project-ref.supabase.co"
export SUPABASE_SERVICE_KEY="sb_secret_your-key"
export MEDGEMMA_URL="https://your-owner-medgemma.hf.space"
export MEDGEMMA_TOKEN="your-medgemma-token"
export TRANSLATEGEMMA_URL="https://your-owner-translate.hf.space/translate"
export TRANSLATEGEMMA_TOKEN="your-translate-token"
```

Two details that cause most first-run failures: `MEDGEMMA_URL` is a base URL with no path, because the client adds `/answer`; `TRANSLATEGEMMA_URL` is used exactly as written, so it must end in `/translate`.

### 5. Run locally and connect WhatsApp

```bash
python3 main.py                       # serves on port 5000
ngrok http 5000                       # or cloudflared, to get a public https URL
```

In the Meta developer dashboard, open WhatsApp, Configuration and set the callback URL to `https://your-tunnel-url/webhook` and the verify token to your `VERIFY_TOKEN`. Subscribe to the `messages` field. You can test the verification step first:

```bash
curl "http://localhost:5000/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY_TOKEN&hub.challenge=12345"
# prints 12345
```

Send "Hello" to your WhatsApp number. You should get a welcome with a user ID, and rows should appear in the Supabase `users`, `messages` and `kv` tables.

### 6. Deploy to Vercel

Push to GitHub, import the repository in Vercel, add the environment variables, raise the function duration and deploy. Point Meta's callback URL at `https://your-project.vercel.app/webhook`.

### 7. Set up human agents

Edit `AGENTS` in `main.py` with the WhatsApp numbers of the people who answer escalations. If the repository is public, read the numbers from an environment variable instead of committing them. To enforce the 60 second timeout, schedule a call to `POST /agent-timeout` with `{"user_number": "...", "phone_id": "..."}` (for example with a scheduler such as QStash or a Vercel cron job).

## Configuration reference

### Bot (Vercel or local)

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `WA_TOKEN` | yes | | WhatsApp Cloud API access token |
| `VERIFY_TOKEN` | no | `BOT` | Must match the verify token in the Meta dashboard. Set your own |
| `GEN_API` | yes | | Gemini key, used only for language detection |
| `SUPABASE_URL` | yes | | `https://<project-ref>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | yes | | Supabase secret key (server side only) |
| `SUPABASE_TIMEOUT` | no | 8 | Seconds per database request |
| `MEDGEMMA_URL` | yes | | Base URL of the MedGemma service, no path |
| `MEDGEMMA_TOKEN` | if the service checks one | | Bearer token for `/answer` |
| `MEDGEMMA_TIMEOUT` | no | 25 | Seconds per MedGemma request |
| `TRANSLATEGEMMA_URL` | yes for pivoted languages | | Full URL ending in `/translate` |
| `TRANSLATEGEMMA_TOKEN` | if the service checks one | | Bearer token for `/translate` |
| `TRANSLATEGEMMA_TIMEOUT` | no | 25 | Seconds per translation request. About 10 works better on serverless |
| `TRANSLATEGEMMA_MAX_CHARS` | no | 1200 | Longer text is translated in chunks |
| `TRANSLATE_LANGS` | no | all six non-English languages | Comma separated languages that use the pivot |
| `PHONE_ID`, `OWNER_PHONE` | no | | Read at startup but not needed by the current flow |
| `FLASK_DEBUG` | no | off | Set to `1` for local debugging only |

### Model Spaces

| Variable | Space | Purpose |
|---|---|---|
| `HF_TOKEN` | both | Download access to the gated base models |
| `MEDGEMMA_TOKEN` | MedGemma | Bearer token the service requires |
| `ADAPTER` | MedGemma | Adapter repo or local path |
| `TRANSLATEGEMMA_TOKEN` | TranslateGemma | Bearer token the service requires |

## Repository layout

```
main.py                    Flask app: webhook, router, answer pipeline, agents, shop
db.py                      Supabase storage layer (users, messages, referrals, orders, kv)
medgemma_client.py         HTTP client for the MedGemma service
medgemma_guardrails.py     Danger signs, output cleaning, referral footer
translategemma_client.py   HTTP client for the TranslateGemma service
products_data.py           Shop catalogue
training/                  Health content and training data modules
templates/connected.html   Landing page served at /
supabase_schema.sql        Database schema with row level security
requirements.txt           Python dependencies
vercel.json                Vercel configuration
LICENSE                    GNU Affero General Public License v3.0
```

`apply_supabase_patch.py` was a one-off tool for moving an older copy of the code from Redis to Supabase. It is not needed once `main.py` has been patched and can be deleted.

## Testing and verification

Before enabling a language in production:

1. Run the connection checks in [Start the model services](#3-start-the-model-services).
2. Check every language code against the translation service with a short sentence.
3. Have a native speaker review a handful of translated answers per language, with special attention to numbers, weeks, doses and danger signs. Translating a reply back to English and comparing is a quick extra check.
4. Compare answers with and without the pivot on your evaluation set, and set `TRANSLATE_LANGS` to the languages where the pivot wins.
5. Send real messages through WhatsApp and confirm the history looks right in Supabase: user messages with `message_en` filled for pivoted languages, and no fixed UI text stored as bot messages.

The server logs the pipeline (question, English question, English reply, final reply) at INFO level, which is the fastest way to see where a bad answer came from. Remember these logs can contain health information and should be handled accordingly.

## Adding or changing a language

1. Add the language to `SUPPORTED_LANGS`, the Gemini classifier prompt, and the keyword dictionaries (`EXACT_MATCHES`, `LANGUAGE_KEYWORDS`, `LANGUAGE_PHRASES`) in `main.py`.
2. Add its code to `LANG_CODES` in `translategemma_client.py` and to `TRANSLATE_LANGS` if it needs the pivot.
3. Add translations for every fixed message map (welcome, greetings, thinking, failure, agent hand-off, follow-up and shop texts). Have a native speaker write or review them, because these messages are shown exactly when something has gone wrong.
4. Extend the danger sign list in `medgemma_guardrails.py` with terms from a clinician who speaks the language.
5. Test as described above.

## Clinical safety

Rudo is an information service, not a clinician.

- The danger sign list in `medgemma_guardrails.py` is a starter list: English plus the blood and bleeding words already in the repo. Have a clinician who speaks each language add and review terms before relying on it.
- Machine translation and a 4B model can both be wrong. Do not treat any answer as authoritative, and keep the referral footer on.
- Always keep a route to a human. If the services fail, the bot says so and offers an agent instead of guessing.
- Review a sample of real conversations regularly, with clinicians, and feed corrections back into training data and guardrails.
- If someone may be in an emergency, they should go to the nearest clinic or hospital or call local emergency services.

## Data protection

Rudo handles phone numbers and health related messages.

- Row level security is on for all tables. Keep the Supabase secret key out of source control and browsers.
- Decide how long chat history is kept and delete older messages on a schedule.
- Be transparent with users about what is stored, and get consent where your local law requires it. Check the data protection laws that apply to you, for example Zimbabwe's Cyber and Data Protection Act and Zambia's Data Protection Act.
- If you fine-tune on real conversations, remove or mask names, numbers and other identifying details first, and only use data you have the right to use.
- Language detection sends message text to Google's Gemini API. Take that into account in your privacy notice, or replace the detector.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| A Space crashes with `GatedRepoError` or a 401 for `google/medgemma-4b-it` | The `HF_TOKEN` secret is missing, misnamed, has stray quotes or spaces, or belongs to an account that has not accepted the model terms. Fix the secret and restart the Space |
| Webhook verification fails in Meta | `VERIFY_TOKEN` does not match, or the URL does not end in `/webhook` |
| The bot always replies with the failure message | `MEDGEMMA_URL` includes `/answer` (it must not), `TRANSLATEGEMMA_URL` is missing `/translate`, a token is wrong, or a Space is asleep or crashed. Check each service with the curl commands above |
| Users in a local language get English answers | The pivot is off for that language (check `TRANSLATE_LANGS`), or translation is failing. Check the pipeline log lines |
| Replies are slow or Vercel times out | Lower the per-call timeouts, raise the function duration, and keep the Spaces warm |
| Nothing appears in Supabase | `SUPABASE_URL` or the key is missing, or the publishable key was used instead of the secret key. Redeploy after changing variables |
| `Invalid JWT` from Supabase | A `sb_secret_` key is being sent as a bearer token, or the pasted key has extra characters. `db.py` sends it in the `apikey` header only |
| `relation "kv" does not exist` | `supabase_schema.sql` has not been run |
| Duplicate replies | The message id check is failing. Confirm the `kv` table exists and the Supabase key works |
| Wrong language detected | Gemini is failing and the keyword fallback is guessing. Check `GEN_API` and the logs |

## Contributing

Issues and pull requests are welcome. Please:

- keep fixed user-facing text in the per-language maps and have native speakers review any change
- never commit tokens, keys, phone numbers of real users or exported conversations
- describe how you tested a change, ideally with sample messages in more than one language
- remember that contributions are licensed under the AGPL-3.0, like the rest of the project

For security issues, contact hello@dawa-health.com instead of opening a public issue.

## License

Copyright (C) 2026 Dawa Health

This program is free software: you can redistribute it and modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3. See the [LICENSE](LICENSE) file for the full text.

The AGPL has one requirement that matters for a chatbot. If you run a modified version of this software so that people interact with it over a network, such as through WhatsApp, you must offer those users the corresponding source code of your modified version (section 13). A simple way to do that is to link to your repository from the welcome message or a "source" command.

What the license covers and what it does not:

- It covers the code in this repository.
- It does not change the terms of the models. MedGemma and TranslateGemma are governed by Google's terms for those models (see each model page). The LoRA adapter, the training data and the health content in `training/` may have their own terms and are not licensed by this file unless stated.
- Third party services (WhatsApp, Gemini, Hugging Face, Supabase, Vercel) have their own terms.

## Acknowledgements

Built on [MedGemma](https://huggingface.co/google/medgemma-4b-it) and [TranslateGemma](https://huggingface.co/google/translategemma-4b-it) from Google, [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning and inference, [Gradio](https://www.gradio.app/) and Hugging Face Spaces, [Supabase](https://supabase.com/), [Vercel](https://vercel.com/) and the Meta WhatsApp Cloud API.
