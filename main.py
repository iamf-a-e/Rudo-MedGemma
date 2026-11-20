import json
import logging
import os
import random
import re
import string
from datetime import datetime

import google.generativeai as genai
import requests
from flask import Flask, jsonify, render_template, request
import db

from medgemma_client import ask_medgemma, medgemma_enabled, pairs_from_history
from medgemma_guardrails import clean_model_output, compose_reply
from products_data import products_by_category
from translategemma_client import needs_translation, translate, translategemma_enabled
from training import (  # noqa: F401
    instructions, product_images, products, pregnancy_data, pregnancy_data_shona,
    pregnancy_data_ndebele, pregnancy_data_tonga, pregnancy_data_chinyanja,
    pregnancy_data_bemba, pregnancy_data_lozi, cervical_cancer_data,
    cervical_cancer_data_chinyanja, cervical_cancer_data_lozi,
)

logging.basicConfig(level=logging.INFO, force=True)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
verify_token = os.environ.get("VERIFY_TOKEN", "BOT")

model_name = "gemini-2.5-flash"   # language detection only
genai.configure(api_key=gen_api)

bot_name = "Rudo"

AGENTS = {
    "Agent 1": "+260978760105",
}
AGENT_TIMEOUT_SECONDS = 60

# How many stored messages (user + bot) are passed to MedGemma as context
HISTORY_ENTRIES_FOR_MODEL = 12

app = Flask(__name__)

# Gemini is only asked to classify language, but medical wording can trip the
# default safety filters, so they stay off for that call.
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Supabase. `redis_client` keeps its old name: it is a small key/value layer (db.kv) with the
# same get / set(ex, nx) / delete calls, used for agent hand-offs and message de-duplication.
# Users, chat history, referrals and orders live in their own tables (see db.py).
if db.enabled():
    redis_client = db.kv
    try:
        db.kv.ping()
        logging.info("Successfully connected to Supabase")
    except Exception as e:
        logging.error(f"Supabase connection check failed: {e}")
else:
    redis_client = None
    logging.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not set, persistence disabled")

# Per-worker cache. Redis is the source of truth, the webhook refreshes from it on every message.
user_states = {}


# ─────────────────────────────────────────────
#  SMALL HELPERS
# ─────────────────────────────────────────────

def _contains_signal(prompt_lower: str, phrases: list) -> bool:
    """
    True if any phrase is present in prompt_lower.
    Single words match on word boundaries, multi-word phrases match as substrings.
    """
    for phrase in phrases:
        phrase_l = phrase.lower()
        if " " in phrase_l:
            if phrase_l in prompt_lower:
                return True
        else:
            if re.search(rf"\b{re.escape(phrase_l)}\b", prompt_lower):
                return True
    return False


def normalize_phone(phone: str) -> str:
    return phone.lstrip("+")


AGENT_PHONES_NORM = {normalize_phone(p) for p in AGENTS.values()}


# ─────────────────────────────────────────────
#  PER-USER STATE (Redis backed)
# ─────────────────────────────────────────────

def save_single_user_state(sender):
    if sender in user_states:
        db.save_user_state(sender, user_states[sender])


def load_user_state(sender):
    return db.get_user_state(sender)


def ensure_user_state(sender, refresh=False):
    """
    Make sure `sender` has a state dict in memory. Returns True if the user is brand new.
    refresh=True drops the cached copy first so Redis wins over a stale worker cache.
    """
    if refresh and redis_client:
        user_states.pop(sender, None)

    if sender in user_states:
        return False

    saved = load_user_state(sender)
    if saved:
        user_states[sender] = saved
        return False

    # "language_detection" is kept as the marker for "has not been onboarded yet"
    user_states[sender] = {
        "step": "language_detection",
        "language": "english",
        "registered": False,
        "phone_digits": None,
        "user_id": None,
        "first_message": True,
    }
    return True


# ─────────────────────────────────────────────
#  CONVERSATION HISTORY
# ─────────────────────────────────────────────

def get_user_conversation(sender):
    return db.get_messages(sender)


def save_user_conversation(sender, role, message, message_en=None):
    """message_en is the English version of the message, stored for pivoted languages."""
    db.add_message(sender, role, str(message), message_en)


def set_last_message_en(sender, role, text_en):
    """Attach an English translation to the most recent stored message of `role`."""
    db.set_last_message_en(sender, role, text_en)


def clear_conversation(sender):
    db.delete_messages(sender)


def drop_last_user_message(sender):
    """Remove a bare yes/no from the history so MedGemma never sees it."""
    db.delete_last_message_if_role(sender, "user")


# ─────────────────────────────────────────────
#  SENDING
# ─────────────────────────────────────────────

WHATSAPP_MAX_CHARS = 3800


def _split_for_whatsapp(text: str, limit: int = WHATSAPP_MAX_CHARS):
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for para in text.split("\n"):
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = para
    if current:
        parts.append(current)
    return parts


def send(answer, sender, phone_id_val, message_en=None, save=True):
    """
    Send a WhatsApp text message.
    message_en: English version of the reply, stored with the history for MedGemma context.
    save=False for messages that should not enter the conversation history. Use it for every
    fixed UI text (welcome, greetings, resets, agent hand-offs, failures): history is what
    MedGemma sees, and local-language boilerplate in it makes the model answer in that language.
    """
    url = f"https://graph.facebook.com/v19.0/{phone_id_val}/messages"
    headers = {
        "Authorization": f"Bearer {wa_token}",
        "Content-Type": "application/json",
    }

    response = None
    for part in _split_for_whatsapp(answer):
        data = {
            "messaging_product": "whatsapp",
            "to": sender,
            "type": "text",
            "text": {"body": part},
        }
        response = requests.post(url, headers=headers, json=data, timeout=15)
        logging.info(f"Send status to {sender}: {response.status_code}")
        if response.status_code >= 400:
            logging.error(f"Send failed: {response.text}")

    if save and normalize_phone(sender) not in AGENT_PHONES_NORM:
        save_user_conversation(sender, "bot", answer, message_en)
    return response


def say(text, sender, phone_id_val):
    """Fixed UI text: never stored in the conversation history."""
    return send(text, sender, phone_id_val, save=False)


def send_interactive_buttons(phone_number, body_text, buttons, phone_id_val):
    """buttons: list of {'id', 'title'} (max 3, title max 20 chars)."""
    url = f"https://graph.facebook.com/v19.0/{phone_id_val}/messages"
    headers = {
        "Authorization": f"Bearer {wa_token}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": btn["id"], "title": btn["title"]}}
                    for btn in buttons
                ]
            },
        },
    }
    resp = requests.post(url, headers=headers, json=data, timeout=15)
    logging.info(f"Interactive button send to {phone_number}: {resp.status_code} {resp.text}")
    return resp


# ─────────────────────────────────────────────
#  HUMAN AGENT SYSTEM
# ─────────────────────────────────────────────

HUMAN_AGENT_PHRASES = [
    "human agent", "speak to agent", "talk to agent", "talk to a person",
    "speak to a person", "real person", "human support", "live agent",
    "live support", "connect me to an agent", "i need help from a person",
    "speak to someone", "talk to someone", "customer service",
]
# These words alone are too common in free-form medical questions, so they only
# count when the whole message is very short.
HUMAN_AGENT_SINGLE_WORDS = ["agent", "human", "support"]


def is_human_agent_request(prompt: str) -> bool:
    p = prompt.lower().strip()
    if _contains_signal(p, HUMAN_AGENT_PHRASES):
        return True
    return len(p.split()) <= 3 and _contains_signal(p, HUMAN_AGENT_SINGLE_WORDS)


def _agent_request_key(user_number):
    return f"agent_request:{user_number}"


def _agent_session_key(user_number):
    return f"agent_session:{user_number}"


def _agent_rejections_key(user_number):
    return f"agent_rejections:{user_number}"


def notify_agents_of_request(sender, current_phone_id):
    """Broadcast a chat request to all agents with Accept / Reject buttons."""
    state = user_states.get(sender, {})
    user_id = state.get("user_id", sender)
    lang = state.get("language", "english")

    request_data = {
        "user_number": sender,
        "user_id": user_id,
        "language": lang,
        "phone_id": current_phone_id,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "accepted_by": None,
        "rejections": [],
    }

    if redis_client:
        try:
            redis_client.set(
                _agent_request_key(sender),
                json.dumps(request_data),
                ex=AGENT_TIMEOUT_SECONDS + 10,
            )
            redis_client.set(_agent_rejections_key(sender), json.dumps([]), ex=AGENT_TIMEOUT_SECONDS + 10)
        except Exception as e:
            logging.error(f"Error saving agent request to Redis: {e}")

    body = (
        f"🔔 *New Chat Request*\n\n"
        f"User ID : {user_id}\n"
        f"Language: {lang}\n"
        f"Phone   : {sender}\n\n"
        f"Do you want to accept this chat?"
    )
    buttons = [
        {"id": f"agent_accept:{sender}", "title": "✅ Accept"},
        {"id": f"agent_reject:{sender}", "title": "❌ Reject"},
    ]

    for agent_name, agent_phone in AGENTS.items():
        logging.info(f"Notifying agent {agent_name} ({agent_phone}) of request from {sender}")
        send_interactive_buttons(agent_phone, body, buttons, current_phone_id)


def handle_agent_accept(agent_phone, user_number, current_phone_id):
    """Called when an agent taps Accept."""
    if not redis_client:
        send("Sorry, the agent system is unavailable right now.", user_number, current_phone_id, save=False)
        return

    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            send("⚠️ This chat request has already expired or been accepted by another agent.",
                 agent_phone, current_phone_id, save=False)
            return

        request_data = json.loads(raw) if isinstance(raw, str) else raw

        if request_data.get("status") != "pending":
            send("⚠️ This chat has already been accepted by another agent.", agent_phone, current_phone_id,
                 save=False)
            return

        request_data["status"] = "accepted"
        request_data["accepted_by"] = agent_phone
        redis_client.set(_agent_request_key(user_number), json.dumps(request_data), ex=3600)

        session_data = {
            "user_number": user_number,
            "agent_phone": agent_phone,
            "phone_id": current_phone_id,
            "started_at": datetime.now().isoformat(),
        }
        redis_client.set(_agent_session_key(user_number), json.dumps(session_data), ex=3600)
        redis_client.set(f"agent_user_session:{normalize_phone(agent_phone)}", json.dumps(session_data), ex=3600)

        agent_name = next((n for n, p in AGENTS.items() if p == agent_phone), agent_phone)

        ensure_user_state(user_number, refresh=True)
        user_states[user_number]["step"] = "human_agent_chat"
        user_states[user_number]["agent_phone"] = agent_phone
        save_single_user_state(user_number)

        send(
            f"✅ You are now connected to the user ({user_number}).\n"
            f"Their messages will be forwarded to you. Reply here to send messages to them.\n"
            f"Type *END CHAT* to end the session.",
            agent_phone, current_phone_id, save=False
        )
        send(
            f"✅ Great news! {agent_name} has accepted your chat request.\n"
            f"You are now connected!",
            user_number, current_phone_id, save=False
        )

        for other_name, other_phone in AGENTS.items():
            if other_phone != agent_phone:
                send(
                    f"ℹ️ The chat request from user {user_number} has been accepted by {agent_name}.",
                    other_phone, current_phone_id, save=False
                )

    except Exception as e:
        logging.error(f"Error in handle_agent_accept: {e}", exc_info=True)


def handle_agent_reject(agent_phone, user_number, current_phone_id):
    """Called when an agent taps Reject."""
    if not redis_client:
        return

    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            send("⚠️ This chat request has already expired.", agent_phone, current_phone_id, save=False)
            return

        request_data = json.loads(raw) if isinstance(raw, str) else raw
        if request_data.get("status") != "pending":
            send("ℹ️ This request was already handled.", agent_phone, current_phone_id, save=False)
            return

        rejections_raw = redis_client.get(_agent_rejections_key(user_number))
        if isinstance(rejections_raw, str):
            rejections = json.loads(rejections_raw)
        else:
            rejections = rejections_raw or []
        if agent_phone not in rejections:
            rejections.append(agent_phone)
        redis_client.set(_agent_rejections_key(user_number), json.dumps(rejections), ex=AGENT_TIMEOUT_SECONDS + 10)

        send("👍 You have rejected this chat request.", agent_phone, current_phone_id, save=False)

        if set(rejections) >= set(AGENTS.values()):
            _handle_no_agents_available(user_number, current_phone_id)

    except Exception as e:
        logging.error(f"Error in handle_agent_reject: {e}", exc_info=True)


def _handle_no_agents_available(user_number, current_phone_id):
    """Tell the user nobody is available and hand them back to the bot."""
    ensure_user_state(user_number, refresh=True)
    user_states[user_number]["step"] = "chat"
    user_states[user_number].pop("agent_phone", None)
    save_single_user_state(user_number)

    if redis_client:
        try:
            redis_client.delete(_agent_request_key(user_number))
            redis_client.delete(_agent_rejections_key(user_number))
        except Exception:
            pass

    lang = user_states[user_number].get("language", "english")
    no_agent_map = {
        "shona": (
            "😔 Ndine urombo, hapana mubatsiri anowanikwa parizvino.\n"
            "Ndinokudzoseredzai kuna Rudo, mubatsiri wedu wepamhepo.\n"
            "Pane chimwe chandingakubatsira nacho here?"
        ),
        "ndebele": (
            "😔 Uxolo, awukho umuntu otholakalayo okwamanje.\n"
            "Siyakubuyisela ku-Rudo, umsizi wethu we-inthanethi.\n"
            "Ingabe kukhona okunye engingakusiza ngakho?"
        ),
        "chinyanja": (
            "😔 Pepani, palibe wothandiza ali ndi ntchito pakali pano.\n"
            "Tikubwereza kwa Rudo, wothandiza wathu wa intaneti.\n"
            "Kodi pali zina zomwe ndingakuthandizireni?"
        ),
        "bemba": (
            "😔 Njelelako, tapali mwafwilishi ulipo pali ino nshita.\n"
            "Tulakulekela kuli Rudo, umwafwilishi wesu wa ku intaneti.\n"
            "Kuli fintu fimbi ifyo ningamwafwilisha?"
        ),
        "lozi": (
            "😔 Ni maswabi, ha ku na mubasi ya fumaneha cwale.\n"
            "Lu ku kutiseza kwa Rudo, mubasi wa luna wa intaneti.\n"
            "Ki sina sika ni ka thusa ka sona?"
        ),
        "tonga": (
            "😔 Ndatola, kunyina wakugwasya ulikonzeka lino.\n"
            "Tulamubweza kuli Rudo, wakugwasya wesu wa intaneti.\n"
            "Hena muli amubuyo umbi?"
        ),
    }
    send(
        no_agent_map.get(lang, (
            "😔 Sorry, no agents are available right now.\n"
            "You have been handed back to Rudo, our virtual assistant.\n"
            "Is there anything else I can help you with?"
        )),
        user_number, current_phone_id, save=False
    )


def relay_user_message_to_agent(sender, prompt, current_phone_id) -> bool:
    """Forward a user's message to their agent. Returns True if relayed."""
    if not redis_client:
        return False
    try:
        raw = redis_client.get(_agent_session_key(sender))
        if not raw:
            return False
        session = json.loads(raw) if isinstance(raw, str) else raw
        agent_phone = session.get("agent_phone")
        if not agent_phone:
            return False
        send(f"💬 *User ({sender}):* {prompt}", agent_phone, current_phone_id, save=False)
        return True
    except Exception as e:
        logging.error(f"Error relaying user message to agent: {e}")
        return False


def relay_agent_message_to_user(agent_phone, prompt, current_phone_id) -> bool:
    """
    If this sender is an agent with an active session, forward the message to the user.
    Returns True if handled (the caller must then skip normal bot processing).
    """
    if not redis_client:
        return False

    try:
        norm_agent = normalize_phone(agent_phone)
        raw = redis_client.get(f"agent_user_session:{norm_agent}")
        if not raw:
            return False
        session = json.loads(raw) if isinstance(raw, str) else raw
        user_number = session.get("user_number")
        if not user_number:
            return False

        prompt_stripped = prompt.strip()

        if prompt_stripped.upper() == "END CHAT":
            redis_client.delete(f"agent_user_session:{norm_agent}")
            redis_client.delete(_agent_session_key(user_number))
            redis_client.delete(_agent_request_key(user_number))

            ensure_user_state(user_number, refresh=True)
            user_states[user_number]["step"] = "chat"
            user_states[user_number].pop("agent_phone", None)
            save_single_user_state(user_number)

            send("✅ You have ended the chat session.", agent_phone, current_phone_id, save=False)
            lang = user_states[user_number].get("language", "english")
            end_map = {
                "shona": (
                    "👋 Mubatsiri wangu akunge ngumi chisarai.\n"
                    "Madzoka kumubatsiri wedu wepamhepo Rudo.\n"
                    "Pane chimwe chandingakubatsira nacho?"
                ),
                "ndebele": (
                    "👋 Umhloli wami uphethile inkulumo yakho.\n"
                    "Ubuyela ku-Rudo, umsizi wethu we-inthanethi.\n"
                    "Ingabe kukhona okunye engingakusiza ngakho?"
                ),
                "chinyanja": (
                    "👋 Wothandiza wanu wamaliza kuchatitana nawo.\n"
                    "Mumabwerera kwa Rudo, wothandiza wathu wa intaneti.\n"
                    "Kodi pali zina zomwe ndingakuthandizireni?"
                ),
                "bemba": (
                    "👋 Umwafwilishi wenu ashile ukumana nawe.\n"
                    "Mwabwela kuli Rudo, umwafwilishi wesu wa ku intaneti.\n"
                    "Kuli fintu fimbi ifyo ningamwafwilisha?"
                ),
                "lozi": (
                    "👋 Mubasi wa hao u felelize puisano ya hao.\n"
                    "U bwela kwa Rudo, mubasi wa luna wa intaneti.\n"
                    "Ki sina sika ni ka thusa ka sona?"
                ),
                "tonga": (
                    "👋 Wakugwasya wanu wamanizya kulumbaana anywi.\n"
                    "Mubweza kuli Rudo, wakugwasya wesu wa intaneti.\n"
                    "Hena muli amubuyo umbi?"
                ),
            }
            send(
                end_map.get(lang, (
                    "👋 Your agent has ended the chat session.\n"
                    "You have been returned to Rudo, our virtual assistant.\n"
                    "Is there anything else I can help you with?"
                )),
                user_number, current_phone_id, save=False
            )
            return True

        send(f"💬 *Agent:* {prompt_stripped}", user_number, current_phone_id, save=False)
        return True

    except Exception as e:
        logging.error(f"Error relaying agent message to user: {e}")
        return False


def check_agent_request_timeout(user_number, current_phone_id):
    """If the agent request key has expired, hand the user back to the bot."""
    if not redis_client:
        _handle_no_agents_available(user_number, current_phone_id)
        return
    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            _handle_no_agents_available(user_number, current_phone_id)
    except Exception as e:
        logging.error(f"Error checking agent timeout: {e}")
        _handle_no_agents_available(user_number, current_phone_id)


# ─────────────────────────────────────────────
#  REFERRAL SOURCE TRACKING
# ─────────────────────────────────────────────

REFERRAL_TRIGGERS = [
    "got connected from", "i got connected from", "i was connected from",
    "connected from", "referred from", "coming from", "came from",
    "i came from", "i'm from", "im from", "from the poster",
    "from a poster", "from flyer", "from a flyer", "from the flyer",
    "saw your poster", "saw a poster", "saw the poster",
    "hey dawamom", "hi dawamom", "hello dawamom",
]


def extract_referral_source(prompt: str):
    """Returns the referral string (for example 'Mandebvu Pharmacy Poster 1') or None."""
    prompt_lower = prompt.lower().strip()

    if not any(t in prompt_lower for t in REFERRAL_TRIGGERS):
        return None

    for prep in ["from ", "from the ", "from a "]:
        idx = prompt_lower.rfind(prep)
        if idx != -1:
            source = prompt[idx + len(prep):].strip().rstrip(".,!?;:")
            if source:
                return source

    for trigger in sorted(REFERRAL_TRIGGERS, key=len, reverse=True):
        idx = prompt_lower.find(trigger)
        if idx != -1:
            source = prompt[idx + len(trigger):].strip().rstrip(".,!?;:")
            if source:
                return source

    return None


def save_referral_source(sender, source):
    db.add_referral(sender, source, user_states.get(sender, {}).get("user_id", "unregistered"))
    logging.info(f"Referral saved for {sender}: {source}")


# QR codes open WhatsApp with a prefilled "Hi DawaMom, I got connected from <place>" message.
_PREFILLED_REFERRAL = re.compile(r"^\s*(hi|hello|hey)\s+dawamom\b", re.I)


def is_referral_greeting(prompt: str) -> bool:
    return bool(_PREFILLED_REFERRAL.match(prompt)) and extract_referral_source(prompt) is not None


# ─────────────────────────────────────────────
#  LANGUAGE DETECTION (Gemini primary, keywords as offline fallback)
# ─────────────────────────────────────────────

SUPPORTED_LANGS = ["english", "shona", "ndebele", "chinyanja", "bemba", "tonga", "lozi"]

EXACT_MATCHES = {
    "shona": [
        "mhoro", "mhoroi", "makadini", "hesi",
        "maswera", "mangwanani", "manheru",
        "ndinonzi", "ndatenda", "ndinotenda",
        "ndapota", "ndibatsirei", "batsirai",
        "pamuviri", "nhumbu", "chibereko",
        "chiremba", "kurwadziwa", "kusvotwa",
        "gomarara", "zviratidzo", "muviri",
    ],
    "ndebele": [
        "sawubona", "salibonani", "unjani", "linjani",
        "ngiyabonga", "siyabonga", "ngicela", "ngiyacela",
        "ngisize", "ngidinga", "angikwazi",
        "ukukhulelwa", "isibeletho", "izimpawu",
        "udokotela", "ubuhlungu", "umhlaza",
        "imikhiqizo",
    ],
    "bemba": [
        "mwaiseni", "ulishani", "muli shani",
        "nalikutemwa", "natotela", "twatotela",
        "njelelako", "napapata", "ngafweniko",
        "nshishibe", "bushe kuti",
        "umwanakashi", "abanakashi", "wanakashi",
        "fumo", "kufuma", "iloba",
        "ifyakulya", "cilikwisa", "ubushiku",
        "ukubomba", "icisungu", "icibemba",
        "shaleenipo", "ubuchende",
    ],
    "chinyanja": [
        "moni", "zikomo", "zikomo kwambiri",
        "pepani", "ndapota", "chonde",
        "nitandizeni", "nankani", "thandizani",
        "ndithandizeni", "sindikudziwa",
        "ndikufuna", "ndikumva", "ndinafuna",
        "matenda", "zizindikiro", "dokotala",
        "mavitamini", "thanzo",
    ],
    "tonga": [
        "mwabuka", "mwabukile", "mwalandwa",
        "ndatotela", "lugwazyo", "mubuzyo",
        "zitondezyo", "mutumbu", "dokota",
        "kusilikwa", "mbubo", "buumi",
        "chibadela", "kaambo nzi", "kaambo",
    ],
    "lozi": [
        "ndalumba", "ndalumba hahulu",
        "zibonelelo", "kuhula", "kushisa",
        "maviki", "mutango", "mupilo",
        "mubonelelo", "kacenu", "muzuhile",
        "kimanzibwana", "mulumele",
        "silelezwa", "musimbi", "bulwazi",
        "wapimwa",
    ],
}

LANGUAGE_KEYWORDS = {
    "shona": [
        "mhoro", "mhoroi", "makadini", "maswera sei",
        "mamuka sei", "hesi", "ko sei",
        "ndatenda", "ndinotenda", "maita basa", "maita henyu",
        "ndapota", "ndibatsirei", "batsirai",
        "ndinonzi", "ndiri", "ndine",
        "ndinoda", "unoda", "anoda",
        "ndinoziva", "ndoziva", "handizivi", "handina",
        "zvino", "zvakare", "zvakadaro",
        "zvakanaka", "zvakaipa", "zvichida",
        "zvinobvira", "zviripo", "hapana",
        "hongu", "kwete",
        "sei", "nei", "chii", "ndiani",
        "kupiko", "riini", "papi", "kupi",
        "asi", "uye", "kana", "nekuti", "saka",
        "zvekuti", "kana kuti",
        "nditsanangurirei", "ndibatsirei",
        "ndaka", "ndichada", "ndichaita",
        "ndakaita", "ndakadaro",
        "pamuviri", "nhumbu", "nhumbu yangu",
        "zvigadzirwa", "chirwere", "gomarara",
        "chibereko", "zviratidzo", "chiremba",
        "kusvotwa", "kurwadziwa",
        "kubuda ropa", "mushonga", "mishonga",
        "chipatara", "kiriniki", "utano",
        "kubereka", "kusununguka",
    ],
    "ndebele": [
        "sawubona", "salibonani", "unjani", "linjani",
        "livuke njani", "kusile njani",
        "ngiyabonga", "siyabonga",
        "ngiyacela", "ngicela",
        "ngisize", "ngidinga",
        "ngingu", "ngiyazi", "uyazi",
        "angikwazi", "angazi",
        "ngifuna", "ngiyafuna",
        "sifuna", "bafuna",
        "nginayo", "unayo", "unalo",
        "ngikhona", "ukhona",
        "yebo", "hatshi", "akukho",
        "khona", "lapha", "lapho",
        "ngani", "kungani", "yini", "ubani",
        "nini", "kuphi", "njani",
        "kodwa", "futhi", "ngoba", "ngakho",
        "kakhulu", "manje", "mhlawumbe",
        "ukwenza", "ukwazi", "ukufuna",
        "ukusiza", "ukukhuluma",
        "ukubuza", "ukuphendula",
        "ngitshele", "ngichazele",
        "ukukhulelwa", "isisu", "umntwana",
        "imikhiqizo", "umhlaza", "isibeletho",
        "izimpawu", "udokotela", "igazi",
        "ubuhlungu", "umuthi", "imithi",
        "isibhedlela", "umtholampilo",
        "impilo", "ukubeletha", "ukuzala",
    ],
    "chinyanja": [
        "moni", "muli bwanji", "zikomo",
        "zikomo kwambiri", "pepani",
        "chonde", "ndapota",
        "nankani", "thandizani",
        "ndithandizeni", "nitandizeni",
        "ndili", "ulipo", "alipo",
        "tilipo", "mulipo",
        "ndikufuna", "sindikufuna",
        "akufuna", "tikufuna", "mukufuna",
        "ndikudziwa", "sindikudziwa",
        "ndimva", "ndikumva",
        "ndinafuna", "ndinati",
        "inde", "ayi",
        "pano", "pamenepo",
        "kuno", "kuti",
        "bwanji", "chiyani", "ndani", "liti",
        "chifukwa chiyani",
        "koma", "komanso", "chifukwa", "choncho",
        "tsopano", "kwambiri",
        "kuchita", "kufuna", "kudziwa",
        "kuthandiza", "kulankhula",
        "kufunsa", "kuyankha",
        "ndiuzeni", "ndifotokozereni",
        "matenda", "kansa", "zizindikiro",
        "dokotala", "magazi", "mavitamini",
        "thanzo", "ululu", "kupweteka",
        "mankhwala", "chipatala",
        "mimba", "kubereka", "umoyo",
    ],
    "lozi": [
        "ndalumba", "ndalumba hahulu",
        "mulumele", "kacenu",
        "muzuhile", "silelezwa",
        "zibonelelo", "kuhula", "kushisa",
        "maviki", "mutango", "mupilo",
        "mubonelelo", "kimanzibwana",
        "butuku", "musimbi",
        "bulwazi", "wapimwa",
        "ki", "kikani", "kuli",
        "mang", "kai", "kuzwa",
        "kuli cwa", "kacani", "mangani",
        "ni batla", "ha ni", "ha ku",
        "dokota", "mwana", "musali",
        "mukuli", "mushonga", "chipatela",
        "mutumbu", "butuku", "bulwazi",
    ],
    "bemba": [
        "mwaiseni", "ulishani", "muli shani",
        "mwabukeni", "mwabuka",
        "natotela", "twatotela",
        "napapata", "njelelako",
        "nshishibe", "shishibe",
        "ndefwaya", "ngafwaya",
        "tufwaya", "mufwaya",
        "nshifwaya", "shifwaya",
        "ndifye", "alifye",
        "takwaba", "te",
        "bushe", "bushe kuti",
        "kuti", "inshi", "nani",
        "lini", "kwisa", "shani", "bwanji",
        "kabili", "lelo", "pali", "pano",
        "kanshi", "nangu", "pantu",
        "ukubomba", "ukufwaya",
        "ukwishiba", "ukwita",
        "ukupoka", "ukupala",
        "ukwafya", "landeni",
        "twalanda",
        "umwanakashi", "abanakashi", "wanakashi",
        "fumo", "kufuma", "iloba",
        "ifyakulya", "cilikwisa", "ubushiku",
        "icisungu", "icibemba",
        "ubuchende", "ubumi", "imiti",
        "icipatala", "dokota",
    ],
    "tonga": [
        "mwabuka", "mwabukile", "mwalandwa",
        "mwalibuka", "mwaabuka",
        "ndatotela", "ndatotela kapati",
        "lugwazyo", "mubuzyo", "zitondezyo",
        "mutumbu", "dokota", "kusilikwa",
        "mbubo", "buumi", "chibadela",
        "kaambo nzi", "kaambo",
        "nzi", "nzi kuti",
        "nkani", "muno", "kuno",
        "kwakuli", "takwe",
        "ndiyanda", "ndiyanda kuti",
        "tuyanda", "muyanda",
        "ndikonzya", "takonzyi",
        "ndilizi", "tandizi",
        "nani", "lini", "kwi",
        "butami", "malwazi",
        "dokota", "mwana", "mama",
        "chibadela", "mutumbu",
        "kusilikwa", "zitondezyo",
        "mubili", "mukaintu",
    ],
    "english": [
        "the", "and", "or", "but", "because",
        "with", "from", "about",
        "this", "that", "these", "those",
        "what", "why", "when", "where",
        "who", "how", "which",
        "can", "could", "would", "should",
        "will", "shall",
        "have", "has", "had",
        "does", "did", "is", "are",
        "was", "were", "please",
        "yes", "no", "thanks", "thank you",
        "want", "need", "know", "think",
        "understand", "help", "tell",
        "explain", "give", "get",
        "make", "go", "come",
        "feel", "say", "ask", "answer",
        "symptoms", "information", "pregnancy",
        "pregnant", "doctor", "medicine",
        "medication", "cancer", "cervix",
        "uterus", "bleeding", "pain",
        "nausea", "vomiting",
        "antenatal", "maternal",
        "health", "appointment", "clinic",
        "hospital", "baby", "mother",
        "birth", "delivery",
        "quote",
    ],
}

LANGUAGE_PHRASES = {
    "chinyanja": [
        "muli bwanji", "uli ndi chani", "zikomo kwambiri", "muli bwino",
        "ndili bwino", "nitandizeni nankani", "ndithandizeni chonde",
        "chifukwa chiyani", "ndikufuna kudziwa", "sindikudziwa",
        "sindikufuna", "ndikufuna thandizo", "ndifotokozereni",
        "ndiuzeni chonde", "mukutanthauza chiyani",
    ],
    "shona": [
        "makadini", "maswera sei", "mamuka sei", "zvakadii",
        "zvakanaka sei", "ndatenda zvikuru", "maita basa", "ndiri bho",
        "ndiriku", "ndiri kuda", "ndinoda kuziva", "handizivi",
        "handina chokwadi", "ndibatsireiwo", "ndibatsirei chiremba",
        "nditsanangurirei", "ndiudzeiwo", "chii chiri kuitika",
        "zvinoita sei", "zvinorevei", "sei zvichidaro",
    ],
    "ndebele": [
        "unjani wena", "linjani wena", "ngiyabonga kakhulu",
        "siyabonga kakhulu", "sicela ungichazele", "ngicela ungisize",
        "ngicela ungitshele", "ngifuna ukwazi", "angikwazi",
        "angazi ukuthi", "ngidinga usizo", "ngifuna usizo",
        "ngitshele ukuthi", "ngichazele ukuthi", "kwenzakalani",
        "kwenzenjani", "kutshoni lokhu", "kungani kunjalo",
        "kuyini lokhu",
    ],
    "lozi": [
        "uli bwanji", "ni bata", "ha ndi zibi", "ndalumba hahulu",
        "ndalumba kapati", "muli bwanji", "ni batla", "ha ni zibi",
        "ni tusa", "ni tusa hahulu", "ndifundise", "ni bolelele",
        "ki kaambang", "kikani seo",
    ],
    "bemba": [
        "muli shani", "napapata", "nshishibe", "bushe kuti",
        "mwaiseni", "ulishani", "muli shani imwe", "natotela sana",
        "twatotela sana", "ndefwaya ukwishiba", "nshifwaya", "ndefwaya",
        "ngafweniko", "njelelako", "ndefwaya ukweba", "muntungileko",
        "bushe kuti mwangafwa", "inshi cintu ici", "kwisa kuti",
    ],
    "tonga": [
        "mmuli buti", "ndakomba", "sena kuti", "ndatotela kapati",
        "kaambo nzi", "ndiyanda kuti", "ndikonzya", "takonzyi",
        "tandizi", "ndilizi", "ndiyanda kuziba", "ndikomba kuti",
        "muni ndibatsire", "ndibatsire chonde", "nzi kuti", "nzi kaambo",
    ],
    "english": [
        "how are you", "what are", "what is", "what does", "what do you",
        "can you", "could you", "would you", "tell me", "tell me about",
        "i need", "i want", "i would like", "please tell", "please explain",
        "please help", "help me", "how do i", "how can i", "what should",
        "what should i", "what can i", "why is", "why are", "where can i",
        "when should i", "watch out", "i don't know", "i am pregnant",
        "i am having",
    ],
}


def _llm_detect_language(message: str):
    """Gemini language classifier. Returns a language name or None."""
    try:
        classifier_prompt = (
            "Identify which ONE of these languages the following WhatsApp message "
            "is written in: english, shona, ndebele, chinyanja, bemba, tonga, lozi. "
            "These are languages spoken in Zimbabwe and Zambia. The message may mix "
            "in a few English loanwords (e.g. medical terms like 'cervical cancer' "
            "or 'HPV') while still being primarily one of the other languages. In "
            "that case, classify by the surrounding grammar/vocabulary, not the "
            "loanwords. Reply with ONLY the single lowercase language name and "
            "nothing else, no punctuation, no explanation.\n\n"
            f"Message: \"{message}\""
        )
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": 0, "max_output_tokens": 10},
            safety_settings=safety_settings,
        )
        response = gemini_model.generate_content(classifier_prompt)

        if not response.candidates:
            logging.warning("[_llm_detect_language] No candidates returned at all")
            return None

        candidate = response.candidates[0]
        if candidate.finish_reason != 1:  # 1 == STOP
            logging.warning(
                f"[_llm_detect_language] Non-STOP finish_reason={candidate.finish_reason} "
                f"safety_ratings={getattr(candidate, 'safety_ratings', None)}"
            )
            return None

        guess = re.sub(r"[^a-z]", "", response.text.strip().lower())
        if guess in SUPPORTED_LANGS:
            logging.info(f"[_llm_detect_language] Classified as: {guess}")
            return guess
        logging.warning(f"[_llm_detect_language] Unrecognised output: {guess!r}")
    except Exception as e:
        logging.error(f"[_llm_detect_language Error] {type(e).__name__}: {e}")
    return None


def detect_language(message, sender=None):
    message_lower = message.lower().strip()

    current_lang = "english"
    if sender and sender in user_states:
        current_lang = user_states[sender].get("language", "english")

    # Pure digits are never a language signal
    if message_lower.isdigit():
        return current_lang

    # Fast path: exact single-word matches
    for lang, words in EXACT_MATCHES.items():
        if message_lower in words:
            logging.info(f"Exact match: {message_lower} -> {lang}")
            return lang

    # Primary: Gemini classification
    llm_guess = _llm_detect_language(message)
    if llm_guess:
        if llm_guess != current_lang:
            logging.info(f"[LLM] Language switch for {sender}: {current_lang} -> {llm_guess}")
        return llm_guess

    logging.warning(
        f"[detect_language] Gemini classification failed/empty for {sender!r}, "
        f"falling back to keyword scoring"
    )

    # Offline fallback: keyword and phrase scoring
    scores = {lang: 0 for lang in LANGUAGE_KEYWORDS}

    for lang, phrases in LANGUAGE_PHRASES.items():
        for phrase in phrases:
            if phrase in message_lower:
                scores[lang] = scores.get(lang, 0) + 5

    for lang, keywords in LANGUAGE_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", message_lower):
                scores[lang] = scores.get(lang, 0) + 3

    max_score = max(scores.values()) if scores else 0

    if max_score > 0:
        top_langs = [lang for lang, s in scores.items() if s == max_score]
        logging.info(f"[fallback] Keyword scores: {scores} -> candidates: {top_langs}")
        if len(top_langs) == 1:
            return top_langs[0]
        if current_lang in top_langs:
            return current_lang
        return top_langs[0]

    logging.info(f"[fallback] No signal at all for {sender!r}, keeping {current_lang}")
    return current_lang


def maybe_update_language(sender, prompt):
    """Re-detect the language on each message and store it. Returns the current language."""
    state = user_states[sender]
    current_lang = state.get("language", "english")
    stripped = prompt.strip()

    if not stripped or stripped.isdigit():
        return current_lang

    detected = detect_language(stripped, sender)

    # A one or two word message ("ok", "thanks") that looks English is not enough
    # evidence to flip a user who has been writing in a local language.
    if len(stripped.split()) <= 2 and detected == "english" and current_lang != "english":
        return current_lang

    if detected != current_lang:
        logging.info(f"[maybe_update_language] Language switch for {sender}: {current_lang} -> {detected}")
        state["language"] = detected
        save_single_user_state(sender)

    return state["language"]


ENGLISH_TELLS = {
    "the", "is", "are", "you", "your", "how", "many", "months", "week", "weeks",
    "should", "would", "could", "will", "does", "doctor", "please", "thank",
    "and", "with", "this", "that", "have", "has", "when", "what", "can",
}


def looks_like_english_leak(text: str, lang: str) -> bool:
    """Output check for the local-language reply: too much English means the translation failed."""
    if lang == "english" or not text:
        return False
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 6:
        return False
    hits = sum(1 for w in words if w in ENGLISH_TELLS)
    return (hits / len(words)) > 0.15


# Extra everyday English words, so list style medical replies ("- Missed period, nausea, ...")
# are not mistaken for non-English text.
_ENGLISH_CHECK_WORDS = ENGLISH_TELLS | {
    "of", "to", "in", "for", "if", "or", "it", "be", "may", "also", "from", "not", "but",
    "pregnancy", "pregnant", "baby", "clinic", "test", "period", "symptoms", "include",
}
_ENGLISH_MIN_RATIO = 0.08


def looks_english(text: str) -> bool:
    """
    Inverse of looks_like_english_leak: True when `text` reads as English.
    Short text is given the benefit of the doubt. Used on the MedGemma reply before it is
    translated, because garbled local-language output from the model would otherwise be
    'translated' into something even further from the truth.
    """
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    if len(words) < 6:
        return True
    hits = sum(1 for w in words if w in _ENGLISH_CHECK_WORDS)
    return (hits / len(words)) >= _ENGLISH_MIN_RATIO


# ─────────────────────────────────────────────
#  ONBOARDING
# ─────────────────────────────────────────────

def auto_register(sender):
    """Derive the last 4 digits from the WhatsApp number and create the user ID."""
    state = user_states[sender]
    digits = re.sub(r"\D", "", sender)[-4:]
    letters = ''.join(random.choices(string.ascii_uppercase, k=4))
    state["phone_digits"] = digits
    state["user_id"] = f"DH-{digits}-{letters}"
    state["registered"] = True
    state["step"] = "chat"
    state["first_message"] = False
    return state["user_id"]


# Google Form with the terms and the AI training choice ("Yes" pre-selected). Commas and
# brackets are percent-encoded so WhatsApp keeps the whole link clickable.
CONSENT_FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLScGYIW_T7VgedVrJSzO1QPDwmy33Yp3G2ms8I4yJSh-cQ3Q8w/viewform"
    "?usp=pp_url"
    "&entry.1221265732=Yes%2C+I+opt+in.+Dawa+Health+may+use+my+conversations+with+Rudo+on+WhatsApp"
    "%2C+with+my+name+and+contact+details+removed%2C+to+train+and+improve+%28fine-tune%29+its+AI+models."
    "&entry.1012319286=The+text+of+my+messages%2C+including+what+my+voice+notes+say"
)

# Sent once, right after the welcome, to every new number, with a button that opens the form
# so the long link never shows. Add reviewed translations keyed by language ("shona",
# "bemba", ...). Languages without one get the English text.
TERMS_NOTICE_MAP = {
    "english": (
        "By chatting with me, you agree to Dawa Health's terms, and Dawa Health may use your "
        "conversations, with your name and number removed, to improve its AI. Saying no does "
        "not change the help you get from me. You can read the terms or say no here:"
    ),
}
# Button label. WhatsApp allows at most 20 characters.
TERMS_BUTTON_MAP = {
    "english": "Read terms / say no",
}


def send_cta_url(phone_number, body_text, button_text, url, phone_id_val):
    """A message with one button that opens `url`, so the raw link is not shown."""
    endpoint = f"https://graph.facebook.com/v19.0/{phone_id_val}/messages"
    headers = {
        "Authorization": f"Bearer {wa_token}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": phone_number,
        "type": "interactive",
        "interactive": {
            "type": "cta_url",
            "body": {"text": body_text},
            "action": {
                "name": "cta_url",
                "parameters": {"display_text": button_text[:20], "url": url},
            },
        },
    }
    resp = requests.post(endpoint, headers=headers, json=data, timeout=15)
    logging.info(f"CTA URL send to {phone_number}: {resp.status_code}")
    if resp.status_code >= 400:
        logging.error(f"CTA URL send failed: {resp.text}")
    return resp


def send_terms_notice(sender, phone_id_val, lang):
    """Terms notice with a button. If WhatsApp rejects the button, send the plain link."""
    text = TERMS_NOTICE_MAP.get(lang, TERMS_NOTICE_MAP["english"])
    button = TERMS_BUTTON_MAP.get(lang, TERMS_BUTTON_MAP["english"])
    try:
        resp = send_cta_url(sender, text, button, CONSENT_FORM_URL, phone_id_val)
        if resp.status_code < 400:
            return
    except Exception as e:
        logging.error(f"CTA URL send error: {type(e).__name__}: {e}")
    say(f"{text}\n{CONSENT_FORM_URL}", sender, phone_id_val)


def send_welcome_with_id(sender, phone_id_val, lang, user_id):
    welcome_map = {
        "shona": f"Mhoro! Ndinonzi Rudo, mubatsiri wepamhepo weDawa Health. ID yenyu ndeye: {user_id}. Chengetedza ID iyi nekuti ichakumbirwa kumaDawa clinics. Ndingakubatsirei nhasi?",
        "ndebele": f"Sawubona! Ngingu Rudo, isiphathamandla se-Dawa Health. I-ID yakho ithi: {user_id}. Gcina le ID ngoba izocelwa kumaDawa clinics. Ngingakusiza ngani namuhla?",
        "bemba": f"Mwaiseni! Nine Rudo, wakufwailisha wa Dawa Health. ID yenu ni: {user_id}. Sungeni ID iyi pantu ikabombwa ku Dawa clinics. Nga kuti namwafwa shani lelo?",
        "chinyanja": f"Moni! Ndine Rudo, mphungu wa Dawa Health. ID yanu ndi: {user_id}. Sungani ID iyi chifukwa idzafunsidwa kumakliniki a Dawa. Ndingakuthandizireni lero?",
        "tonga": f"Muli buti! Ndime Rudo, wakugwasya Dawa Health. ID yenu nji: {user_id}. Mweelede kuisunga kabotu ID kambo iyakubeleka ku Dawa clinics. Nga ndamukyasya buti lino?",
        "lozi": f"Mwa bona! Mina ki Rudo, mubasi wa ku thusa wa Dawa Health. ID ya hao ki: {user_id}. Boloka ID ye hantši kakuli u ta buzwa yona kwa makiliniki a Dawa. Nka ku thusa ka mini sunu?",
    }
    welcome = welcome_map.get(
        lang,
        f"Hello! I'm Rudo, Dawa Health's virtual assistant. Your ID is: {user_id}. "
        f"Keep this ID safe because it'll be asked for at the Dawa clinics. How can I help you today?"
    )
    send(
        welcome,
        sender, phone_id_val,
        save=False,   # fixed text, must not reach MedGemma through the history
    )
    send_terms_notice(sender, phone_id_val, lang)


GREET_MAP = {
    "shona": "Mhoroi! Ndingakubatsirei nhasi?",
    "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
    "chinyanja": "Moni! Ndingakuthandizireni lero?",
    "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
    "tonga": "Mwabuka buti! Nga ndamugwasya buti sunu?",
    "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
}

GREETING_PHRASES = [
    "hi", "hello", "hey", "hie", "hi there",
    "good morning", "good afternoon", "good evening",
    "mhoro", "mhoroi", "hesi", "makadini", "wadini",
    "sawubona", "salibonani",
    "moni", "muli bwanji",
    "mwabuka", "mwabuka buti", "mwatambulwa", "buti", "mbuti",
    "mwalandwa", "mwalandwa buti",
    "mwaiseni", "muli shani", "shani", "muli buti",
    "mwa bona",
]
_GREETING_FILLERS = {"there", "rudo", "dawa", "dawamom", "all", "everyone", "sir", "madam", "ma"}
GREETING_VOCAB = {w for phrase in GREETING_PHRASES for w in phrase.split()} | _GREETING_FILLERS

RESET_PHRASES = ["start over", "restart", "new conversation", "reset"]


def is_greeting_only(prompt: str) -> bool:
    """True when the message is nothing but a greeting ('hi', 'muli shani', 'good morning Rudo')."""
    tokens = re.findall(r"[\w']+", prompt.lower())
    if not tokens or len(tokens) > 5:
        return False
    return all(t in GREETING_VOCAB for t in tokens)


def is_reset_request(prompt: str) -> bool:
    return _contains_signal(prompt.lower().strip(), RESET_PHRASES)


# ─────────────────────────────────────────────
#  USER FACING FIXED TEXT
# ─────────────────────────────────────────────

THINKING_MAP = {
    "shona": "Ndiri kufunga...",
    "ndebele": "Ngiyacabangisisa...",
    "chinyanja": "Ndikuganiza...",
    "tonga": "Ndichiyandaula...",
    "bemba": "ndefwailisha...",
    "lozi": "Ni nahana...",
}

FALLBACK_MAP = {
    "shona": "Pane dambudziko pakupindura mubvunzo wako.",
    "ndebele": "Kunenkinga ekuphenduleni umbuzo wakho.",
    "chinyanja": "Pali vuto popanga yankho la funso lanu.",
    "tonga": "Kuli ipenzi mukupandula mubuzyo wanu.",
    "bemba": "Cabulanda, kuliko ubwafya pakwasuka kuyasuka ilipusho lyobe.",
    "lozi": "Ku na bothata ka ku arabela lipuzo la hao.",
}

# The word "agent" is the trigger that connects a person, so it stays in English.
# Replace these with proper translations in each language when you have them.
AGENT_HINT = " You can also type *agent* to speak to a person."


def failure_message(lang: str) -> str:
    base = FALLBACK_MAP.get(lang, "Sorry, there was a problem getting an answer.")
    return base + AGENT_HINT


NON_TEXT_MAP = {
    "shona": "Ndine urombo, handigoni kugamuchira mameseji asiri mavara chete. Ndapota tumira meseji yemavara.",
    "ndebele": "Uxolo, angikwazi ukwamukela imilayezo engeyona imibhalo kuphela. Sicela uthumele umlayezo wombhalo.",
    "bemba": "Njelelako, Dekwanisha fye ukumona ama text meseji. Napapata, lemba meseji.",
    "chinyanja": "Pepani, sindingathe kulandira mameseji enama osati a zilembo. Chonde tumirani meseji ya zilembo.",
    "tonga": "Ndausa, tandikonzya kubelesya mameseji aali kunze a mabbala. Ndalomba tumizya meseji ya mabbala.",
    "lozi": "Ni maswabi, ha na kona kuzwela miiala yeng'wi kufita feela ya mangolo. Ndapota, lumeza molaala wa mangolo.",
}


# ─────────────────────────────────────────────
#  ANSWER PIPELINE: TranslateGemma + MedGemma
# ─────────────────────────────────────────────

def _retry(fn, attempts=2):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            logging.warning(f"[retry] attempt {i + 1}/{attempts} failed: {type(e).__name__}: {e}")
    raise last


def _history_for_model(sender, lang):
    """
    Stored history shaped for pairs_from_history.

    * The newest stored entry is the current user message. It is dropped here, and a blank
      sentinel entry is appended because pairs_from_history always drops the last entry.
      The current question is passed to the model separately.
    * For pivoted languages only entries that have an English version (message_en) are kept.
      Local-language text must never reach MedGemma: with a weak grasp of these languages it
      starts answering in broken Shona/Ndebele/etc. instead of English.
    * English users have no pivot, so their plain message text is used.
    """
    history = get_user_conversation(sender)[-HISTORY_ENTRIES_FOR_MODEL:]
    out = []
    for entry in history[:-1]:
        text = entry.get("message_en")
        if not text and lang == "english":
            text = entry.get("message", "")
        if text and text.strip():
            out.append({**entry, "message": text})
    out.append({"role": "user", "message": ""})   # sentinel, dropped by pairs_from_history
    return out


def generate_reply(sender, prompt, lang):
    """
    Returns (reply_in_user_language, reply_in_english_or_None).

    Pivoted languages: prompt -> English -> MedGemma -> guardrails -> back to user language.
    compose_reply always runs, even if translation or the model failed, so the danger sign
    referral can still fire from the user's own words.
    """
    pivot = needs_translation(lang)
    q_en = None
    reply_en = None
    local_text = None

    try:
        if pivot:
            if not translategemma_enabled():
                raise RuntimeError(f"TranslateGemma is not configured but {lang} needs it")
            q_en = _retry(lambda: translate(prompt, lang, "english"))
            set_last_message_en(sender, "user", q_en)

        if not medgemma_enabled():
            raise RuntimeError("MedGemma is not configured")

        model_question = q_en if pivot else prompt
        history = pairs_from_history(_history_for_model(sender, lang))
        logging.info(f"[pipeline] history_to_model={history!r}")
        reply_en = clean_model_output(ask_medgemma(model_question, history=history))

        # The model must answer in English. If it did not, retry once with no history,
        # and if that fails too, drop the reply so the user gets the failure message
        # instead of garbled medical advice.
        if pivot and reply_en and not looks_english(reply_en):
            logging.warning(
                f"[generate_reply] MedGemma reply is not English, retrying without history: {reply_en!r}"
            )
            reply_en = clean_model_output(ask_medgemma(model_question, history=[]))
            if reply_en and not looks_english(reply_en):
                logging.error(f"[generate_reply] retry also not English, discarding: {reply_en!r}")
                reply_en = None

        if reply_en:
            if pivot:
                local_text = _retry(lambda: translate(reply_en, "english", lang))
                if looks_like_english_leak(local_text, lang):
                    logging.warning(f"[generate_reply] English leak after translation for lang={lang}, discarding")
                    local_text = None
            else:
                local_text = reply_en

    except Exception as e:
        logging.error(f"[generate_reply] {type(e).__name__}: {e}", exc_info=True)

    logging.info(f"[pipeline] prompt={prompt!r} q_en={q_en!r} reply_en={reply_en!r} local={local_text!r}")

    # Guardrails get the original wording and, when we have it, the English translation,
    # so danger sign checks work whichever language the keywords live in.
    guard_question = f"{prompt}\n{q_en}" if q_en else prompt
    final = compose_reply(guard_question, lang, local_text)

    if final:
        return final, (reply_en if local_text else None)
    return None, None


def answer_user(sender, prompt, phone_id_val):
    lang = user_states[sender].get("language", "english")

    if len(prompt.split()) > 2:
        send(THINKING_MAP.get(lang, "Let me think..."), sender, phone_id_val, save=False)

    reply, reply_en = generate_reply(sender, prompt, lang)
    failed = not reply
    if failed:
        reply = failure_message(lang)
        reply_en = None

    # A failure message is fixed local text, so keep it out of the history.
    send(reply, sender, phone_id_val, message_en=reply_en, save=not failed)

    if failed:
        user_states[sender]["step"] = "chat"
        save_single_user_state(sender)
    else:
        ask_follow_up_question(sender, phone_id_val)   # "Is there anything else I can help you with?"


# ─────────────────────────────────────────────
#  FOLLOW-UP QUESTION + SHOP FLOW
#  Stateful only for these short steps. Everything else stays free-form chat.
#  All text here is fixed UI text, so it is sent with save=False.
# ─────────────────────────────────────────────

SHOP_STEPS = {
    "shop_interest", "shop_browse", "shop_order_decision", "shop_product_name",
    "shop_more_categories", "shop_quantity", "shop_add_more", "shop_address",
}
# Steps where a long message is most likely a new medical question, not a shop answer
SHOP_ESCAPABLE = {"shop_interest", "shop_order_decision", "shop_more_categories", "shop_add_more"}

YES_RESPONSES = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo", "emukwayi"]
NO_RESPONSES = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi",
                "awe", "iyo", "pe", "that's all", "that is all", "no thanks", "no thank you"]

SHORT_REPLY_MAX_WORDS = 4


def _is_short(prompt):
    return len(prompt.split()) <= SHORT_REPLY_MAX_WORDS


def is_yes(prompt):
    p = prompt.lower().strip()
    return _is_short(p) and _contains_signal(p, YES_RESPONSES)


def is_no(prompt):
    p = prompt.lower().strip()
    return _is_short(p) and _contains_signal(p, NO_RESPONSES)


def looks_like_new_question(prompt):
    return "?" in prompt or len(prompt.split()) >= 5


def skip_language_update(step, prompt):
    """Short yes/no/number/address replies in these steps are not a language signal."""
    if step in SHOP_STEPS:
        return True
    return step == "follow_up" and len(prompt.split()) <= 2


def reset_flow(sender):
    """Back to plain chat. Clears any half-finished shop order."""
    state = user_states[sender]
    state["step"] = "chat"
    for k in ("cart", "shop_selected_product", "shop_selected_price", "shop_category"):
        state.pop(k, None)
    save_single_user_state(sender)


def _all_products():
    return [p for items in products_by_category.values() for p in items]


def _match_product(prompt_lower):
    if len(prompt_lower) < 3:
        return None
    return next(
        (p for p in _all_products()
         if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()),
        None,
    )


def _select_product(sender, phone_id_val, lang, product):
    state = user_states[sender]
    state["shop_selected_product"] = product["name"]
    state["shop_selected_price"] = product["price"]
    _ask_quantity(sender, phone_id_val, lang, product["name"])


# ── "Anything else?" ─────────────────────────

FOLLOWUP_MAP = {
    "shona": "Pane chimwe chandingakubatsira nacho here?",
    "ndebele": "Ingabe kukhona okunye engingakusiza ngakho?",
    "tonga": "Hena muli amubuyo umbi?",
    "chinyanja": "Kodi pali zina zomwe ndingakuthandizireni?",
    "bemba": "Kuli fintu fimbi ifyo ningamwafwilisha?",
    "lozi": "Ki sina sika ni ka thusa ka sona",
}

ASK_QUESTION_MAP = {
    "shona": "Bvunzai mubvunzo wenyu.",
    "ndebele": "Ngiyacela ubuze umbuzo wakho.",
    "tonga": "Amubuye mubuyo",
    "chinyanja": "Chonde funsani funso lanu.",
    "bemba": "Nomba, ipusha ilipusho lyobe.",
    "lozi": "Nkumbira ubuze mubvuzo wako.",
}


def ask_follow_up_question(sender, phone_id_val):
    state = user_states[sender]
    lang = state.get("language", "english")
    say(FOLLOWUP_MAP.get(lang, "Is there anything else I can help you with?"), sender, phone_id_val)
    state["step"] = "follow_up"
    save_single_user_state(sender)


def handle_follow_up_reply(sender, prompt, phone_id_val):
    """
    Returns True if the message was a yes/no answer to "anything else?" and is fully handled.
    Returns False if it is a new question: step goes back to chat and the caller answers it.
    """
    state = user_states[sender]
    lang = state.get("language", "english")

    if is_no(prompt):
        drop_last_user_message(sender)
        _ask_purchase_interest(sender, phone_id_val, lang)
        return True

    if is_yes(prompt):
        drop_last_user_message(sender)
        say(ASK_QUESTION_MAP.get(lang, "Please ask your question."), sender, phone_id_val)
        state["step"] = "chat"
        save_single_user_state(sender)
        return True

    state["step"] = "chat"
    save_single_user_state(sender)
    return False


# ── Purchase interest ────────────────────────

def _ask_purchase_interest(sender, phone_id_val, lang):
    state = user_states[sender]
    ask_map = {
        "shona": "Ungada here kutenga zvimwe zvezvigadzirwa zvedu? ",
        "ndebele": "Ungathanda ukuthenga noma yimuphi imikhiqizo yethu? ",
        "chinyanja": "Kodi mukufuna kugula zinthu zina mu zithu zathu? ",
        "tonga": "Mulakonzya kuyanda kuula zimwi zintu zyesu? ",
        "bemba": "Bushe kuti mwatemwa ukushita ifipe fyesu fimo? ",
        "lozi": "Kana u bata ku landa swakupila sa luna? ",
    }
    say(ask_map.get(lang, "Would you like to purchase any of our products? We have ultrasound, Birth Kits, HPV Test etc."),
        sender, phone_id_val)
    state["step"] = "shop_interest"
    save_single_user_state(sender)


def _interpret_shop_intent(prompt_lower):
    browse_signals = [
        "yes", "yeah", "yep", "please", "sure", "ok", "okay", "alright",
        "ehe", "hongu", "ndizvo", "inde", "yebo",
        "product", "products", "what do you have", "what have you got",
        "show me", "available", "categories", "add more", "something else",
        "buy", "purchase", "looking for",
        "zvigadzirwa", "zvinhu", "imikhiqizo", "zinthu", "imisansa", "swakupila",
    ]
    decline_signals = [
        "no", "nah", "nope", "not really", "not now", "that's all", "that is all",
        "done", "finish", "complete", "checkout", "later", "goodbye", "bye",
        "hapana", "kwete", "aiwa", "a'a", "ayi", "cha",
    ]
    if _contains_signal(prompt_lower, browse_signals):
        return "browse"
    if _contains_signal(prompt_lower, decline_signals):
        return "decline"
    return "unknown"


BYE_MAP = {
    "shona": "Zvakanaka! Iva nezuva rakanaka. Tanga patsva nekuti 'hesi' kana uine mimwe mibvunzo.",
    "ndebele": "Kulungile! Ube nosuku oluhle. Qala kabusha ngo-'unjani' uma uneminye imibuzo.",
    "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Yambani ndi 'muli bwanji' ngati muli ndi mafunso.",
    "tonga": "Kabotu! Mukale kabotu. Amutalike alimwi akulemba kuti 'mwabuka buti' kuti muli amibvunzo imbi.",
    "bemba": "Chisuma, Mwikale bwino. Lembeni 'shani' nga muli ne nefipusho nafimbi.",
    "lozi": "Ho lokile! Mube ni lizazi le linde. Qalisa ndi 'mwa bona' ha mu na lipuzo.",
}


def handle_shop_interest(sender, prompt, phone_id_val):
    lang = user_states[sender].get("language", "english")
    intent = _interpret_shop_intent(prompt.lower().strip())

    if intent == "decline":
        say(BYE_MAP.get(lang, "Alright! Have a nice day. Say 'hi' if you have more questions."), sender, phone_id_val)
        reset_flow(sender)
    else:
        _send_shop_categories(sender, phone_id_val, lang)


# ── Categories and products ──────────────────

def _send_shop_categories(sender, phone_id_val, lang):
    header_map = {
        "shona": "🛒 Makategi eZvigadzirwa:\n",
        "ndebele": "🛒 Imigqa Yemikhiqizo:\n",
        "chinyanja": "🛒 Mitundu ya Zinthu:\n",
        "tonga": "🛒 Misela ya Zintu:\n",
        "bemba": "🛒 Imisango ya fipe:\n",
        "lozi": "🛒 Mibeko ya Swakupila:\n",
    }
    lines = [header_map.get(lang, "🛒 Product Categories:\n")]

    for idx, (cat_name, items) in enumerate(products_by_category.items(), 1):
        lines.append(f"*{idx}. {cat_name}*")
        for item in items[:2]:
            lines.append(f"   • {item['name']} — {item['price']}")
        if len(items) > 2:
            more_map = {
                "shona": f"   ...uye zvimwe {len(items) - 2}",
                "ndebele": f"   ...namanye a-{len(items) - 2}",
                "chinyanja": f"   ...ndi ena {len(items) - 2}",
                "tonga": f"   ...azimwi {len(items) - 2}",
                "bemba": f"   ...na fimo {len(items) - 2}",
                "lozi": f"   ...ni ze ñwi {len(items) - 2}",
            }
            lines.append(more_map.get(lang, f"   ...and {len(items) - 2} more"))
        lines.append("")

    prompt_map = {
        "shona": "Tumira nhamba yekategi kuti uone zvigadzirwa zvose, kana udza zita rechigadzirwa chaunoda kutenga.",
        "ndebele": "Thumela inombolo yomugqa ukuze ubone wonke umkhiqizo, noma sitshele igama lomkhiqizo ofuna ukuthenga.",
        "chinyanja": "Tumizani nambala ya gulu kuti muone zinthu zonse, kapena uzani dzina la chinthu mukufuna kugula.",
        "tonga": "Tumizya nambala ya musela kuti mubone zintu zyoonse, naa mulembe zina lya cintu ncimuyanda kuuula.",
        "bemba": "Tuma nambala yafi putulwa pa kutila mumone ifintu fyonse ifilipo, nelyo bikeni ishina lya fintu mulefwaya ukushita.",
        "lozi": "Lumeza nomolo ya sibaka ku bona swakupila kaufela, kamba u bulele libizo la swakupila u bata ku landa.",
    }
    lines.append(prompt_map.get(
        lang, "Send the category number to see all products, or tell us the name of the product you'd like to order."))

    say("\n".join(lines), sender, phone_id_val)
    user_states[sender]["step"] = "shop_browse"
    save_single_user_state(sender)


def handle_shop_browse(sender, prompt, phone_id_val):
    state = user_states[sender]
    lang = state.get("language", "english")
    prompt_lower = prompt.lower().strip()
    categories = list(products_by_category.keys())

    if prompt_lower.isdigit():
        idx = int(prompt_lower) - 1
        if 0 <= idx < len(categories):
            cat_name = categories[idx]
            lines = [f"🏥 *{cat_name}*\n"]
            for i, item in enumerate(products_by_category[cat_name], 1):
                lines.append(f"{i}. {item['name']}")
                lines.append(f"   💰 {item['price']} | 📦 {item['availability']}")
                lines.append(f"   {item['description']}\n")

            order_map = {
                "shona": "\nUngada here kuodha chimwe chezvigadzirwa izvi? Pindura 'hongu' uye udza zita rechigadzirwa, kana 'aiwa'.",
                "ndebele": "\nUngathanda ukuodha noma yimuphi yale mikhiqizo? Phendula 'yebo' ubese usitshele igama lomkhiqizo, noma 'cha'.",
                "chinyanja": "\nKodi mukufuna kugula chinthu cha zinthu izi? Yankha 'inde' ndipo uzani dzina la chinthu, kapena 'ayi'.",
                "tonga": "\nSena muyanda kugula cimwi kuzintu eeci? Amupandule kuti 'mbubo' alimwi mulembesye zina lya cintu, naa kuti 'pe'.",
                "bemba": "\nBushe ulefwaya ukushita fima pali ifi? Yasuka 'ehe,' ulande neshina lyafyo ulefwaya ukushita nangu wasuke ukutila 'awe'.",
                "lozi": "\nKana u bata ku landa se si liñwi sa swakupila se? Arabela 'inde' u bulele libizo, kamba 'ayi'.",
            }
            lines.append(order_map.get(
                lang, "\nWould you like to order any of these products? Reply 'yes' and tell us the product name, or 'no'."))
            say("\n".join(lines), sender, phone_id_val)
            state["step"] = "shop_order_decision"
            state["shop_category"] = cat_name
            save_single_user_state(sender)
        else:
            invalid_map = {
                "shona": f"Nhamba isiriyo. Ndapota sarudza pakati pa 1 ne {len(categories)}.",
                "ndebele": f"Inombolo engavumelekile. Ngicela ukhethe phakathi kuka-1 no-{len(categories)}.",
                "chinyanja": f"Nambala yolakwika. Chonde sankhani pakati pa 1 ndi {len(categories)}.",
                "tonga": f"Nambala teyilungeme. Chabota, musale pakati pa 1 a {len(categories)}.",
                "bemba": f"Napapata, sala nambala pakati pa 1 na {len(categories)}.",
                "lozi": f"Nomolo e si ya. Ndapota, khetha ku zwana 1 ku ya ku {len(categories)}.",
            }
            say(invalid_map.get(lang, f"Invalid number. Please choose between 1 and {len(categories)}."),
                sender, phone_id_val)
        return

    matched = _match_product(prompt_lower)
    if matched:
        _select_product(sender, phone_id_val, lang, matched)
        return

    _send_shop_categories(sender, phone_id_val, lang)


def handle_shop_order_decision(sender, prompt, phone_id_val):
    state = user_states[sender]
    lang = state.get("language", "english")
    prompt_lower = prompt.lower().strip()

    if _contains_signal(prompt_lower, YES_RESPONSES):
        ask_which_map = {
            "shona": "Zvakanaka! Nyora zita rechigadzirwa chaunoda kuodha.",
            "ndebele": "Kulungile! Bhala igama lomkhiqizo ofuna ukuwodha.",
            "chinyanja": "Chabwino! Lemba dzina la chinthu chomwe mukufuna kugula.",
            "tonga": "Kabotu! Mulembesye zina lya cintu ncimuyanda kuula.",
            "bemba": "Chawama! nomba Lemba ishina lya fyo ulefwaya ukushita.",
            "lozi": "Ho lokile! Ñola libizo la swakupila u bata ku landa.",
        }
        say(ask_which_map.get(lang, "Great! Please type the name of the product you'd like to order."),
            sender, phone_id_val)
        state["step"] = "shop_product_name"
        save_single_user_state(sender)
        return

    if _contains_signal(prompt_lower, NO_RESPONSES):
        see_more_map = {
            "shona": "Zvakanaka! Ungada here kuona mamwe makategi? ",
            "ndebele": "Kulungile! Ungathanda ukubona eminye imigqa? ",
            "chinyanja": "Chabwino! Kodi mukufuna kuona mitundu ina? ",
            "tonga": "Kabotu! Mulakonzya kuyanda kubona misela imbi? ",
            "bemba": "Chisuma! Bushe ulefwaya ukumona ifiputulwa nangu ifipe fimbi? ",
            "lozi": "Ho lokile! Kana u bata ku bona mibeko ina? ",
        }
        say(see_more_map.get(lang, "Alright! Would you like to see other categories?"), sender, phone_id_val)
        state["step"] = "shop_more_categories"
        save_single_user_state(sender)
        return

    matched = _match_product(prompt_lower)
    if matched:
        _select_product(sender, phone_id_val, lang, matched)
        return

    ask_yes_no_map = {
        "shona": "Ndapota pindura 'hongu' kana 'aiwa'.",
        "ndebele": "Ngicela uphendule 'yebo' noma 'cha'.",
        "chinyanja": "Chonde yankha 'inde' kapena 'ayi'.",
        "tonga": "Chabota, mupandule kuti 'iya' naa 'pe'.",
        "bemba": "Napapata, yasuka 'ehe' kapena 'awe'.",
        "lozi": "Ndapota, arabela 'inde' kamba 'ayi'.",
    }
    say(ask_yes_no_map.get(lang, "Please reply 'yes' or 'no'."), sender, phone_id_val)


def handle_shop_product_name(sender, prompt, phone_id_val):
    lang = user_states[sender].get("language", "english")
    matched = _match_product(prompt.lower().strip())

    if matched:
        _select_product(sender, phone_id_val, lang, matched)
    else:
        not_found_map = {
            "shona": "Handina kuwana chigadzirwa ichocho. Ndapota nyora zita rakanangana rechigadzirwa kubva kumureza.",
            "ndebele": "Angitholi umkhiqizo lowo. Ngicela ubhale igama elicacile lomkhiqizo uvela ohlwini.",
            "chinyanja": "Sindipeza chinthu ichi. Chonde lemba dzina loyenera la chinthu kuchokera pamndandanda.",
            "tonga": "Tandajana cintu eeco. Ndapota, mulembesye zina lilungene lya cintu kuzwa pamudandaano.",
            "bemba": "Nshasangile ichiputulwa ichi, lemba ishina chiputulwa ififine chilemoneka muchilangililo.",
            "lozi": "Ha ni fumani swakupila se. Ndapota, ñola libizo le li nepahala la swakupila ku luhelo.",
        }
        say(not_found_map.get(lang, "I couldn't find that product. Please type the exact product name from the list."),
            sender, phone_id_val)


def handle_shop_more_categories(sender, prompt, phone_id_val):
    lang = user_states[sender].get("language", "english")
    intent = _interpret_shop_intent(prompt.lower().strip())

    if intent == "decline":
        say(BYE_MAP.get(lang, "Alright! Have a nice day. Say 'hi' if you have more questions."), sender, phone_id_val)
        reset_flow(sender)
    else:
        _send_shop_categories(sender, phone_id_val, lang)


# ── Quantity, cart, address, confirmation ────

def _ask_quantity(sender, phone_id_val, lang, product_name):
    qty_map = {
        "shona": f"Zvakanaka! Mungada mangani e *{product_name}*?",
        "ndebele": f"Kulungile! Ufuna izinga elingakanani le *{product_name}*?",
        "chinyanja": f"Chabwino! Mukufuna kuchulukitsa *{product_name}* kangati?",
        "tonga": f"Kabotu! manji buti ma *{product_name}* njo muyanda?",
        "bemba": f"Cisuma! niyanga ama *{product_name}* mulefwaya?",
        "lozi": f"Ho lokile! U bata ku landa *{product_name}* hañata mañi?",
    }
    say(qty_map.get(lang, f"Great! How many *{product_name}* would you like?"), sender, phone_id_val)
    user_states[sender]["step"] = "shop_quantity"
    save_single_user_state(sender)


def handle_shop_quantity(sender, prompt, phone_id_val):
    state = user_states[sender]
    lang = state.get("language", "english")

    qty_match = re.search(r"\d+", prompt)
    qty = int(qty_match.group()) if qty_match else 0

    if qty >= 1:
        product = state.get("shop_selected_product", "Unknown")
        state.setdefault("cart", []).append({
            "product": product,
            "price": state.get("shop_selected_price", "N/A"),
            "quantity": qty,
        })
        more_map = {
            "shona": f"✅ *{product}* x{qty} yawedzerwa! Ungada here kuwedzera chimwe chigadzirwa?",
            "ndebele": f"✅ *{product}* x{qty} yengezwe! Ungathanda ukwengeza umkhiqizo wolunye?",
            "chinyanja": f"✅ *{product}* x{qty} yaonjezedwa! Kodi mukufuna kuwonjezera chinthu china?",
            "tonga": f"✅ *{product}* x{qty} canjila! Sena mulayanda kunjizya cintu cimwi?",
            "bemba": f"✅ *{product}* x{qty} Yalundwapo! Bushe mulefwaya nafimbi?",
            "lozi": f"✅ *{product}* x{qty} i yemelizwe! Kana u bata ku yema swakupila si liñwi?",
        }
        say(more_map.get(lang, f"✅ *{product}* x{qty} added! Would you like to add anything else?"),
            sender, phone_id_val)
        state["step"] = "shop_add_more"
        save_single_user_state(sender)
    else:
        invalid_qty_map = {
            "shona": "Ndapota pinda nhamba (semuenzaniso: 1, 2, 3).",
            "ndebele": "Ngicela ufake inombolo (isibonelo: 1, 2, 3).",
            "chinyanja": "Chonde lowetsani nambala (mwachitsanzo: 1, 2, 3).",
            "tonga": "Ndakmba, mubesye nambala (mucikozyanyo: 1, 2, 3).",
            "bemba": "Napapata, ingisha inambala (ichilangililo: 1, 2, 3).",
            "lozi": "Ndapota, kenya nomolo (semuenzaniso: 1, 2, 3).",
        }
        say(invalid_qty_map.get(lang, "Please enter a number (e.g. 1, 2, 3)."), sender, phone_id_val)


def handle_shop_add_more(sender, prompt, phone_id_val):
    state = user_states[sender]
    lang = state.get("language", "english")
    prompt_lower = prompt.lower().strip()

    matched = _match_product(prompt_lower)
    if matched:
        _select_product(sender, phone_id_val, lang, matched)
        return

    if _interpret_shop_intent(prompt_lower) == "browse":
        _send_shop_categories(sender, phone_id_val, lang)
        return

    addr_map = {
        "shona": "Zvakanaka! Ndapota tipa kero yako yekuendesa (guta, nharaunda, uye mamwe mashoko akabatsira).",
        "ndebele": "Kulungile! Ngicela unike ikheli lakho lokuhambisa (idolobha, indawo, noma ulwazi olwengeziwe).",
        "chinyanja": "Chabwino! Chonde tipatseni adilesi yanu yokumanga (mzinda, dera, ndi chilichonse china chopindulitsa).",
        "tonga": "Kabotu! Ndapota, mutupe adilesi yanu yakuleeta zintu (tauni, ncobukala, alimwi azimwi zyo nga twabelesya kuzyba busena).",
        "bemba": "Chisuma! Napapata, mpeele adilesi yenu iya kuletako ifyomuleshita (tawuni, cifulo, kabili nafimbi ifyo twingeshibilako pa nchende).",
        "lozi": "Ho lokile! Ndapota, nipe aderesi ya hao ya ku alafa (tauni, sibaka, ni ze ñwi ze thusang).",
    }
    say(addr_map.get(lang, "Great! Please provide your delivery address (town, area, and any helpful details)."),
        sender, phone_id_val)
    state["step"] = "shop_address"
    save_single_user_state(sender)


def _save_orders_to_redis(sender, cart, address):
    """Name kept so callers do not change. Orders now go to the Supabase `orders` table."""
    user_id = user_states[sender].get("user_id", sender)
    for item in cart:
        db.add_order(sender, user_id, item["product"], item["price"], item["quantity"], address)
        logging.info(f"Order saved for {sender}: {item}")


def _send_order_confirmation(sender, phone_id_val, lang, cart, address):
    def build_lines(header, addr_label, closing):
        parts = [header, ""]
        for item in cart:
            parts.append(f"  📦 {item['product']} x{item['quantity']} — {item['price']}")
        parts += ["", f"📍 {addr_label}: {address}", closing]
        return "\n".join(parts)

    msg_map = {
        "shona": build_lines("✅ *Odha Yakugamuchirwa!*", "Kero", "Tichakubata munguva pfupi. Ndatenda! 😊"),
        "ndebele": build_lines("✅ *Ioda Ikugunyazwe!*", "Ikheli", "Sizokuthinta masinyane. Ngiyabonga! 😊"),
        "chinyanja": build_lines("✅ *Dongosolo Lasinthidwa!*", "Adilesi", "Tidzakuumbanani posachedwapa. Zikomo! 😊"),
        "tonga": build_lines("✅ *Oda Yatambulwa!*", "Adilesi", "Tula mwambila akaindi kaniini. Twatotela! 😊"),
        "bemba": build_lines("✅ *Oda yapokelelwa!*", "Aderesi", "Twalalanda naimwe mukashita akanono. Natotela! 😊"),
        "lozi": build_lines("✅ *Landa Le Li Amuhezwi!*", "Aderesi", "Lu ta ku ama ka nako ye nyinyani. Ndalumba! 😊"),
    }
    default = build_lines("✅ *Order Confirmed!*", "Delivery address", "We'll be in touch shortly. Thank you! 😊")
    say(msg_map.get(lang, default), sender, phone_id_val)


def handle_shop_address(sender, prompt, phone_id_val):
    state = user_states[sender]
    lang = state.get("language", "english")
    address = prompt.strip()
    cart = state.get("cart", [])

    if cart:
        _save_orders_to_redis(sender, cart, address)
        _send_order_confirmation(sender, phone_id_val, lang, cart, address)
    reset_flow(sender)


SHOP_HANDLERS = {
    "shop_interest": handle_shop_interest,
    "shop_browse": handle_shop_browse,
    "shop_order_decision": handle_shop_order_decision,
    "shop_product_name": handle_shop_product_name,
    "shop_more_categories": handle_shop_more_categories,
    "shop_quantity": handle_shop_quantity,
    "shop_add_more": handle_shop_add_more,
    "shop_address": handle_shop_address,
}


# ─────────────────────────────────────────────
#  MESSAGE ROUTER (no menus)
# ─────────────────────────────────────────────

CONNECTING_MAP = {
    "shona": (
        "🔍 Tiri kutsvaga mubatsiri wemunhu kuti akubatsirei...\n"
        "Ndapota mirira. Mubatsiri achabvumira kana aramba mukati memaminitsi rimwe.\n"
        "Nyora *CANCEL* kana uchida kudzosera kuna Rudo."
    ),
    "ndebele": (
        "🔍 Sifuna umuntu ozokusiza...\n"
        "Sicela ulinde. Umhloli uzowamukela noma ayenqabe ngemizuzwana engama-60.\n"
        "Bhala *CANCEL* uma ufuna ukubuyela ku-Rudo."
    ),
    "chinyanja": (
        "🔍 Tikufunafuna wothandiza wangwiro kwa inu...\n"
        "Chonde dikirini. Wothandiza adzayankha kapena akane mu mphindi imodzi.\n"
        "Lembani *CANCEL* mukafuna kubwera kwa Rudo."
    ),
    "bemba": (
        "🔍 Tukasanga umwafwilishi uwa bantu kwa imwe...\n"
        "Napapata dikileni. Umwafwilishi alayasuka naa ataike mu mamineti yimo.\n"
        "Lemba *CANCEL* nga mwafwaya ukubwela kuli Rudo."
    ),
    "lozi": (
        "🔍 Lu bata mubasi wa mutu ku thusa hao...\n"
        "Ndapota linda. Mubasi u ta amuhela kamba hana mwa miniti ye ñwi.\n"
        "Ñola *CANCEL* ha u bata ku bwela kwa Rudo."
    ),
    "tonga": (
        "🔍 Tulasanga wakugwasya wabantu kuti akugwasye...\n"
        "Ndapota linda. Wakugwasya ulamwaambila naa akane mukati aaminiti yomwe.\n"
        "Ñola *CANCEL* kuti mubweze kuli Rudo."
    ),
}

BACK_MAP = {
    "shona": "Kubatanidza kwake kwakugumira. Ndinokudzoseredzai kuna Rudo. Ndingakubatsirei?",
    "ndebele": "Uxhumano lwakho luphelile. Sibuyela ku-Rudo. Ngingakusiza?",
    "chinyanja": "Maukumano anu amaliza. Tikubweza kwa Rudo. Ndingakuthandizireni?",
    "bemba": "Ukumana kwenu kwashila. Mwabwela kuli Rudo. Kuti ningamwafwilisha?",
    "lozi": "Puisano ya hao i fela. Ubwela kwa Rudo. Nka ku thusa?",
    "tonga": "Kulumbaana kwanu kumanizya. Mubweza kuli Rudo. Nkugwasya buti?",
}

CANCEL_MAP = {
    "shona": "Tapota, chikumbiro chenyu chakanzurwa. Ndinokudzoseredzai kuna Rudo.",
    "ndebele": "Kulungile, isicelo senu sikhansiliwe. Sibuyela ku-Rudo.",
    "chinyanja": "Chabwino, pempho lanu lasanikizidwa. Tikubweza kwa Rudo.",
    "bemba": "Chisuma, icipemba chenu chatanshibwa. Mwabwela kuli Rudo.",
    "lozi": "Ho lokile, kopo ya hao i hamulelwa. Ubwela kwa Rudo.",
    "tonga": "Kabotu, bulombwi bwanu bwasinkiwa. Mubweza kuli Rudo.",
}


def handle_message(sender, prompt, phone_id_val):
    state = user_states[sender]
    step = state.get("step")

    # 1. Human agent sessions bypass the bot entirely
    if step == "human_agent_chat":
        if not relay_user_message_to_agent(sender, prompt, phone_id_val):
            state["step"] = "chat"
            state.pop("agent_phone", None)
            save_single_user_state(sender)
            lang = state.get("language", "english")
            send(BACK_MAP.get(lang, "Your agent session has ended. Returning you to Rudo. How can I help?"),
                 sender, phone_id_val, save=False)
        return

    if step == "waiting_for_agent":
        if prompt.strip().upper() == "CANCEL":
            if redis_client:
                try:
                    redis_client.delete(_agent_request_key(sender))
                    redis_client.delete(_agent_rejections_key(sender))
                except Exception:
                    pass
            state["step"] = "chat"
            save_single_user_state(sender)
            lang = state.get("language", "english")
            send(CANCEL_MAP.get(lang, "Your request has been cancelled. Returning you to Rudo."),
                 sender, phone_id_val, save=False)
        else:
            check_agent_request_timeout(sender, phone_id_val)
        return

    # 2. Onboarding on the first message, ordinary language tracking afterwards
    if step == "language_detection":
        state["language"] = detect_language(prompt, sender)
        user_id = auto_register(sender)
        send_welcome_with_id(sender, phone_id_val, state["language"], user_id)
        save_single_user_state(sender)
        if is_greeting_only(prompt) or is_referral_greeting(prompt):
            return
    elif not skip_language_update(step, prompt):
        maybe_update_language(sender, prompt)

    lang = state.get("language", "english")
    step = state.get("step")

    # 2b. A delivery address is free text, so it skips the keyword checks below
    if step == "shop_address":
        handle_shop_address(sender, prompt, phone_id_val)
        return

    # 3. Ask for a human
    if is_human_agent_request(prompt):
        send(CONNECTING_MAP.get(lang, "🔍 Looking for a human agent to assist you...\nPlease wait.\n"),
             sender, phone_id_val, save=False)
        state["step"] = "waiting_for_agent"
        save_single_user_state(sender)
        notify_agents_of_request(sender, phone_id_val)
        return

    # 4. Reset conversation context
    if is_reset_request(prompt):
        clear_conversation(sender)
        reset_flow(sender)
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_val, save=False)
        return

    # 5. Plain greeting, no model call needed
    if is_greeting_only(prompt) or is_referral_greeting(prompt):
        reset_flow(sender)
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_val, save=False)
        return

    # 6. Answer to "anything else?" (yes / no / a new question)
    if step == "follow_up":
        if handle_follow_up_reply(sender, prompt, phone_id_val):
            return
        # otherwise it is a new question: fall through to MedGemma

    # 7. Shop flow
    elif step in SHOP_HANDLERS:
        if step in SHOP_ESCAPABLE and looks_like_new_question(prompt):
            # A real question in the middle of shopping: drop the order flow and answer it.
            reset_flow(sender)
            save_user_conversation(sender, "user", prompt)   # shop steps do not store messages
        else:
            SHOP_HANDLERS[step](sender, prompt, phone_id_val)
            return

    # 8. Everything else is a free-form question
    answer_user(sender, prompt, phone_id_val)


# ─────────────────────────────────────────────
#  WEBHOOK
# ─────────────────────────────────────────────

def _already_processed(message_id) -> bool:
    """
    WhatsApp redelivers a webhook if we do not answer with 200 quickly, and a pivoted reply
    takes several seconds. Remember message ids so a redelivery does not get a second reply.
    """
    if not redis_client or not message_id:
        return False
    try:
        was_set = redis_client.set(f"wa_msg:{message_id}", "1", ex=3600, nx=True)
        return not bool(was_set)
    except Exception as e:
        logging.error(f"Error checking message id: {e}")
        return False


@app.route("/agent-timeout", methods=["POST"])
def agent_timeout():
    """
    Called by a scheduled job (for example Upstash QStash) after AGENT_TIMEOUT_SECONDS.
    Body JSON: { "user_number": "+260...", "phone_id": "..." }
    """
    try:
        body = request.get_json(force=True) or {}
        user_number = body.get("user_number")
        current_phone_id = body.get("phone_id")

        if not user_number or not current_phone_id:
            return jsonify({"status": "error", "message": "Missing user_number or phone_id"}), 400

        if redis_client:
            raw = redis_client.get(_agent_request_key(user_number))
            if raw:
                request_data = json.loads(raw) if isinstance(raw, str) else raw
                if request_data.get("status") == "pending":
                    _handle_no_agents_available(user_number, current_phone_id)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logging.error(f"Error in agent_timeout endpoint: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return render_template("connected.html")


def _process_message(message, value):
    sender = message["from"]
    incoming_phone_id = value["metadata"]["phone_number_id"]

    if _already_processed(message.get("id")):
        logging.info(f"Duplicate delivery ignored: {message.get('id')}")
        return

    # Agent button taps
    if message.get("type") == "interactive":
        interactive = message.get("interactive", {})
        if interactive.get("type") == "button_reply":
            btn_id = interactive["button_reply"]["id"]
            logging.info(f"Button reply from {sender}: {btn_id}")
            if btn_id.startswith("agent_accept:"):
                handle_agent_accept(sender, btn_id.split("agent_accept:", 1)[1], incoming_phone_id)
            elif btn_id.startswith("agent_reject:"):
                handle_agent_reject(sender, btn_id.split("agent_reject:", 1)[1], incoming_phone_id)
        return

    if "text" not in message:
        logging.info(f"Non-text message received from {sender}")
        ensure_user_state(sender, refresh=True)
        lang = user_states.get(sender, {}).get("language", "english")
        send(NON_TEXT_MAP.get(lang, "Sorry, I can only process text messages. Please send a text message."),
             sender, incoming_phone_id, save=False)
        return

    prompt = message["text"]["body"]
    logging.info(f"Processing message from {sender}: {prompt}")

    # Messages from an agent in an active session go to the user
    if normalize_phone(sender) in AGENT_PHONES_NORM:
        if relay_agent_message_to_user(sender, prompt, incoming_phone_id):
            return

    is_new = ensure_user_state(sender, refresh=True)
    if not is_new:
        user_states[sender]["first_message"] = False

    # Shop replies (quantities, addresses, product names) must not enter the history,
    # because the history is what MedGemma sees.
    if user_states[sender].get("step") not in SHOP_STEPS:
        save_user_conversation(sender, "user", prompt)

    referral = extract_referral_source(prompt)
    if referral:
        save_referral_source(sender, referral)
        logging.info(f"Referral detected for {sender}: {referral}")

    handle_message(sender, prompt, incoming_phone_id)


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == verify_token:
            return challenge, 200
        return "Failed", 403

    data = request.get_json(silent=True) or {}
    logging.info(f"Received webhook data: {data}")

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            if "messages" in value:
                for message in value["messages"]:
                    try:
                        _process_message(message, value)
                    except Exception as e:
                        # Always answer 200 so WhatsApp does not redeliver and double-reply
                        logging.error(f"Error processing message: {e}", exc_info=True)
            elif "statuses" in value:
                logging.info("Message status update received, ignoring.")
            else:
                logging.info("Webhook received non-message event, ignoring.")

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
