"""
Rudo WhatsApp bot (Dawa Health)

Flow for every text message:
    1. Agent relay checks (human agent sessions bypass the bot completely)
    2. Language detection (Gemini classifier, keyword scoring as offline fallback)
    3. For Shona, Ndebele, Chinyanja, Bemba, Tonga and Lozi:
           TranslateGemma (user language -> English)
           MedGemma answers in English
           guardrails (compose_reply) run on the result
           TranslateGemma (English -> user language)
       English goes straight to MedGemma.
    4. Reply is sent. There are no menus, every message is free-form chat.

Gemini is used ONLY for language detection. Replies never come from Gemini.

Env vars:
    WA_TOKEN, PHONE_ID, OWNER_PHONE, VERIFY_TOKEN (default "BOT")
    UPSTASH_REDIS_URL, UPSTASH_REDIS_TOKEN
    GEN_API                       Gemini key (language detection only)
    MEDGEMMA_URL                  read by medgemma_client.py
    TRANSLATEGEMMA_URL            read by translategemma_client.py
    TRANSLATE_LANGS               optional, languages to pivot through English
"""

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
from upstash_redis import Redis

from medgemma_client import ask_medgemma, medgemma_enabled, pairs_from_history
from medgemma_guardrails import clean_model_output, compose_reply
from translategemma_client import needs_translation, translate, translategemma_enabled

logging.basicConfig(level=logging.INFO)

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

# Upstash Redis
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        redis_client.ping()
        logging.info("Successfully connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Failed to connect to Upstash Redis: {e}")
        redis_client = None
else:
    redis_client = None
    logging.warning("UPSTASH_REDIS_URL or UPSTASH_REDIS_TOKEN not set, Redis functionality disabled")

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
    if redis_client and sender in user_states:
        try:
            redis_client.set(f"user_state:{sender}", json.dumps(user_states[sender]))
        except Exception as e:
            logging.error(f"Error saving state for {sender}: {e}")


def load_user_state(sender):
    if redis_client:
        try:
            state_data = redis_client.get(f"user_state:{sender}")
            if state_data:
                if isinstance(state_data, (dict, list)):
                    return state_data
                return json.loads(state_data)
        except Exception as e:
            logging.error(f"Error loading user state for {sender}: {e}")
    return None


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
    if redis_client:
        try:
            history = redis_client.get(f"conversation:{sender}")
            if not history:
                return []
            if isinstance(history, list):
                return history
            if isinstance(history, str):
                parsed = json.loads(history)
                if isinstance(parsed, list):
                    return parsed
            return []
        except Exception as e:
            logging.error(f"Error getting conversation: {e}")
            return []
    return []


def save_user_conversation(sender, role, message, message_en=None):
    """message_en is the English version of the message, stored for pivoted languages."""
    if redis_client:
        try:
            conversation = get_user_conversation(sender)
            if not isinstance(conversation, list):
                conversation = []
            entry = {
                "role": role,
                "message": str(message),
                "timestamp": datetime.now().isoformat(),
            }
            if message_en:
                entry["message_en"] = str(message_en)
            conversation.append(entry)
            if len(conversation) > 100:
                conversation = conversation[-100:]
            redis_client.set(f"conversation:{sender}", json.dumps(conversation))
        except Exception as e:
            logging.error(f"Error saving conversation: {e}")


def set_last_message_en(sender, role, text_en):
    """Attach an English translation to the most recent stored message of `role`."""
    if not redis_client or not text_en:
        return
    try:
        conversation = get_user_conversation(sender)
        for entry in reversed(conversation):
            if entry.get("role") == role:
                entry["message_en"] = str(text_en)
                break
        redis_client.set(f"conversation:{sender}", json.dumps(conversation))
    except Exception as e:
        logging.error(f"Error attaching English text to conversation: {e}")


def clear_conversation(sender):
    if redis_client:
        try:
            redis_client.delete(f"conversation:{sender}")
        except Exception as e:
            logging.error(f"Error clearing conversation: {e}")


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
    save=False for messages that should not enter the conversation history.
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
        send("Sorry, the agent system is unavailable right now.", user_number, current_phone_id)
        return

    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            send("⚠️ This chat request has already expired or been accepted by another agent.",
                 agent_phone, current_phone_id)
            return

        request_data = json.loads(raw) if isinstance(raw, str) else raw

        if request_data.get("status") != "pending":
            send("⚠️ This chat has already been accepted by another agent.", agent_phone, current_phone_id)
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
            agent_phone, current_phone_id
        )
        send(
            f"✅ Great news! {agent_name} has accepted your chat request.\n"
            f"You are now connected!",
            user_number, current_phone_id
        )

        for other_name, other_phone in AGENTS.items():
            if other_phone != agent_phone:
                send(
                    f"ℹ️ The chat request from user {user_number} has been accepted by {agent_name}.",
                    other_phone, current_phone_id
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
            send("⚠️ This chat request has already expired.", agent_phone, current_phone_id)
            return

        request_data = json.loads(raw) if isinstance(raw, str) else raw
        if request_data.get("status") != "pending":
            send("ℹ️ This request was already handled.", agent_phone, current_phone_id)
            return

        rejections_raw = redis_client.get(_agent_rejections_key(user_number))
        if isinstance(rejections_raw, str):
            rejections = json.loads(rejections_raw)
        else:
            rejections = rejections_raw or []
        if agent_phone not in rejections:
            rejections.append(agent_phone)
        redis_client.set(_agent_rejections_key(user_number), json.dumps(rejections), ex=AGENT_TIMEOUT_SECONDS + 10)

        send("👍 You have rejected this chat request.", agent_phone, current_phone_id)

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
        user_number, current_phone_id
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
        send(f"💬 *User ({sender}):* {prompt}", agent_phone, current_phone_id)
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

            send("✅ You have ended the chat session.", agent_phone, current_phone_id)
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
                user_number, current_phone_id
            )
            return True

        send(f"💬 *Agent:* {prompt_stripped}", user_number, current_phone_id)
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
    if not redis_client:
        logging.warning("Redis not available, referral not saved.")
        return
    try:
        entry = {
            "sender": sender,
            "source": source,
            "user_id": user_states.get(sender, {}).get("user_id", "unregistered"),
            "timestamp": datetime.now().isoformat(),
        }
        key = f"referrals:{sender}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        redis_client.set(key, json.dumps(entry))
        logging.info(f"Referral saved {key}: {entry}")
    except Exception as e:
        logging.error(f"Error saving referral: {e}")


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
    if lang == "english" or not text:
        return False
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 6:
        return False
    hits = sum(1 for w in words if w in ENGLISH_TELLS)
    return (hits / len(words)) > 0.15


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


def send_welcome_with_id(sender, phone_id_val, lang, user_id):
    welcome_map = {
        "shona": f"Mhoro! Ndinonzi Rudo, mubatsiri wepamhepo weDawa Health. ID yenyu ndeye: {user_id}. Chengetedza ID iyi nekuti ichakumbirwa kumaDawa clinics. Ndingakubatsirei nhasi?",
        "ndebele": f"Sawubona! Ngingu Rudo, isiphathamandla se-Dawa Health. I-ID yakho ithi: {user_id}. Gcina le ID ngoba izocelwa kumaDawa clinics. Ngingakusiza ngani namuhla?",
        "bemba": f"Mwaiseni! Nine Rudo, wakufwailisha wa Dawa Health. ID yenu ni: {user_id}. Sungeni ID iyi pantu ikabombwa ku Dawa clinics. Nga kuti namwafwa shani lelo?",
        "chinyanja": f"Moni! Ndine Rudo, mphungu wa Dawa Health. ID yanu ndi: {user_id}. Sungani ID iyi chifukwa idzafunsidwa kumakliniki a Dawa. Ndingakuthandizireni lero?",
        "tonga": f"Muli buti! Ndime Rudo, wakugwasya Dawa Health. ID yenu nji: {user_id}. Mweelede kuisunga kabotu ID kambo iyakubeleka ku Dawa clinics. Nga ndamukyasya buti lino?",
        "lozi": f"Mwa bona! Mina ki Rudo, mubasi wa ku thusa wa Dawa Health. ID ya hao ki: {user_id}. Boloka ID ye hantši kakuli u ta buzwa yona kwa makiliniki a Dawa. Nka ku thusa ka mini sunu?",
    }
    send(
        welcome_map.get(
            lang,
            f"Hello! I'm Rudo, Dawa Health's virtual assistant. Your ID is: {user_id}. "
            f"Keep this ID safe because it'll be asked for at the Dawa clinics. How can I help you today?"
        ),
        sender, phone_id_val
    )


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


def _history_for_model(sender):
    """
    Stored history shaped for pairs_from_history, preferring the English version of each turn.
    The current user message is already stored by the webhook, so it is dropped here and
    passed to ask_medgemma separately. Adjust if your ask_medgemma expects it in history.
    """
    history = get_user_conversation(sender)
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    history = history[-HISTORY_ENTRIES_FOR_MODEL:]
    out = []
    for entry in history:
        text = entry.get("message_en") or entry.get("message", "")
        out.append({**entry, "message": text})
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
        history = pairs_from_history(_history_for_model(sender))
        reply_en = clean_model_output(ask_medgemma(model_question, history=history))

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
    if not reply:
        reply = failure_message(lang)
        reply_en = None

    send(reply, sender, phone_id_val, message_en=reply_en)


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
                 sender, phone_id_val)
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
                 sender, phone_id_val)
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
    else:
        maybe_update_language(sender, prompt)

    lang = state.get("language", "english")

    # 3. Ask for a human
    if is_human_agent_request(prompt):
        send(CONNECTING_MAP.get(lang, "🔍 Looking for a human agent to assist you...\nPlease wait.\n"),
             sender, phone_id_val)
        state["step"] = "waiting_for_agent"
        save_single_user_state(sender)
        notify_agents_of_request(sender, phone_id_val)
        return

    # 4. Reset conversation context
    if is_reset_request(prompt):
        clear_conversation(sender)
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_val)
        return

    # 5. Plain greeting, no model call needed
    if is_greeting_only(prompt) or is_referral_greeting(prompt):
        send(GREET_MAP.get(lang, "Hello! How can I help you today?"), sender, phone_id_val)
        return

    # 6. Everything else is a free-form question
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
             sender, incoming_phone_id)
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
