import google.generativeai as genai
from flask import Flask, request, jsonify, render_template 
import requests    
import os  
import fitz 
import sched   
import time     
import logging    
from mimetypes import guess_type 
from datetime import datetime, timedelta 
from urlextract import URLExtract
from training import instructions, product_images
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.api_core.exceptions import ResourceExhausted
from training import products, instructions, pregnancy_data, pregnancy_data_shona, pregnancy_data_ndebele, pregnancy_data_tonga, pregnancy_data_chinyanja, pregnancy_data_bemba, pregnancy_data_lozi, cervical_cancer_data, cervical_cancer_data_chinyanja, cervical_cancer_data_lozi

from products_data import products_by_category
from upstash_redis import Redis
from medgemma_client import ask_medgemma, medgemma_enabled, pairs_from_history
from medgemma_guardrails import clean_model_output, compose_reply
import json
import re
import random
import string

logging.basicConfig(level=logging.INFO)

# Initialize Upstash Redis connection
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

# Global in-memory cache (per worker)
user_states = {}

wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.5-flash"
genai.configure(api_key=gen_api)
name = "Fae"
bot_name = "Rudo"

# ── AGENT DICTIONARY ─────────────────────────────────────────────────────────
AGENTS = {
    "Agent 1": "+260978760105",
}
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TIMEOUT_SECONDS = 60   # seconds before "no agents available" fires

app = Flask(__name__)
genai.configure(api_key=gen_api)

class CustomURLExtract(URLExtract):
    def _get_cache_file_path(self):
        cache_dir = "/tmp"
        return os.path.join(cache_dir, "tlds-alpha-by-domain.txt")

extractor = CustomURLExtract(limit=1)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


# ─────────────────────────────────────────────
#  SHARED MATCHING HELPER
# ─────────────────────────────────────────────

def _contains_signal(prompt_lower: str, phrases: list) -> bool:
    """
    True if any phrase in `phrases` is present in prompt_lower.
    - Single-word phrases are matched on WORD BOUNDARIES so they don't
      false-positive inside unrelated longer words (e.g. 'no' inside
      'know', 'cha' inside 'purchase', 'ok' inside 'broke').
    - Multi-word phrases are matched as plain substrings, since a
      multi-word phrase is specific enough not to appear by accident.
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


# ─────────────────────────────────────────────
#  PER-USER REDIS STATE
# ─────────────────────────────────────────────

def save_single_user_state(sender):
    if redis_client and sender in user_states:
        try:
            redis_client.set(f"user_state:{sender}", json.dumps(user_states[sender]))
            logging.debug(f"Saved state for {sender}")
        except Exception as e:
            logging.error(f"Error saving state for {sender}: {e}")

def load_user_state(sender):
    if redis_client:
        try:
            state_data = redis_client.get(f"user_state:{sender}")
            if state_data:
                return json.loads(state_data)
        except Exception as e:
            logging.error(f"Error loading user state for {sender}: {e}")
    return None

def save_user_states():
    for sender in list(user_states.keys()):
        save_single_user_state(sender)

def load_user_states():
    global user_states
    user_states = {}
    logging.info("User states initialised (lazy per-user loading enabled)")


# ─────────────────────────────────────────────
#  HELPER: ensure a sender is in user_states
# ─────────────────────────────────────────────

def ensure_user_state(sender):
    if sender in user_states:
        return False

    saved = load_user_state(sender)
    if saved:
        user_states[sender] = saved
        return False

    user_states[sender] = {
        "step": "language_detection",
        "language": "english",
        "registered": False,
        "phone_digits": None,
        "user_id": None,
        "topic": None,
        "needs_language_confirmation": False,
        "first_message": True
    }
    return True


def reset_conversation(sender):
    user_states[sender] = {
        "step": "main_menu",
        "language": user_states[sender].get("language", "english"),
        "registered": True,
        "phone_digits": user_states[sender].get("phone_digits"),
        "user_id": user_states[sender].get("user_id"),
        "topic": None,
        "needs_language_confirmation": False,
        "first_message": False
    }
    save_single_user_state(sender)


# ─────────────────────────────────────────────
#  HUMAN AGENT SYSTEM
# ─────────────────────────────────────────────

HUMAN_AGENT_TRIGGERS = [
    "human agent", "speak to agent", "talk to agent", "talk to a person", "human",
    "speak to a person", "real person", "human support", "live agent", "support",
    "live support", "connect me to an agent", "i need help from a person",
    "speak to someone", "talk to someone", "customer service", "agent",
]

def is_human_agent_request(prompt: str) -> bool:
    """Return True if the user is asking for a human agent."""
    p = prompt.lower().strip()
    return _contains_signal(p, HUMAN_AGENT_TRIGGERS)


def normalize_phone(phone: str) -> str:
    """Strip leading '+' so phone numbers are consistent regardless of source."""
    return phone.lstrip("+")


def send_interactive_buttons(phone_number: str, body_text: str, buttons: list, phone_id_val: str):
    """
    Send a WhatsApp interactive button message.
    buttons: list of dicts with keys 'id' and 'title' (max 3 buttons, title max 20 chars).
    """
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
    resp = requests.post(url, headers=headers, json=data)
    logging.info(f"Interactive button send to {phone_number}: {resp.status_code} {resp.text}")
    return resp


def _agent_request_key(user_number: str) -> str:
    return f"agent_request:{user_number}"


def _agent_session_key(user_number: str) -> str:
    return f"agent_session:{user_number}"


def _agent_rejections_key(user_number: str) -> str:
    return f"agent_rejections:{user_number}"


def notify_agents_of_request(sender: str, current_phone_id: str):
    """
    Broadcast a chat request to all agents with Accept / Reject buttons.
    Stores request metadata in Redis with a TTL of AGENT_TIMEOUT_SECONDS.
    """
    state = user_states.get(sender, {})
    user_id = state.get("user_id", sender)
    lang = state.get("language", "english")

    request_data = {
        "user_number": sender,
        "user_id": user_id,
        "language": lang,
        "phone_id": current_phone_id,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",          # pending | accepted | expired
        "accepted_by": None,
        "rejections": [],
    }

    if redis_client:
        try:
            redis_client.set(
                _agent_request_key(sender),
                json.dumps(request_data),
                ex=AGENT_TIMEOUT_SECONDS + 10,   # slight buffer
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


def handle_agent_accept(agent_phone: str, user_number: str, current_phone_id: str):
    """Called when an agent taps Accept."""
    if not redis_client:
        send("Sorry, the agent system is unavailable right now.", user_number, current_phone_id)
        return

    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            send_interactive_buttons(
                agent_phone,
                "⚠️ This chat request has already expired or been accepted by another agent.",
                [], current_phone_id
            )
            send("⚠️ This chat request has already expired or been accepted by another agent.", agent_phone, current_phone_id)
            return

        request_data = json.loads(raw)

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

        ensure_user_state(user_number)
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


def handle_agent_reject(agent_phone: str, user_number: str, current_phone_id: str):
    """Called when an agent taps Reject."""
    if not redis_client:
        return

    try:
        raw = redis_client.get(_agent_request_key(user_number))
        if not raw:
            send("⚠️ This chat request has already expired.", agent_phone, current_phone_id)
            return

        request_data = json.loads(raw)
        if request_data.get("status") != "pending":
            send("ℹ️ This request was already handled.", agent_phone, current_phone_id)
            return

        rejections_raw = redis_client.get(_agent_rejections_key(user_number))
        rejections = json.loads(rejections_raw) if rejections_raw else []
        if agent_phone not in rejections:
            rejections.append(agent_phone)
        redis_client.set(_agent_rejections_key(user_number), json.dumps(rejections), ex=AGENT_TIMEOUT_SECONDS + 10)

        send("👍 You have rejected this chat request.", agent_phone, current_phone_id)

        if set(rejections) >= set(AGENTS.values()):
            _handle_no_agents_available(user_number, current_phone_id)

    except Exception as e:
        logging.error(f"Error in handle_agent_reject: {e}", exc_info=True)


def _handle_no_agents_available(user_number: str, current_phone_id: str):
    """Notify the user that no agents are available and return to bot."""
    ensure_user_state(user_number)
    user_states[user_number]["step"] = "main_menu"
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


def relay_user_message_to_agent(sender: str, prompt: str, current_phone_id: str) -> bool:
    """
    If the user is in an active agent session, forward their message to the agent.
    Returns True if message was relayed, False otherwise.
    """
    if not redis_client:
        return False

    try:
        raw = redis_client.get(_agent_session_key(sender))
        if not raw:
            return False
        session = json.loads(raw)
        agent_phone = session.get("agent_phone")
        if not agent_phone:
            return False

        send(f"💬 *User ({sender}):* {prompt}", agent_phone, current_phone_id)
        return True
    except Exception as e:
        logging.error(f"Error relaying user message to agent: {e}")
        return False


def relay_agent_message_to_user(agent_phone: str, prompt: str, current_phone_id: str) -> bool:
    """
    If this sender is an agent with an active session, forward their message to the user.
    Returns True if message was relayed (i.e., caller should NOT do normal bot processing).
    """
    if not redis_client:
        return False

    try:
        norm_agent = normalize_phone(agent_phone)
        raw = redis_client.get(f"agent_user_session:{norm_agent}")
        if not raw:
            return False
        session = json.loads(raw)
        user_number = session.get("user_number")
        if not user_number:
            return False

        prompt_stripped = prompt.strip()

        if prompt_stripped.upper() == "END CHAT":
            redis_client.delete(f"agent_user_session:{norm_agent}")
            redis_client.delete(_agent_session_key(user_number))
            redis_client.delete(_agent_request_key(user_number))

            ensure_user_state(user_number)
            user_states[user_number]["step"] = "main_menu"
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


def check_agent_request_timeout(user_number: str, current_phone_id: str):
    """
    Check if the agent request has timed out (no Redis TTL means it expired).
    Call this when the user sends a message while step == 'waiting_for_agent'.
    """
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

def extract_referral_source(prompt: str) -> str | None:
    """
    Returns the referral string (e.g. 'Mandebvu Pharmacy Poster 1')
    if the message contains a referral signal, otherwise None.
    """
    prompt_lower = prompt.lower().strip()

    triggered = any(t in prompt_lower for t in REFERRAL_TRIGGERS)
    if not triggered:
        return None

    for prep in ["from ", "from the ", "from a "]:
        idx = prompt_lower.rfind(prep)
        if idx != -1:
            source = prompt[idx + len(prep):].strip()
            source = source.rstrip(".,!?;:")
            if source:
                return source

    for trigger in sorted(REFERRAL_TRIGGERS, key=len, reverse=True):
        idx = prompt_lower.find(trigger)
        if idx != -1:
            source = prompt[idx + len(trigger):].strip().rstrip(".,!?;:")
            if source:
                return source

    return None


def save_referral_source(sender: str, source: str):
    """Persist a referral entry to Redis under referrals:<sender>:<timestamp>."""
    if not redis_client:
        logging.warning("Redis not available — referral not saved.")
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
        logging.info(f"Referral saved → {key}: {entry}")
    except Exception as e:
        logging.error(f"Error saving referral: {e}")
        


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

def save_user_conversation(sender, role, message):
    if redis_client:
        try:
            conversation = get_user_conversation(sender)
            if not isinstance(conversation, list):
                conversation = []
            conversation.append({
                "role": role,
                "message": str(message),
                "timestamp": datetime.now().isoformat()
            })
            if len(conversation) > 100:
                conversation = conversation[-100:]
            redis_client.set(f"conversation:{sender}", json.dumps(conversation))
            logging.debug(f"Saved conversation for {sender}")
        except Exception as e:
            logging.error(f"Error saving conversation: {e}")


# ─────────────────────────────────────────────
#  LLM FALLBACK LANGUAGE CLASSIFIER
#  Used only when local keyword/phrase scoring
#  finds zero signal. Prevents permanent
#  mis-defaulting to English for languages
#  whose word lists have gaps.
# ─────────────────────────────────────────────

def _llm_detect_language(message: str):
    supported = ["english", "shona", "ndebele", "chinyanja", "bemba", "tonga", "lozi"]
    try:
        classifier_prompt = (
            "Identify which ONE of these languages the following WhatsApp message "
            "is written in: english, shona, ndebele, chinyanja, bemba, tonga, lozi. "
            "These are languages spoken in Zimbabwe and Zambia. The message may mix "
            "in a few English loanwords (e.g. medical terms like 'cervical cancer' "
            "or 'HPV') while still being primarily one of the other languages — in "
            "that case, classify by the surrounding grammar/vocabulary, not the "
            "loanwords. Reply with ONLY the single lowercase language name and "
            "nothing else — no punctuation, no explanation.\n\n"
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
        if candidate.finish_reason != 1:  # 1 == STOP (normal completion)
            logging.warning(
                f"[_llm_detect_language] Non-STOP finish_reason={candidate.finish_reason} "
                f"safety_ratings={getattr(candidate, 'safety_ratings', None)}"
            )
            return None

        guess = re.sub(r"[^a-z]", "", response.text.strip().lower())
        if guess in supported:
            logging.info(f"[_llm_detect_language] Classified as: {guess}")
            return guess
        logging.warning(f"[_llm_detect_language] Unrecognised output: {guess!r}")
    except Exception as e:
        logging.error(f"[_llm_detect_language Error] {type(e).__name__}: {e}")
    return None


def detect_language(message, sender=None):
    message_lower = message.lower().strip()

    # Pure digit entries (menu choices, week numbers, etc.) are never a
    # language signal — keep whatever language is already set.
    if message_lower.isdigit():
        if sender and sender in user_states:
            return user_states[sender].get("language", "english")
        return "english"

    current_lang = "english"
    if sender and sender in user_states:
        current_lang = user_states[sender].get("language", "english")

    # ── Fast path: exact single-word greeting matches ───────────────────
    # Cheap and unambiguous — no need to hit the LLM for these.
    exact_matches = {
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
    
    for lang, words in exact_matches.items():
        if message_lower in words:
            logging.info(f"Exact match: {message_lower} -> {lang}")
            return lang

    # ── Primary: Gemini classification, run on every real message ───────
    # No word-count gate — even short local-language phrases ("sei",
    # "chii") are worth a real classification rather than falling
    # through to keyword scoring.
    llm_guess = _llm_detect_language(message)
    if llm_guess:
        if llm_guess != current_lang:
            logging.info(f"[LLM] Language switch for {sender}: {current_lang} -> {llm_guess}")
        return llm_guess

    logging.warning(
        f"[detect_language] Gemini classification failed/empty for {sender!r} "
        f"— falling back to keyword scoring"
    )

    # ── Offline fallback: keyword/phrase scoring (only reached if the ──
    # ── Gemini call raised or returned nothing usable)                ──
    language_keywords = {
    "shona": [
        # Greetings / expressions
        "mhoro", "mhoroi", "makadini", "maswera sei",
        "mamuka sei", "hesi", "ko sei",
        "ndatenda", "ndinotenda", "maita basa", "maita henyu",
        "ndapota", "ndibatsirei", "batsirai",

        # Distinctive Shona expressions / constructions
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

        # Health / maternal
        "pamuviri", "nhumbu", "nhumbu yangu",
        "zvigadzirwa", "chirwere", "gomarara",
        "chibereko", "zviratidzo", "chiremba",
        "kusvotwa", "kurwadziwa",
        "kubuda ropa", "mushonga", "mishonga",
        "chipatara", "kiriniki", "utano",
        "kubereka", "kusununguka",
    ],

    "ndebele": [
        # Greetings / expressions
        "sawubona", "salibonani", "unjani", "linjani",
        "livuke njani", "kusile njani",
        "ngiyabonga", "siyabonga",
        "ngiyacela", "ngicela",
        "ngisize", "ngidinga",

        # Distinctive Ndebele expressions / constructions
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

        # Health / maternal
        "ukukhulelwa", "isisu", "umntwana",
        "imikhiqizo", "umhlaza", "isibeletho",
        "izimpawu", "udokotela", "igazi",
        "ubuhlungu", "umuthi", "imithi",
        "isibhedlela", "umtholampilo",
        "impilo", "ukubeletha", "ukuzala",
    ],

    "chinyanja": [
        # Greetings / expressions
        "moni", "muli bwanji", "zikomo",
        "zikomo kwambiri", "pepani",
        "chonde", "ndapota",
        "nankani", "thandizani",
        "ndithandizeni", "nitandizeni",

        # Distinctive Chinyanja expressions / constructions
        "ndili", "ulipo", "alipo",
        "tilipo", "mulipo",
        "ndikufuna", "sindikufuna",
        "akufuna", "tikufuna", "mukufuna",
        "ndikudziwa", "sindikudziwa",
        "ndimva", "ndikumva",
        "sindikudziwa", "sindikufuna",
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

        # Health / maternal
        "matenda", "kansa", "zizindikiro",
        "dokotala", "magazi", "mavitamini",
        "thanzo", "ululu", "kupweteka",
        "mankhwala", "chipatala",
        "mimba", "kubereka", "umoyo",
    ],

    "lozi": [
        # Greetings / expressions
        "ndalumba", "ndalumba hahulu",
        "mulumele", "kacenu",
        "muzuhile", "silelezwa",

        # Distinctive Lozi vocabulary
        "zibonelelo", "kuhula", "kushisa",
        "maviki", "mutango", "mupilo",
        "mubonelelo", "kimanzibwana",
        "butuku", "musimbi",
        "bulwazi", "wapimwa",

        # Distinctive question / sentence forms
        "ki", "kikani", "kuli",
        "mang", "kai", "kuzwa",
        "kuli cwa", "kacani", "mangani",
        "ni batla", "ha ni", "ha ku",

        # Health / maternal
        "dokota", "mwana", "musali",
        "mukuli", "mushonga", "chipatela",
        "mutumbu", "butuku", "bulwazi",
    ],

    "bemba": [
        # Greetings / expressions
        "mwaiseni", "ulishani", "muli shani",
        "mwabukeni", "mwabuka",
        "natotela", "twatotela",
        "napapata", "njelelako",

        # Distinctive Bemba expressions / constructions
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

        # Health / maternal
        "umwanakashi", "abanakashi", "wanakashi",
        "fumo", "kufuma", "iloba",
        "ifyakulya", "cilikwisa", "ubushiku",
        "icisungu", "icibemba",
        "ubuchende", "ubumi", "imiti",
        "icipatala", "dokota",
    ],

    "tonga": [
        # Greetings / expressions
        "mwabuka", "mwabukile", "mwalandwa",
        "mwalibuka", "mwaabuka",
        "ndatotela", "ndatotela kapati",

        # Distinctive Tonga vocabulary / constructions
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

        # Question words / constructions
        "nani", "lini", "kwi",
        "kaambo nzi",

        # Health / maternal
        "butami", "malwazi",
        "dokota", "mwana", "mama",
        "chibadela", "mutumbu",
        "kusilikwa", "zitondezyo",
        "mubili", "mukaintu",
    ],

    "english": [
        # Common English words
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

        # Common English verbs
        "want", "need", "know", "think",
        "understand", "help", "tell",
        "explain", "give", "get",
        "make", "go", "come",
        "feel", "say", "ask", "answer",

        # Health / maternal
        "symptoms", "information", "pregnancy",
        "pregnant", "doctor", "medicine",
        "medication", "cancer", "cervix",
        "uterus", "bleeding", "pain",
        "nausea", "vomiting",
        "antenatal", "maternal",
        "health", "appointment", "clinic",
        "hospital", "baby", "mother",
        "birth", "delivery",

        # Other domain terms
        "quote",
    ],
}
    
    language_phrases = {
    "chinyanja": [
        "muli bwanji",
        "uli ndi chani",
        "zikomo kwambiri",
        "muli bwino",
        "ndili bwino",
        "nitandizeni nankani",
        "ndithandizeni chonde",
        "chifukwa chiyani",
        "ndikufuna kudziwa",
        "sindikudziwa",
        "sindikufuna",
        "ndikufuna thandizo",
        "ndifotokozereni",
        "ndiuzeni chonde",
        "mukutanthauza chiyani",
    ],

    "shona": [
        "makadini",
        "maswera sei",
        "mamuka sei",
        "zvakadii",
        "zvakanaka sei",
        "ndatenda zvikuru",
        "maita basa",
        "ndiri bho",
        "ndiriku",
        "ndiri kuda",
        "ndinoda kuziva",
        "handizivi",
        "handina chokwadi",
        "ndibatsireiwo",
        "ndibatsirei chiremba",
        "nditsanangurirei",
        "ndiudzeiwo",
        "chii chiri kuitika",
        "zvinoita sei",
        "zvinorevei",
        "sei zvichidaro",
    ],

    "ndebele": [
        "unjani wena",
        "linjani wena",
        "ngiyabonga kakhulu",
        "siyabonga kakhulu",
        "sicela ungichazele",
        "ngicela ungisize",
        "ngicela ungitshele",
        "ngifuna ukwazi",
        "angikwazi",
        "angazi ukuthi",
        "ngidinga usizo",
        "ngifuna usizo",
        "ngitshele ukuthi",
        "ngichazele ukuthi",
        "kwenzakalani",
        "kwenzenjani",
        "kutshoni lokhu",
        "kungani kunjalo",
        "kuyini lokhu",
    ],

    "lozi": [
        "uli bwanji",
        "ni bata",
        "ha ndi zibi",
        "ndalumba hahulu",
        "ndalumba kapati",
        "muli bwanji",
        "ni batla",
        "ha ni zibi",
        "ni tusa",
        "ni tusa hahulu",
        "ndifundise",
        "ni bolelele",
        "ki kaambang",
        "kikani seo",
    ],

    "bemba": [
        "muli shani",
        "napapata",
        "nshishibe",
        "bushe kuti",
        "mwaiseni",
        "ulishani",
        "muli shani imwe",
        "natotela sana",
        "twatotela sana",
        "ndefwaya ukwishiba",
        "nshifwaya",
        "ndefwaya",
        "ngafweniko",
        "njelelako",
        "ndefwaya ukweba",
        "muntungileko",
        "bushe kuti mwangafwa",
        "inshi cintu ici",
        "kwisa kuti",
    ],

    "tonga": [
        "mmuli buti",
        "ndakomba",
        "sena kuti",
        "ndatotela kapati",
        "kaambo nzi",
        "ndiyanda kuti",
        "ndikonzya",
        "takonzyi",
        "tandizi",
        "ndilizi",
        "ndiyanda kuziba",
        "ndikomba kuti",
        "muni ndibatsire",
        "ndibatsire chonde",
        "nzi kuti",
        "nzi kaambo",
    ],

    "english": [
        "how are you",
        "what are",
        "what is",
        "what does",
        "what do you",
        "can you",
        "could you",
        "would you",
        "tell me",
        "tell me about",
        "i need",
        "i want",
        "i would like",
        "please tell",
        "please explain",
        "please help",
        "help me",
        "how do i",
        "how can i",
        "what should",
        "what should i",
        "what can i",
        "why is",
        "why are",
        "where can i",
        "when should i",
        "watch out",
        "i don't know",
        "i am pregnant",
        "i am having",
    ],
}

    scores = {lang: 0 for lang in language_keywords}

    for lang, phrases in language_phrases.items():
        for phrase in phrases:
            if phrase in message_lower:
                scores[lang] = scores.get(lang, 0) + 5

    for lang, keywords in language_keywords.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", message_lower):
                scores[lang] = scores.get(lang, 0) + 3

    max_score = max(scores.values()) if scores else 0

    if max_score > 0:
        top_langs = [lang for lang, s in scores.items() if s == max_score]
        logging.info(f"[fallback] Keyword scores: {scores} -> candidates: {top_langs}")

        if len(top_langs) == 1:
            # Unique winner — trust it even if it differs from current_lang.
            # (No more "current_lang wins ties" bias — a real tie is
            # handled below, but a clean win is a clean win.)
            return top_langs[0]

        # Genuine tie between two or more languages with no LLM signal
        # to break it — only *now* does current_lang get any preference,
        # and only as a last resort among equally-scored candidates.
        if current_lang in top_langs:
            return current_lang
        return top_langs[0]

    # No LLM signal and no keyword signal at all — safest fallback is to
    # keep the existing language rather than guess.
    logging.info(f"[fallback] No signal at all for {sender!r}, keeping {current_lang}")
    return current_lang
    

def is_question(prompt):
    prompt_lower = prompt.lower().strip()
    
    question_indicators = [
        "what", "how", "when", "why", "where", "who", "which", "can", "should", 
        "could", "would", "will", "does", "is", "are", "do you", "tell me about",
        "explain", "describe", "please tell", "i want to know", "i need",
        "kuti", "sei", "ndeipi", "ndiani", "kupi", "zvakadii", "zvinei", 
        "unogona", "ungandiudza", "ndapota tsanangura", "chii", "vanani",
        "rini", "kuti chii", "ndinoda kuziva", "ndapota undiudze",
        "yini", "kanjani", "nini", "ngobani", "kuphi", "kungani", "ngabe",
        "ungangitshela", "ngicela uchaze", "bengifuna ukwazi", "ngitshele",
        "chaza", "ngicela ungichazele", "kutheni", "ubani", "liphi",
        "bushe", "shani", "lili", "mulandu shani", "kwisa", "ngani", "wani",
        "kanshi", "mwe", "bushe kuti", "mukwai", "nalanda", "njisheni",
        "nalefwaya ukwishiba", "napapita njishibe", "cinga", "kabalume",
        "londololeni", "bushe ni", "ifwe", "mwebo",
        "kodi", "bwanji", "liti", "kotani", "kuti", "ndani", "kuti chani",
        "mungandiuze", "ndifunse", "fotokozani", "chonde", "chifukwa",
        "monga bwanji", "ndikufuna kudziwa", "mungandiwuza", "tanifotokozerani",
        "kodi mungandiuze", "pamene", "chiyani", "ndiye",
        "buti", "lili", "kuti", "ngaani", "mboni", "chakuti", "mbubuti",
        "ndapota ndijanye", "ndiyanda kuziba", "mbu", "ingwasi", "kuti kuli",
        "nga", "njise", "ndatola ndiyande", "mwami", "bakwe", "kuti mbuli",
        "mbalimbali", "taamba", "ndapota mundijanye",
        "ñi", "kai", "lini", "cwani", "kwapi", "mulanduñi", "na",
        "unga ni byela", "ndapota taluse", "ni buza", "mang", "kuti ñi",
        "kuli cwani", "wani", "ni batanga kuziba", "ndapota ni talusele",
        "na mutu", "likande", "mwa", "ñilo", "kwa", "ni ku buza",
        "na mwana", "ndapota ni byele"
    ]
    
    has_question_mark = "?" in prompt
    starts_with_question_word = any(prompt_lower.startswith(word + " ") for word in question_indicators)
    contains_question_word = any(" " + word + " " in " " + prompt_lower + " " for word in question_indicators)
    
    return has_question_mark or starts_with_question_word or contains_question_word


def get_pregnancy_data(language):
    if language == "shona":
        return pregnancy_data_shona.pregnancy_data_shona
    elif language == "ndebele":
        return pregnancy_data_ndebele.pregnancy_data_ndebele
    elif language == "chinyanja":
        return pregnancy_data_chinyanja.pregnancy_data_chinyanja
    elif language == "lozi":
        return pregnancy_data_lozi.pregnancy_data_lozi 
    elif language == "bemba":
        return pregnancy_data_bemba.pregnancy_data_bemba
    elif language == "tonga":
        return pregnancy_data_tonga.pregnancy_data_tonga
    else:
        return pregnancy_data.pregnancy_data
        
        
def get_cervical_data(language):
    if language == "shona":       
        return cervical_cancer_data.cervical_cancer_data  
    elif language == "ndebele":       
        return cervical_cancer_data.cervical_cancer_data  
    elif language == "chinyanja":       
        return cervical_cancer_data_chinyanja.cervical_cancer_data_chinyanja  
    elif language == "lozi":        
        return cervical_cancer_data_lozi.cervical_cancer_data_lozi 
    elif language == "bemba":        
        return cervical_cancer_data_bemba.cervical_cancer_data_bemba
    elif language == "tonga":        
        return cervical_cancer_data_tonga.cervical_cancer_data_tonga
    else:
        return cervical_cancer_data.cervical_cancer_data
        

def send(answer, sender, phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    type = "text"
    body = "body"
    content = answer
    image_urls = product_images.image_urls

    if "product_image" in answer:
        product_match = re.search(r'product_image_(\w+)', answer)
        if product_match:
            product_name = product_match.group(1)
            if product_name in image_urls:
                image_url = image_urls[product_name]
                mime_type, _ = guess_type(image_url.split("/")[-1])
                if mime_type and mime_type.startswith("image"):
                    type = "image"
                    body = "link"
                    content = image_url
                    answer = re.sub(r'product_image_\w+', '', answer)

    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": type,
        type: {
            body: content,
            **({"caption": answer.strip()} if type != "text" else {})
        },
    }

    response = requests.post(url, headers=headers, json=data)

    print("Send status:", response.status_code)
    print("Send response:", response.text)

    save_user_conversation(sender, "bot", answer)
    return response

def remove(*file_paths):
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)


# ─────────────────────────────────────────────
#  Continuous language detection
# ─────────────────────────────────────────────

# How often (in messages) to force a language re-check even on digit-only
# input or steps that would normally skip detection. Prevents a wrong
# language from getting permanently frozen in Redis.
LANGUAGE_RECHECK_EVERY_N = 8


def maybe_update_language(sender, prompt):
    """
    Re-detect language on incoming messages (after registration).
    - Normally skips pure-digit messages (menu choices, week numbers) to
      save LLM calls.
    - Every LANGUAGE_RECHECK_EVERY_N messages, forces a re-check using
      recent conversation history even if the current message is digits,
      so a bad early detection doesn't stay wrong forever.
    Returns the (possibly updated) language string.
    """
    state = user_states[sender]
    current_step = state.get("step", "main_menu")

    if current_step in ["language_detection", "registration"]:
        return state.get("language", "english")

    current_lang = state.get("language", "english")

    # Track how many messages we've seen since the last real check.
    msg_count = state.get("_lang_check_counter", 0) + 1
    state["_lang_check_counter"] = msg_count

    is_digit_only = prompt.strip().isdigit()
    force_recheck = (msg_count % LANGUAGE_RECHECK_EVERY_N == 0)

    if is_digit_only and not force_recheck:
        return current_lang

    if is_digit_only and force_recheck:
        # Don't classify the digit string itself — pull the last
        # non-digit thing the user actually said and classify that.
        history = get_user_conversation(sender)
        last_text = None
        for entry in reversed(history):
            if entry.get("role") != "user":
                continue
            msg = entry.get("message", "").strip()
            if msg and not msg.isdigit():
                last_text = msg
                break
        if not last_text:
            return current_lang
        detected = detect_language(last_text, sender)
    else:
        detected = detect_language(prompt, sender)

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
    

def auto_register(sender):
    """Derive the last 4 digits from the WhatsApp number and create the user ID."""
    state = user_states[sender]
    digits = re.sub(r"\D", "", sender)[-4:]          # sender is already digits only, this is just defensive
    letters = ''.join(random.choices(string.ascii_uppercase, k=4))
    state["phone_digits"] = digits
    state["user_id"] = f"DH-{digits}-{letters}"
    state["registered"] = True
    state["step"] = "main_menu"
    return state["user_id"]


def send_welcome_with_id(sender, phone_id, lang, user_id):
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
        sender, phone_id
    )


def handle_language_detection(sender, prompt, phone_id):
    detected_lang = detect_language(prompt, sender)
    user_states[sender]["language"] = detected_lang
    user_states[sender]["needs_language_confirmation"] = False

    user_id = auto_register(sender)
    send_welcome_with_id(sender, phone_id, detected_lang, user_id)
    save_single_user_state(sender)


def handle_registration(sender, prompt, phone_id):
    state = user_states[sender]
    if state.get("phone_digits") is None:
        user_id = auto_register(sender)
        send_welcome_with_id(sender, phone_id, state["language"], user_id)
    else:
        state["step"] = "main_menu"
    save_single_user_state(sender)
    

def handle_follow_up(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    greeting_words = [
        "hi", "hello", "hey", "hie",
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        "sawubona", "salibonani",
        "moni", "muli bwanji",
        "buti", "muli buti", "mwatambulwa",
        "mwaiseni", "muli shani",
        "mwa bona",
    ]
    if _contains_signal(prompt_lower, greeting_words):
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        greet_map = {"shona":"Mhoroi! Ndingakubatsirei nhasi?","ndebele":"Sawubona! Ngingakusiza ngani namuhla?","chinyanja":"Moni! Ndingakuthandizireni lero?","lozi":"Mwa bona! Nka ku thusa ka mini sunu?","tonga":"Muli buti! Nga ndamukwasya buti sunu?","bemba":"Muli shani! Bushe kuti namwafwa shani lelo?"}
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        save_single_user_state(sender)
        return

    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "awe", "pepe", "cha", "ayi"]

    if _contains_signal(prompt_lower, no_responses):
        _ask_purchase_interest(sender, phone_id, lang)
        return

    if len(prompt_lower.split()) > 2:
        _send_thinking(sender, phone_id, lang)

    reply = ask_gemini_general(prompt, lang, sender=sender)
    send(reply, sender, phone_id)
    _send_more_questions(sender, phone_id, lang)

    state["step"] = "general_followup"
    save_single_user_state(sender)


def is_exact_match(text, responses):
    words = re.findall(r"\b\w+\b", text)
    return any(word in responses for word in words)


def _send_thinking(sender, phone_id, lang):
    thinking_map = {
        "shona": "Ndiri kufunga...",
        "ndebele": "Ngiyacabangisisa...",
        "chinyanja": "Ndikuganiza...",
        "tonga": "Ndichiyandaula...",
        "bemba": "ndefwailisha...",
        "lozi": "Ni nahana...",
    }
    send(thinking_map.get(lang, "Let me think..."), sender, phone_id)


def _send_more_questions(sender, phone_id, lang):
    more_map = {
        "shona": "Pane chimwe chamunoda kubvunza here?",
        "ndebele": "Uneminye imibuzo yini?",
        "chinyanja": "Kodi muli ndi mafunso ena?",
        "tonga": "Hena muli a mubuzyo?",
        "bemba": "Uli ne fipusho nafimbi?",
        "lozi": "O na mabvuzo a mangi?",
    }
    send(more_map.get(lang, "Do you have any more questions?"), sender, phone_id)


def handle_general_followup(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    greeting_words = [
        "hi", "hello", "hey", "hie",
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        "sawubona", "salibonani",
        "moni", "muli bwanji",
        "mwabuka", "mwabuka buti", "mwalandwa", "mwalandwa buti",
        "mwaiseni", "muli shani",
        "mwa bona",
    ]
    reset_keywords = ["start over", "restart", "new conversation", "main menu", "menu", "reset"]

    is_greeting = _contains_signal(prompt_lower, greeting_words)
    is_reset    = _contains_signal(prompt_lower, reset_keywords)

    if is_greeting or is_reset:
        reset_conversation(sender)
        state = user_states[sender]
        lang  = state["language"]
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
            "tonga": "Mwabonwa! Hena nga ndamukyasya buti sunu?",
            "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
        }
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        save_single_user_state(sender)
        return

    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "inde"]
    no_responses  = ["no", "nah", "aiwa", "kwete", "hapana", "nope", "cha", "ayi"]

    if _contains_signal(prompt_lower, yes_responses):
        ask_map = {
            "shona": "Bvunzai mubvunzo wenyu.",
            "ndebele": "Ngiyacela ubuze umbuzo wakho.",
            "tonga": "Amubuye mubuyo",
            "chinyanja": "Chonde funsani funso lanu.",
            "bemba": "Nomba, ipusha ilipusho lyobe.",
            "lozi": "Nkumbira ubuze mubvuzo wako.",
        }
        send(ask_map.get(lang, "Please ask your question."), sender, phone_id)
        state["step"] = "general_question"
        save_single_user_state(sender)
        return

    if _contains_signal(prompt_lower, no_responses):
        _ask_purchase_interest(sender, phone_id, lang)
        return

    if len(prompt_lower.split()) > 2:
        _send_thinking(sender, phone_id, lang)

    reply = ask_gemini_general(prompt, lang, sender=sender)
    send(reply, sender, phone_id)
    _send_more_questions(sender, phone_id, lang)

    state["step"] = "general_followup"
    save_single_user_state(sender)


def ask_follow_up_question(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    followup_map = {
        "shona": "Pane chimwe chandingakubatsira nacho here?",
        "ndebele": "Ingabe kukhona okunye engingakusiza ngakho?",
        "tonga": "Hena muli amubuyo umbi?",
        "chinyanja": "Kodi pali zina zomwe ndingakuthandizireni?",
        "bemba": "Kuli fintu fimbi ifyo ningamwafwilisha?",
        "lozi": "Ki sina sika ni ka thusa ka sona",
    }
    send(followup_map.get(lang, "Is there anything else I can help you with?"), sender, phone_id)
    
    state["step"] = "follow_up"
    save_single_user_state(sender)


def switch_language_and_respond(sender, prompt, phone_id, current_lang, detected_lang):
    state = user_states[sender]
    state["language"] = detected_lang
    
    current_step = state.get("step", "main_menu")
    logging.info(f"Language switch detected: {current_lang} -> {detected_lang} at step: {current_step}")
    
    if current_step == "main_menu":
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
            "tonga": "Mwabonwa! Hena nga ndamukwasya buti sunu?",
            "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
        }
        send(greet_map.get(detected_lang, "Hello! How can I help you today?"), sender, phone_id)
    
    elif current_step == "registration":
        if state.get("phone_digits") is None:
            reg_map = {
                "shona": "Mhoro! Reggai titange nekunyoresa. Ndapota ndipe manhamba mana ekupedzisira enhare yenyu.",
                "ndebele": "Sawubona! Masige saqala ngokubhalisa. Ngicela unginike amadijithi amane okugcina efoni yakho.",
                "chinyanja": "Moni! Tiyambireni ndi kulembetsa. Chonde ndipatseni manambala anayi omaliza a nambala yanu yafoni.",
                "lozi": "Mwa bona! A re simule ka ku itambula. Dinomolo za mafelele a lina la wena ki zini za mafelele a mane?",
                "tonga": "Mwa bonwa! Atutalikile kulembesha. Amundipe ma nambala ali 4 ali kumamanino ya foni namba yenu",
                 "Bemba": "Mwaiseni! Pakutampa Lembesheni. Lembeni ama namba ayali 4 ayashalikisha kuli namba yenu",
            }
            send(reg_map.get(detected_lang, "Hello! Let's start with registration. What is the last 4 digits of your number?"), sender, phone_id)
    
    elif current_step == "ask_week":
        week_map = {
            "shona": "Ndapota isa vhiki re pamuviri ",
            "ndebele": "Sicela ufake iviki lokukhulelwa ",
            "chinyanja": "Chonde lowetsani sabata la pakati ",
            "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
            "Tonga": " Amubike namba yama wiki yo mwaba andaa (yo mwaba akaati)",
            "Bemba": "Lembeni imilungu mwaba pabukulu",
        }
        send(week_map.get(detected_lang, "Please enter your pregnancy week number "), sender, phone_id)
    
    save_single_user_state(sender)


def handle_main_menu(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    logging.info(f"User {sender} said: '{prompt}' (lowercase: '{prompt_lower}')")
    logging.info(f"Current state: step={state.get('step')}, topic={state.get('topic')}, language={lang}")

    reset_keywords = ["start over", "restart", "new conversation", "main menu", "menu", "reset"]
    greeting_words = [
        "hi", "hello", "hey", "hie", "hi there", "good morning", "good afternoon", "good evening",
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        "sawubona", "salibonani",
        "moni", "muli bwanji",
        "mwabuka", "mwabuka buti", "mwatambulwa", "buti",
        "mwaiseni", "muli shani",
        "mwa bona",
    ]
    
    is_reset = _contains_signal(prompt_lower, reset_keywords)
    is_greeting = _contains_signal(prompt_lower, greeting_words)
    
    if is_greeting or is_reset:
        reset_conversation(sender)
        state = user_states[sender]
        lang = state["language"]
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
            "tonga": "Buti! Nga ndamukwasya buti sunu?",
            "bemba": "Muli shani! Bushe kuti namwafwa shani lelo?",
        }
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        save_single_user_state(sender)
        return

    current_step = state.get("step")
    
    if current_step == "ask_another_week":
        handle_another_week(sender, prompt, phone_id)
        return
        
    if current_step == "cervical_more_info":
        handle_cervical_more_info(sender, prompt, phone_id)
        return
        
    if current_step == "cervical_question_number":
        handle_cervical_question_number(sender, prompt, phone_id)
        return
        
    if current_step == "keep_learning":
        handle_keep_learning(sender, prompt, phone_id)
        return
        
    if current_step == "follow_up":
        handle_follow_up(sender, prompt, phone_id)
        return
        
    if current_step == "product_inquiry":
        handle_purchase_response(sender, prompt, phone_id)
        return
        
    if current_step == "confirm_purchase":
        handle_purchase_confirmation(sender, prompt, phone_id)
        return

    if current_step == "general_followup":
        return handle_general_followup(sender, prompt, phone_id)

    if state.get("step") == "choose_info_type":
        if prompt_lower in ["1", "general", "information", "info", "ruzivo", "ulwazi", "zambiri"]:
            if state.get("topic") == "maternal":
                state["step"] = "ask_week"
                week_map = {
                    "shona": "Ndapota isa vhiki re pamuviri ",
                    "ndebele": "Sicela ufake iviki lokukhulelwa ",
                    "chinyanja": "Chonde lowetsani sabata la pakati ",
                    "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
                    "Tonga": " Amubike namba yama wiki yo mwaba andaa (yo mwaba akaati)",
                    "bemba": "Napapita, ingisha umulungu wa pa nkundi ",
                }
                send(week_map.get(lang, "Please enter your pregnancy week number:"), sender, phone_id)
            elif state.get("topic") == "cervical":
                cervical_data = get_cervical_data(lang)
                if cervical_data and len(cervical_data) > 0:
                    send(str(cervical_data[0]), sender, phone_id)
                else:
                    no_data_map = {
                        "shona": "Ndine urombo, handina kuwana ruzivo rwe cervical cancer parizvino.",
                        "ndebele": "Uxolo, anginayo imininingwane ye-cervical cancer okwamanje.",
                        "chinyanja": "Pepani, sindinapeze zambiri za cervical cancer panopa.",
                        "lozi": "Ndine u luvile, sina kungafumula zintu za kankere ya sibete sunu.",
                    }
                    send(no_data_map.get(lang, "Sorry, I couldn't find cervical cancer information at the moment."), sender, phone_id)
                
                ask_cervical_more_info(sender, phone_id)
            save_single_user_state(sender)
            return

        elif prompt_lower in ["2", "specific", "question", "questions", "mubvunzo", "umbuzo", "funso"]:
            if state.get("topic") == "maternal":
                state["step"] = "maternal_question_choice"
                q_map = {
                    "shona": (
                        "Sarudza mubvunzo:\n"
                        "1. Ndezvikaita zviratidzo zvepamuviri?\n"
                        "2. Ndeapi marairiro ezvokudya?\n"
                        "3. Ndingafanire kuona chiremba riini?"
                    ),
                    "ndebele": (
                        "Khetha umbuzo:\n"
                        "1. Ngabe yiziphi izimpawu zesisu?\n"
                        "2. Ngabe yimaphi amathiphu okudla?\n"
                        "3. Ngabe kufanele ngibone udokotela nini?"
                    ),
                    "chinyanja": (
                        "Sankhani funso:\n"
                        "1. Ndi zizindikiro zotani za pakati?\n"
                        "2. Ndi malangizo otani okudya?\n"
                        "3. Ndingafunire kuona dokotala liti?"
                    ),
                    "lozi": (
                        "U ka khetha mubuzo noma u buze mubuzo wa wena.\n"
                        "1. Zibonelelo ze ku imelela mwana zezi ntini?\n"
                        "2. Ni maano a ku nwa zintu za bupilo a ka landelwa?\n"
                        "3. Nini nka ya kwa dokotela?"
                    ),
                    "tonga": (
                        "Sala mubuzyo:\n"
                        "1. Hena nga mwaiziba buti kutu muntu uli andaa olo uli akaati?\n"
                        "2. Hena zilyo nzi zyo elede kulya mukaintu uli andaa?\n"
                        "3. Hena chiindi nzi cho diyelede kubona ba dokata?"
                    ),
                    "bemba": (
                        "Sala ilipusho:\n"
                        "1. Finshi nigeshibilako ukutila ndi pabukulu?\n"
                        "2. Mabumba ya fyakulya nshi fwile ukulya?\n"
                        "3. Nfwile ukumona dokota lisa?"
                    ),
                }
                send(q_map.get(lang, (
                    "You can choose a question or ask any of your own.\n"
                    "1. What are common pregnancy symptoms?\n"
                    "2. What nutrition tips should I follow?\n"
                    "3. When should I see a doctor?"
                )), sender, phone_id)
            elif state.get("topic") == "cervical":
                state["step"] = "cervical_question_choice"
                cq_map = {
                    "shona": (
                        "Sarudza mubvunzo:\n"
                        "1. Chii chinonzi cervical cancer?\n"
                        "2. Ndezvipi zviratidzo zvekutanga zvecervical cancer?\n"
                        "3. Chii chinokonzera cervical cancer?"
                    ),
                    "ndebele": (
                        "Khetha umbuzo:\n"
                        "1. Yini i-cervical cancer?\n"
                        "2. Ngabe yiziphi izimpawu zokuqala ze-cervical cancer?\n"
                        "3. Yini ebangela i-cervical cancer?"
                    ),
                    "chinyanja": (
                        "Sankhani funso:\n"
                        "1. Ndi chiyani cervical cancer?\n"
                        "2. Ndi zizindikiro zotani zoyamba za cervical cancer?\n"
                        "3. Ndi chiyani chimayambitsa cervical cancer?"
                    ),
                    "lozi": (
                        "U ka khetha mubuzo noma u buze mubuzo wa wena.\n"
                        "1. Kankere ya sibete sa bomme ki yini?\n"
                        "2. Zibonelelo za kutanga za kankere ya sibete zezi ntini?\n"
                        "3. Zini zi bakela kankere ya sibete?"
                    ),
                    "tonga": (
                        "Sarudza mubvuzo:\n"
                        "1. Kansa ya mulomo wa cibeleko nzi?\n"
                        "2. Zizyo zyakutanga zya kansa ya mulomo wa cibeleko nzi?\n"
                        "3. Chiyambitsa kansa ya mulomo wa cibeleko nzi?"
                    ),
                    "bemba": (
                        "Sala ilipusho:\n"
                        "1. Bushe Cervical cancer nichinshi?\n"
                        "2. Finshi ningamwenako ukutila ninkwata cervical cancer?\n"
                        "3. Finshi ifileta Cervical cancer?"
                    ),
                }
                send(cq_map.get(lang, (
                    "You can choose a question or ask any of your own.\n"
                    "1. What is cervical cancer?\n"
                    "2. What are the early symptoms of cervical cancer?\n"
                    "3. What causes cervical cancer?"
                )), sender, phone_id)
            save_single_user_state(sender)
            return

        else:
            invalid_map = {
                "shona": "Pindura ne '1' kuti uwane ruzivo kana '2' kuti ubvunze mibvunzo.",
                "ndebele": "Phendula ngo-'1' ukuze uthole ulwazi noma '2' ukuze ubuze imibuzo.",
                "chinyanja": "Yankhani ndi '1' kuti mupeze zambiri kapena '2' kuti mufunse mafunso.",
                "tonga": "Ingula a '1' kuti uzibe zinji'2' kuti ubuzye",
                "bemba": "Yasuka na '1' ukuti usanga ifingi '2' Ukuti wipushe ilipusho.",
                "lozi": "Arabela ka '1' ku fumana litaba kamba '2' ku buza lipuzo.",
            }
            send(invalid_map.get(lang, "Please reply '1' for information or '2' for questions."), sender, phone_id)
            return

    if state.get("step") == "ask_week":
        try:
            week = int(re.sub(r"\D", "", prompt_lower))
            if 1 <= week <= 40:
                info_text = get_pregnancy_data(lang)
                if lang == "shona":
                    pattern = rf"\*Vhiki {week}:.*?(?=\*Vhiki {week+1}:|\Z)"
                elif lang == "ndebele":
                    pattern = rf"\*Iviki {week}:.*?(?=\*Iviki {week+1}:|\Z)"
                elif lang == "chinyanja":
                    pattern = rf"\*Sabata {week}:.*?(?=\*Sabata {week+1}:|\Z)"
                elif lang == "lozi":
                    pattern = rf"\*Sunda {week}:.*?(?=\*Sunda {week+1}:|\Z)"
                elif lang == "bemba":
                    pattern = rf"\*Umulungu {week}:.*?(?=\*Umulungu {week+1}:|\Z)"
                elif lang == "tonga":
                    pattern = rf"\*Nhwiiiki {week}:.*?(?=\*Nhwiiiki {week+1}:|\Z)"
                else:
                    pattern = rf"\*Week {week}:.*?(?=\*Week {week+1}:|\Z)"
                    
                match = re.search(pattern, info_text, re.S)
                if match:
                    header_map = {
                        "shona": f"Ruzivo rwe *Vhiki {week}:*\n\n",
                        "ndebele": f"Ulwazi lwe *Iviki {week}:*\n\n",
                        "chinyanja": f"Zambiri za *Sabata {week}:*\n\n",
                        "lozi": f"Yezi zintu za lwisisa ka bonya ku *Sunda {week}:*\n\n",
                        "bemba": f"Icibeela ca *Mulungu {week}:*\n\n",
                        "tonga": f"Zinji zya *Wiiki {week}:*\n\n",
                    }
                    header = header_map.get(lang, f"Here's information for *Week {week}:*\n\n")
                    send(f"{header}{match.group(0)}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    no_week_map = {
                        "shona": "Hapana ruzivo rwevhiki iyi.",
                        "ndebele": "Alukho ulwazi lwaleviki.",
                        "chinyanja": "Palibe zambiri za sabata ili.",
                        "lozi": "Sina zintu za ku fumwa ka viki ye.",
                        "bemba": "Tapali ifilifyonse pa mulungu uyu.",
                        "tonga": "Kunyina zinji zya wiiki iyi.",
                    }
                    send(no_week_map.get(lang, "No data available for that week."), sender, phone_id)
                    ask_another_week(sender, phone_id)
        except ValueError:
            invalid_week_map = {
                "shona": "Ndapota pinda nhamba chaiyo yevhiki kubva pa 1 kusvika pa 40.",
                "ndebele": "Sicela ufake inombolo yeviki evumelekile ephakathi kuka-1 no-40.",
                "chinyanja": "Chonde lowetsani nambala yoyenera ya sabata kuchokera pa 1 mpaka 40.",
                "lozi": "Ndapota faka linomolo la viki le li le ka 1 ku ya ka 40.",
                "bemba": "Chisuma, ingisha umulungu ukufuma pa 1 ukufika pa 40.",
                "tonga": "Ndakomba, Njisya namba ya week kutalikila a 1 kusika a 40.",
            }
            send(invalid_week_map.get(lang, "Please enter a valid week number between 1 and 40."), sender, phone_id)
            ask_another_week(sender, phone_id)
        return  

    if state.get("step") == "maternal_question_choice":
        if prompt_lower in ["1", "symptoms", "zviratidzo", "izimpawu", "zizindikiro"]:
            sym_map = {
                "shona": "Zviratidzo zvepamuviri zvinosanganisira kusvotwa, kuneta, kuvava mazamu, uye kuchinja mweya.",
                "ndebele": "Izimpawu zesisu zihlanganisa isicanucanu, ukukhathala, ubuhlungu bezebelé, nokushintsha kwemizwa.",
                "chinyanja": "Zizindikiro za pakati zimaphatikizapo kusanza, kulemba, kubvutika mabele, ndi kusintha kwa maganizo.",
                "lozi": "Limpande ze twayelehileng za buimana li akaretsa ho nyekeloa ke pelo, kukhathala, kubaba kwa matete ni kupotoloka kwa maikuto.",
                "tonga": "Zitondezyo zyakuba ada zyboneka obu: Kuseluka kumoyo, Nkolo kucisa, kukola, kukatala kapati, a kuchinja chinja kwamizezo.",
                "bemba": "Ifyo wingeshibilako ukuti naukwata ifumo, Umuselu, ukunakilila kwama bele, no kuchinja kwamisango ne fichitwa.",
            }
            send(sym_map.get(lang, "Common pregnancy symptoms include nausea, fatigue, breast tenderness, and mood swings."), sender, phone_id)
    
        elif prompt_lower in ["2", "nutrition", "zvokudya", "ukudla", "kudya" "kulya" "Ukulya"]:
            nut_map = {
                "shona": "Marairiro ezvokudya: Idya chikafu chakaringana, wedzera folic acid uye iron, uye nwa mvura yakawanda.",
                "ndebele": "Amathiphu okudla: Yidla ukudla okunempilo, khulisa i-folic acid ne-iron, futhi uhlale unamandla.",
                "chinyanja": "Malangizo okudya: Idyani chakudya chabwino, onjezerani folic acid ndi iron, ndipo muzikhala ndi madzi.",
                "lozi": "Litaba za swakudya: Ja swakudya se se lekalekanang, engetsa kufumana folic acid ni iron, mi u nne u nwa mezi a mangi.",
                "tonga": "Zilyo zyo mwelede kulya: Amulye zilyo zyelede, engesha folic acid ni iron, anilizyo nwa maanzi amanji.",
                "bemba": "Amabumba ya fyakulya: Lya ifya kulya ifya balansa, Mulungizye zilyo zigisi folic acid a iron inji, Amunye menda manji.",
            }
            send(nut_map.get(lang, "Nutrition tips: Eat balanced meals, increase folic acid and iron intake, and stay hydrated."), sender, phone_id)
    
        elif prompt_lower in ["3", "doctor", "chiremba", "udokotela", "dokotala" "dokota"]:
            doc_map = {
                "shona": "Enda kuchiremba kana uine kurwadziwa kwakanyanya, kubuda ropa kwakawanda, kana fivha yepamusoro.",
                "ndebele": "Iya kudokotela uma unobuhlungu obukhulu, ukuphuma kwegazi okukhulu, noma imfiva ephezulu.",
                "chinyanja": "Pitani kudokotala ngati muli ndi kupweteka kwakukulu, kutuluka magazi ambiri, kapena malungo apamwamba.",
                "lozi": "Bona ngaka kapili ha u ka ba ni buhlungu bo boholo, kuelwa mali a mangi, kamba mufufutso o mutuna.",
                "tonga": "Bona dokotela cakufwambana, kutuluka magazi amanji, naa malungo apamwamba.",
                "bemba": "Kamubona musilisi mukufwambaana kuti mwamvwa kucisa kapati, kuswa bulowa bunji, naa kupya kapati mubili.",
            }
            send(doc_map.get(lang, "See a doctor immediately if you experience severe pain, heavy bleeding, or high fever."), sender, phone_id)
    
        else:
            _send_thinking(sender, phone_id, lang)
            gemini_response = ask_gemini(prompt, lang, sender=sender)  
            send(gemini_response, sender, phone_id)
        
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    if state.get("step") == "cervical_question_choice":
        if prompt_lower in ["1", "what is it", "what is cervical cancer", "chii", "yini", "chiyani"]:
            cc_what_map = {
                "shona": "Cervical cancer chirwere che cervix, chikamu chezasi chechibereko chinobatana nechibereko. Ndicho chirwere chegomarara chechipiri chinowanikwa zvakanyanya pasi rose uye ndicho chinonyanya kuitika kuvakadzi muZambia. Chirwere chinodzivirika uye chinorapika, kunyanya kana chikaonekwa nekukurumidza.",
                "ndebele": "I-cervical cancer yisifo se-cervix, ingxenye engezansi yesibeletho ehlobene nesibeletho. Yisifo somhlaza sesibili esivame kakhulu emhlabeni wonke futhi yisifo esivame kakhulu kwabesifazane eZambia. Isifo esingavinjwa futhi singelapheka, ikakhulukazi uma sitholakala ngokushesha.",
                "chinyanja": "Cervical cancer ndi matenda a cervix, gawo lotsika la chibereko lomwe limagwirizana ndi chibereko. Ndimatenda a kansa wachiwiri omwe amapezeka kwambiri padziko lapansi ndipo ndi omwe amachitika kwambiri kwa amayi ku Zambia. Matenda omwe angapweke ndi opatsirika, makamaka akadziwika msanga.",
                "lozi": "Kansa ya mulomo wa popelo ki malwale a mulomo wa popelo, sipande sa fafasi sa popelo se si kopanya kwa mukutu wa botsadi. Ki kansa ya bobeli e atile hahulu kwa basali mwa lifasi kaufela, mi ki yona e atile hahulu kwa basali mwa Zambia. Ki malwale a ka thibelwa ni ku alafiwa, haholoholo ha a lemohuoa kapili.",
                "tonga": "Kansa yamulomo wacizalo (cervical cancer) bulwazi bwamulomo wacizalo, eco icili mbali yansi yacizalo icisunganya kucitenge. Oyu ngomushobo wakansa wabili udumide kapati mubukaji mukwasi woonse, alimwi ngomushobo udumide kapati kuli bamakaintu mucisi ca Zambia. Obu mbulwazi bukonzya kukwabililwa alimwi akusilikwa, kapati kuti bwajanwa acifwambaana.",
                "bemba": "Kansa ya cevix bulwele bwaku cevix, iyaba kushi kwa chisa ichikatisha to bwanakashi. Iyi kansa yaba iyachibili pama kansa eya sangwa sana muli banamayo pano isonde lyonse, mu zambia eyisangwa sana.Kuti yachingililwa no kuposhiwa, sana sana ngelyo iletampa ilyo taila kula",
            }
            send(cc_what_map.get(lang, "Cervical cancer is a disease of the cervix, the lower part of the uterus that connects to the vagina. It is the second most common female malignancy worldwide and the most common in females in Zambia. It is a preventable and treatable disease, especially when detected early."), sender, phone_id)

        elif prompt_lower in ["2", "symptoms", "early symptoms", "zviratidzo", "izimpawu", "zizindikiro", "Zitondezyo"]:
            cc_sym_map = {
                "shona": "Mumatanho ekutanga, cervical cancer kazhinji haina zviratidzo zvinooneka. Ndokusaka kuongororwa nguva nenguva kwakakosha. Sezvo cancer ichikura, zviratidzo zvinogona kusanganisira kubuda ropa kusingawanzo, kubuda kwezvipembenene zvinonhuwa, kana kurwadziwa panguva yekuita bonde.",
                "ndebele": "Ezitebhisini zokuqala, i-cervical cancer ivamise ukungabi nezimpawu ezibonakalayo. Yingakho ukuhlolwa ngesikhathi esithile kubalulekile. Njengoba umhlaza ukhula, izimpawu zingahlanganisa ukuphuma kwegazi okungajwayelekile, ukuphuma kokomkhando olunephunga elibi, noma ubuhlungu ngesikhathi sokwenza ucansi.",
                "chinyanja": "M'magawo oyamba, cervical cancer imayambira mosazindikika. Ndi chifukwa chake kuyezetsa nthawi ndi nthawi ndi kofunikira. Pomwe kansa ikukula, zizindikiro zingakhale kutuluka magazi osayembekezereka, kutuluka kwa chinyezi choipa, kapena kupweteka panthawi ya kugonana.",
                "lozi": "Ka nako ya makalelo, kansa ya mulomo wa sibeleko ha i na mabonelo a bonahala. Ki sona se si ama ku lekolwa ka linako za nako ku ba kwa butokwa. Ha kansa i hula, mabonelo a kona ku akaretsa kuelwa mali ka linako ze sa lebelelwi, ku zwahela kwa tumelo ye nuna, kamba buhlungu bo ba teñi ha ku eza za bunde.",
                "tonga": "Mukutanga kwa matenda, kansa ya mulomo wa cibeleko imaziyizya mosazindikika. Ndi chifukwa chake kuyezetsa nthawi ndi nthawi ndi kofunikira.",
                "bemba": "Mu nsanga ya imituntumuko, kansa ya cibeleshi ifwilika ukuba takuli ifyo balenanga ifilumba. Ndi ifyo ifikoshi ukuyeshiwa nthawi na nthawi.",
            }
            send(cc_sym_map.get(lang, "In its early stages, cervical cancer often has no noticeable symptoms. This is why regular screening is so important. As the cancer progresses, symptoms may include unusual vaginal bleeding (between periods, after sex, or after menopause), foul-smelling vaginal discharge, or pain during sexual intercourse."), sender, phone_id)

        elif prompt_lower in ["3", "causes", "what causes it", "chikonzero", "izimbangela", "zoyambitsa"]:
            cc_cause_map = {
                "shona": "Kazhinji, cervical cancer inokonzerwa nehutachiona husingaperi hweHuman Papilloma Virus (HPV). HPV ihutachiona hwakajairika, hunotapuriranwa nekusangana pabonde. Kunyange immune system yemuviri ichibvisa hutachiona muvanhu vazhinji, hutachiona husingaperi hunogona kukonzera shanduko yamasero inogona kuzopedzisira yaita cancer.",
                "ndebele": "Ezimeni zonke, i-cervical cancer ibangelwa ukutheleleka okungapheli kwe-Human Papilloma Virus (HPV). I-HPV igciwane elivamile, elidluliselwa ngocansi. Ngenkathi amasosha omzimba emuncela igciwane kubantu abaningi, ukutheleleka okungapheli kungaholela ekushintsheni kwamaseli okungajwayelekile okungase igcine kube umhlaza.",
                "chinyanja": "M'magawo onse, cervical cancer imayambitsidwa ndi matenda osatha a Human Papilloma Virus (HPV). HPV ndi matenda amene amapezeka kwambiri, omwe amatengedwa pogonana. Pomwe immune system ya thupi imatulutsa matenda mwa anthu ambiri, matenda osatha angayambitse kusintha kwa maselo komwe kungatheka kukhala kansa.",
                "lozi": "Mwa mikwa kaufela, kansa ya mulomo wa sibeleko i bakiwa ki kulwala ka nako ye telele kwa Human Papilloma Virus (HPV). HPV ki bulwasi bo bu atile hahulu, bo bu fetisezwa ka ku eza za bunde. Niha mili wa mutu u fanga bulwasi ku batu ba bañata, ku lwala ka nako ye telele ku kona ku leza licinceho za liseli ze si za twanelo ze kona ku isa kwa kansa.",
                "tonga": "Mu milandu minji, kansa ya mulomo wa cibeleko (cervical cancer) ibambwa akaambo ka bulwazi bwa Human Papilloma Virus (HPV) busyoma mumuviri kwa ciindi cilamfu. HPV bulwazi buzyibidwe kapati, alimwi bujatikizigwa mukuswaangana kwa bulalelale. Kunyina kuti mumuviri wa bantu banji ulakonzya kuzunda no kufumya bulwazi obu, kuti bwasyoma mumuviri kwa ciindi cilamfu bulakonzya kubamba maselo a mulomo wa cibeleko kuti acince munzila itali kabotu, alimwi kumamanino kungaba kansa.",
                "bemba": "Ku milandu iingi, kansa ya kwi koli (cervical cancer) ilaisa pa mulandu wa tushishi twa Human Papilloma Virus (HPV) twashala mu mubili utushilafuma. Ubu bulwebe bwa HPV bwalyanguka kabili butantikana mu kupita mu mibele ya ku bwaume. Nangu ca kuti abantu abengi umubili wesu ubilwishako no kuposa ubu bulwebe, ukushala kwabo mu mubili kuti kwalenga ifipimo fya mubili ukwaluka, ifyo pa numa fingacita kansa.",
            }
            send(cc_cause_map.get(lang, "In almost all cases, cervical cancer is caused by persistent infection with the Human Papilloma Virus (HPV). HPV is a very common, sexually transmitted virus. While the body's immune system clears the virus in most people, a persistent infection can lead to abnormal cell changes that may eventually develop into cancer."), sender, phone_id)

        else:
            _send_thinking(sender, phone_id, lang)
            gemini_response = ask_gemini_cancer(prompt, lang, sender=sender)  
            send(gemini_response, sender, phone_id)
           
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    maternal_keywords = ["pamuviri", "pakati", "pregnancy", "pregnant", "baby", "maternal", "nhumbu", "fumo", "chibeleko", "isisu", "mimba", "buimana", "bubulemi", "kurutsa", "bump", "months", "trimester", "chibereko",]
    question_words = ["what", "how", "when", "why", "can", "should", "kuti", "sei", "ngani", "kodi", "bwanji", "chifukwa", "ndeipi", "ndiani", "nzira", "zviratidzo", "zizindikiro"]
   
    is_direct_question = (
        any(keyword in prompt_lower for keyword in maternal_keywords) and
        any(question_word in prompt_lower for question_word in question_words)
    )
   
    if is_direct_question:
        _send_thinking(sender, phone_id, lang)
        gemini_response = ask_gemini(prompt, lang, sender=sender)
        send(gemini_response, sender, phone_id)
        ask_follow_up_question(sender, phone_id)
        save_single_user_state(sender)
        return

    gemini_reply = ask_gemini_general(prompt, lang, sender=sender)
    send(gemini_reply, sender, phone_id)
    _send_more_questions(sender, phone_id, lang)
   
    state["step"] = "general_followup"
    save_single_user_state(sender)
    return
   

def _ask_purchase_interest(sender, phone_id, lang):
    state = user_states[sender]
    ask_map = {
        "shona": "Ungada here kutenga zvimwe zvezvigadzirwa zvedu? ",
        "ndebele": "Ungathanda ukuthenga noma yimuphi imikhiqizo yethu? ",
        "chinyanja": "Kodi mukufuna kugula zinthu zina mu zithu zathu? ",
        "tonga": "Mulakonzya kuyanda kuula zimwi zintu zyesu? ",
        "bemba": "Bushe kuti mwatemwa ukushita ifipe fyesu fimo? ",
        "lozi": "Kana u bata ku landa swakupila sa luna? ",
    }
    send(ask_map.get(lang, "Would you like to purchase any of our products? We have ultrasound, Birth Kits, HPV Test etc."), sender, phone_id)
    state["step"] = "shop_interest"
    save_single_user_state(sender)


def _send_shop_categories(sender, phone_id, lang):
    lines = []
    header_map = {
        "shona": "🛒 Makategi eZvigadzirwa:\n",
        "ndebele": "🛒 Imigqa Yemikhiqizo:\n",
        "chinyanja": "🛒 Mitundu ya Zinthu:\n",
        "tonga": "🛒 Misela ya Zintu:\n",
        "bemba": "🛒 Imisango ya fipe:\n",
        "lozi": "🛒 Mibeko ya Swakupila:\n",
    }
    lines.append(header_map.get(lang, "🛒 Product Categories:\n"))

    for idx, (cat_name, items) in enumerate(products_by_category.items(), 1):
        lines.append(f"*{idx}. {cat_name}*")
        for item in items[:2]:
            lines.append(f"   • {item['name']} — {item['price']}")
        if len(items) > 2:
            more_map = {
                "shona": f"   ...uye zvimwe {len(items)-2}",
                "ndebele": f"   ...namanye a-{len(items)-2}",
                "chinyanja": f"   ...ndi ena {len(items)-2}",
                "tonga": f"   ...azimwi {len(items)-2}",
                "bemba": f"   ...na fimo {len(items)-2}",
                "lozi": f"   ...ni ze ñwi {len(items)-2}",
            }
            lines.append(more_map.get(lang, f"   ...and {len(items)-2} more"))
        lines.append("")

    prompt_map = {
        "shona": "Tumira nhamba yekategi kuti uone zvigadzirwa zvose, kana udza zita rechigadzirwa chaunoda kutenga.",
        "ndebele": "Thumela inombolo yomugqa ukuze ubone wonke umkhiqizo, noma sitshele igama lomkhiqizo ofuna ukuthenga.",
        "chinyanja": "Tumizani nambala ya gulu kuti muone zinthu zonse, kapena uzani dzina la chinthu mukufuna kugula.",
        "tonga": "Tumizya nambala ya musela kuti mubone zintu zyoonse, naa mulembe zina lya cintu ncimuyanda kuuula.",
        "bemba": "Tuma nambala yafi putulwa pa kutila mumone ifintu fyonse ifilipo, nelyo bikeni ishina lya fintu mulefwaya ukushita.",
        "lozi": "Lumeza nomolo ya sibaka ku bona swakupila kaufela, kamba u bulele libizo la swakupila u bata ku landa.",
    }
    lines.append(prompt_map.get(lang, "Send the category number to see all products, or tell us the name of the product you'd like to order."))

    send("\n".join(lines), sender, phone_id)
    state = user_states[sender]
    state["step"] = "shop_browse"
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


def handle_shop_interest(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id, lang)
    elif intent == "decline":
        bye_map = {
            "shona": "Zvakanaka! Iva nezuva rakanaka. Tanga patsva nekuti 'hesi' kana uine mimwe mibvunzo.",
            "ndebele": "Kulungile! Ube nosuku oluhle. Qala kabusha ngo-'unjani' uma uneminye imibuzo.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Yambani ndi 'muli bwanji' ngati muli ndi mafunso.",
            "tonga": "Kabotu! Mukale kabotu. Amutalike alimwi akulemba kuti 'mwabuka buti' kuti muli amibvunzo imbi.",
            "bemba": "Chisuma, Mwikale bwino. Lembeni 'shani' nga muli ne nefipusho nafimbi.",
            "lozi": "Ho lokile! Mube ni lizazi le linde. Qalisa ndi 'mwa bona' ha mu na lipuzo.",
        }
        send(bye_map.get(lang, "Alright! Have a nice day. Say 'hi' if you have more questions."), sender, phone_id)
        reset_conversation(sender)
    else:
        _send_shop_categories(sender, phone_id, lang)


def handle_shop_browse(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    categories = list(products_by_category.keys())

    if prompt_lower.isdigit():
        idx = int(prompt_lower) - 1
        if 0 <= idx < len(categories):
            cat_name = categories[idx]
            all_items = products_by_category[cat_name]
            lines = [f"🏥 *{cat_name}*\n"]
            for i, item in enumerate(all_items, 1):
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
            lines.append(order_map.get(lang, "\nWould you like to order any of these products? Reply 'yes' and tell us the product name, or 'no'."))
            send("\n".join(lines), sender, phone_id)
            state["step"] = "shop_order_decision"
            state["shop_category"] = cat_name
            save_single_user_state(sender)
            return
        else:
            invalid_map = {
                "shona": f"Nhamba isiriyo. Ndapota sarudza pakati pa 1 ne {len(categories)}.",
                "ndebele": f"Inombolo engavumelekile. Ngicela ukhethe phakathi kuka-1 no-{len(categories)}.",
                "chinyanja": f"Nambala yolakwika. Chonde sankhani pakati pa 1 ndi {len(categories)}.",
                "tonga": f"Nambala teyilungeme. Chabota, musale pakati pa 1 a {len(categories)}.",
                "bemba": f"Napapata, sala nambala pakati pa 1 na {len(categories)}.",
                "lozi": f"Nomolo e si ya. Ndapota, khetha ku zwana 1 ku ya ku {len(categories)}.",
            }
            send(invalid_map.get(lang, f"Invalid number. Please choose between 1 and {len(categories)}."), sender, phone_id)
            return

    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"] = matched["price"]
        _ask_quantity(sender, phone_id, lang, matched["name"])
        return

    _send_shop_categories(sender, phone_id, lang)


def handle_shop_order_decision(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]

    if _contains_signal(prompt_lower, yes_responses):
        ask_which_map = {
            "shona": "Zvakanaka! Nyora zita rechigadzirwa chaunoda kuodha.",
            "ndebele": "Kulungile! Bhala igama lomkhiqizo ofuna ukuwodha.",
            "chinyanja": "Chabwino! Lemba dzina la chinthu chomwe mukufuna kugula.",
            "tonga": "Kabotu! Mulembesye zina lya cintu ncimuyanda kuula.",
            "bemba": "Chawama! nomba Lemba ishina lya fyo ulefwaya ukushita.",
            "lozi": "Ho lokile! Ñola libizo la swakupila u bata ku landa.",
        }
        send(ask_which_map.get(lang, "Great! Please type the name of the product you'd like to order."), sender, phone_id)
        state["step"] = "shop_product_name"
        save_single_user_state(sender)

    elif _contains_signal(prompt_lower, no_responses):
        see_more_map = {
            "shona": "Zvakanaka! Ungada here kuona mamwe makategi? ",
            "ndebele": "Kulungile! Ungathanda ukubona eminye imigqa? ",
            "chinyanja": "Chabwino! Kodi mukufuna kuona mitundu ina? ",
            "tonga": "Kabotu! Mulakonzya kuyanda kubona misela imbi? ",
            "bemba": "Chisuma! Bushe ulefwaya ukumona ifiputulwa nangu ifipe fimbi? ",
            "lozi": "Ho lokile! Kana u bata ku bona mibeko ina? ",
        }
        send(see_more_map.get(lang, "Alright! Would you like to see other categories?"), sender, phone_id)
        state["step"] = "shop_more_categories"
        save_single_user_state(sender)
    else:
        all_products_flat = [p for items in products_by_category.values() for p in items]
        matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)
        if matched:
            state["shop_selected_product"] = matched["name"]
            state["shop_selected_price"] = matched["price"]
            _ask_quantity(sender, phone_id, lang, matched["name"])
        else:
            ask_yes_no_map = {
                "shona": "Ndapota pindura 'hongu' kana 'aiwa'.",
                "ndebele": "Ngicela uphendule 'yebo' noma 'cha'.",
                "chinyanja": "Chonde yankha 'inde' kapena 'ayi'.",
                "tonga": "Chabota, mupandule kuti 'iya' naa 'pe'.",
                "bemba": "Napapata, yasuka 'ehe' kapena 'awe'.",
                "lozi": "Ndapota, arabela 'inde' kamba 'ayi'.",
            }
            send(ask_yes_no_map.get(lang, "Please reply 'yes' or 'no'."), sender, phone_id)


def handle_shop_product_name(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next((p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()), None)

    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"] = matched["price"]
        _ask_quantity(sender, phone_id, lang, matched["name"])
    else:
        not_found_map = {
            "shona": "Handina kuwana chigadzirwa ichocho. Ndapota nyora zita rakanangana rechigadzirwa kubva kumureza.",
            "ndebele": "Angitholi umkhiqizo lowo. Ngicela ubhale igama elicacile lomkhiqizo uvela ohlwini.",
            "chinyanja": "Sindipeza chinthu ichi. Chonde lemba dzina loyenera la chinthu kuchokera pamndandanda.",
            "tonga": "Tandajana cintu eeco. Ndapota, mulembesye zina lilungene lya cintu kuzwa pamudandaano.",
            "bemba": "Nshasangile ichiputulwa ichi, lemba ishina chiputulwa ififine chilemoneka muchilangililo.",
            "lozi": "Ha ni fumani swakupila se. Ndapota, ñola libizo le li nepahala la swakupila ku luhelo.",
        }
        send(not_found_map.get(lang, "I couldn't find that product. Please type the exact product name from the list."), sender, phone_id)


def handle_shop_more_categories(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id, lang)
    elif intent == "decline":
        bye_map = {
            "shona": "Zvakanaka! Iva nezuva rakanaka. Tanga patsva nekuti 'hesi' kana uine mimwe mibvunzo.",
            "ndebele": "Kulungile! Ube nosuku oluhle. Qala kabusha ngo-'unjani'.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Yambani ndi 'muli bwanji'.",
            "tonga": "Kabotu! Mukale Kabotu. Amutalike alimwi akulemba kuti 'mwabuka buti'.",
            "bemba": "Chisuma! Mwikale bwino. Landeni 'hi' nga muli nefipusho nafimbi .",
            "lozi": "Ho lokile! Mube ni lizazi le linde. Qalisa ndi 'mwa bona'.",
        }
        send(bye_map.get(lang, "Alright! Have a nice day. Say 'hi' if you have more questions."), sender, phone_id)
        reset_conversation(sender)
    else:
        _send_shop_categories(sender, phone_id, lang)


def _ask_quantity(sender, phone_id, lang, product_name):
    state = user_states[sender]
    qty_map = {
        "shona": f"Zvakanaka! Mungada mangani e *{product_name}*?",
        "ndebele": f"Kulungile! Ufuna izinga elingakanani le *{product_name}*?",
        "chinyanja": f"Chabwino! Mukufuna kuchulukitsa *{product_name}* kangati?",
        "tonga": f"Kabotu! manji buti ma *{product_name}* njo muyanda?",
        "bemba": f"Cisuma! niyanga ama *{product_name}* mulefwaya?",
        "lozi": f"Ho lokile! U bata ku landa *{product_name}* hañata mañi?",
    }
    send(qty_map.get(lang, f"Great! How many *{product_name}* would you like?"), sender, phone_id)
    state["step"] = "shop_quantity"
    save_single_user_state(sender)


def handle_shop_quantity(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.strip()

    qty_match = re.search(r"\d+", prompt_lower)
    if qty_match:
        qty = int(qty_match.group())

        cart = state.setdefault("cart", [])
        cart.append({
            "product":  state.get("shop_selected_product", "Unknown"),
            "price":    state.get("shop_selected_price", "N/A"),
            "quantity": qty,
        })

        more_map = {
            "shona":     f"✅ *{state.get('shop_selected_product')}* x{qty} yawedzerwa! Ungada here kuwedzera chimwe chigadzirwa?",
            "ndebele":   f"✅ *{state.get('shop_selected_product')}* x{qty} yengezwe! Ungathanda ukwengeza umkhiqizo wolunye?",
            "chinyanja": f"✅ *{state.get('shop_selected_product')}* x{qty} yaonjezedwa! Kodi mukufuna kuwonjezera chinthu china?",
            "tonga":     f"✅ *{state.get('shop_selected_product')}* x{qty} canjila! Sena mulayanda kunjizya cintu cimwi?",
            "bemba":     f"✅ *{state.get('shop_selected_product')}* x{qty} Yalundwapo! Bushe mulefwaya nafimbi?",
            "lozi":      f"✅ *{state.get('shop_selected_product')}* x{qty} i yemelizwe! Kana u bata ku yema swakupila si liñwi?",
        }
        send(more_map.get(lang, f"✅ *{state.get('shop_selected_product')}* x{qty} added! Would you like to add anything else?"), sender, phone_id)
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
        send(invalid_qty_map.get(lang, "Please enter a number (e.g. 1, 2, 3)."), sender, phone_id)


def _save_orders_to_redis(sender, cart, address):
    if not redis_client:
        return
    user_id = user_states[sender].get("user_id", sender)
    for item in cart:
        order = {
            "user_id": user_id,
            "sender": sender,
            "product": item["product"],
            "price": item["price"],
            "quantity": item["quantity"],
            "address": address,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }
        try:
            order_key = f"orders:{sender}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            redis_client.set(order_key, json.dumps(order))
            logging.info(f"Order saved: {order_key} -> {order}")
        except Exception as e:
            logging.error(f"Error saving order: {e}")


def _send_order_confirmation(sender, phone_id, lang, cart, address):
    def build_lines(header, addr_label, closing):
        parts = [header, ""]
        for item in cart:
            parts.append(f"  📦 {item['product']} x{item['quantity']} — {item['price']}")
        parts.append("")
        parts.append(f"📍 {addr_label}: {address}")
        parts.append(closing)
        return "\n".join(parts)

    msg_map = {
        "shona":     build_lines("✅ *Odha Yakugamuchirwa!*",    "Kero",    "Tichakubata munguva pfupi. Ndatenda! 😊"),
        "ndebele":   build_lines("✅ *Ioda Ikugunyazwe!*",       "Ikheli",  "Sizokuthinta masinyane. Ngiyabonga! 😊"),
        "chinyanja": build_lines("✅ *Dongosolo Lasinthidwa!*",  "Adilesi", "Tidzakuumbanani posachedwapa. Zikomo! 😊"),
        "tonga":     build_lines("✅ *Oda Yatambulwa!*",  "Adilesi", "Tula mwambila akaindi kaniini. Twatotela! 😊"),
        "bemba":     build_lines("✅ *Oda yapokelelwa!*",       "Aderesi", "Twalalanda naimwe mukashita akanono. Natotela! 😊"),
        "lozi":      build_lines("✅ *Landa Le Li Amuhezwi!*",   "Aderesi", "Lu ta ku ama ka nako ye nyinyani. Ndalumba! 😊"),
    }
    default = build_lines("✅ *Order Confirmed!*", "Delivery address", "We'll be in touch shortly. Thank you! 😊")
    send(msg_map.get(lang, default), sender, phone_id)


def handle_shop_address(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    address = prompt.strip()

    cart = state.get("cart", [])
    _save_orders_to_redis(sender, cart, address)
    _send_order_confirmation(sender, phone_id, lang, cart, address)
    reset_conversation(sender)


def handle_shop_add_more(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    all_products_flat = [p for items in products_by_category.values() for p in items]
    matched = next(
        (p for p in all_products_flat if p["name"].lower() in prompt_lower or prompt_lower in p["name"].lower()),
        None
    )
    if matched:
        state["shop_selected_product"] = matched["name"]
        state["shop_selected_price"]   = matched["price"]
        _ask_quantity(sender, phone_id, lang, matched["name"])
        return

    intent = _interpret_shop_intent(prompt_lower)

    if intent == "browse":
        _send_shop_categories(sender, phone_id, lang)
    elif intent == "decline":
        addr_map = {
            "shona":     "Zvakanaka! Ndapota tipa kero yako yekuendesa (guta, nharaunda, uye mamwe mashoko akabatsira).",
            "ndebele":   "Kulungile! Ngicela unike ikheli lakho lokuhambisa (idolobha, indawo, noma ulwazi olwengeziwe).",
            "chinyanja": "Chabwino! Chonde tipatseni adilesi yanu yokumanga (mzinda, dera, ndi chilichonse china chopindulitsa).",
            "tonga":     "Kabotu! Ndakomba, mutupe adilesi yanu yakuleeta zintu (tauni, ncobukala, alimwi azimwi zingatugya kuziba busena).",
            "bemba":     "Chisuma! Napapata, mpeele adilesi yenu iya kuletako ifyomuleshita (tawuni, cifulo, kabili nafimbi ifyo twingeshibilako pa nchende).",
            "lozi":      "Ho lokile! Ndapota, nipe aderesi ya hao ya ku alafa (tauni, sibaka, ni ze ñwi ze thusang).",
        }
        send(addr_map.get(lang, "Great! Please provide your delivery address (town, area, and any helpful details)."), sender, phone_id)
        state["step"] = "shop_address"
        save_single_user_state(sender)
    else:
        addr_map = {
            "shona":     "Zvakanaka! Ndapota tipa kero yako yekuendesa (guta, nharaunda, uye mamwe mashoko akabatsira).",
            "ndebele":   "Kulungile! Ngicela unike ikheli lakho lokuhambisa (idolobha, indawo, noma ulwazi olwengeziwe).",
            "chinyanja": "Chabwino! Chonde tipatseni adilesi yanu yokumanga (mzinda, dera, ndi chilichonse china chopindulitsa).",
            "tonga":     "Kabotu! Ndapota, mutupe adilesi yanu yakuleeta zintu (tauni, ncobukala, alimwi azimwi zyo nga twabelesya kuzyba busena).",
            "bemba":     "Chisuma! Napapata, mpeele adilesi yenu iya kuletako ifyomuleshita (tawuni, cifulo, kabili nafimbi ifyo twingeshibilako pa nchende).",
            "lozi":      "Ho lokile! Ndapota, nipe aderesi ya hao ya ku alafa (tauni, sibaka, ni ze ñwi ze thusang).",
        }
        send(addr_map.get(lang, "Great! Please provide your delivery address (town, area, and any helpful details)."), sender, phone_id)
        state["step"] = "shop_address"
        save_single_user_state(sender)


def handle_purchase_response(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
   
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
   
    if _contains_signal(prompt_lower, no_responses):
        bye_map = {
            "shona": "Ndatenda! Iva nezuva rakanaka. Kana uine mimwe mibvunzo, tanga patsva nekuti 'hesi'.",
            "ndebele": "Ngiyabonga! Ube nosuku oluhle. Uma uneminye imibuzo, qala ingxoxo entsha ngo-'unjani'.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Ngati muli ndi mafunso ena, yambani ponena 'muli bwanji'.",
            "tonga": "Ndaluba! Amube kabotu. Kuti muli amibuzyo imbi, amutalike alimwi akulemba kuti 'mwabuka buti'.",
            "bemba": "Natotela! Mwikale bwino. Nga muli nefipusho nafimbi, lembeni uki 'shani'.",
            "lozi": "Ndalumba! Mube ni lizazi le linde. Ha mu na lipuzo le linwi, qalisa ka ku bulela 'mwa bona'.",
        }
        send(bye_map.get(lang, "Thank you! Have a nice day. If you have more questions, start over by saying 'hi'."), sender, phone_id)
        reset_conversation(sender)
        return
       
    elif _contains_signal(prompt_lower, yes_responses):
        topic = state.get("topic")
       
        if topic == "maternal":
            maternal_products = extract_products_by_category("Maternal Health")
            if maternal_products:
                products_text = format_products_for_display(maternal_products, lang)
                send(products_text, sender, phone_id)
            else:
                no_prod_map = {
                    "shona": "Ndine urombo, hapana zvigadzirwa zvehutano hwepamuviri zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.",
                    "ndebele": "Uxolo, azikho izinto zokunakekela isisu ezitholakalayo okwamanje. Sincoma ukuya esibhedlela sethu ukuze uthole eminye imininingwane.",
                    "tonga": "Ndapota, kunyina zintu zya bumi bwa mukaintu zilimo muli cino ciindi. Twakomba kuti mwende ku kliniki yesu kuti muzyibe zimwi.",
                    "bemba": "Chabulanda, tapali ifipe fya bumi bwa bafyashi. Tulemikonkomesha ukuti mwise kuchipatala chesu pakwishibilapo ifingi.",
                    "lozi": "Ni maswabi, ha ku na swakupila swa buimana se si fumaneha cwale. Lu ku susueza ku ya kwa kiliniki ya luna.",
                }
                send(no_prod_map.get(lang, "Sorry, no maternal health products are currently available. We recommend visiting our clinic for more information."), sender, phone_id)
               
        elif topic == "cervical":
            cervical_products = extract_products_by_category("Cervical Cancer")
            if cervical_products:
                products_text = format_products_for_display(cervical_products, lang)
                send(products_text, sender, phone_id)
            else:
                no_cerv_map = {
                    "shona": "Ndine urombo, hapana zvigadzirwa zvecervical cancer zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.",
                    "ndebele": "Uxolo, azikho izinto zokuvikela isilonda somlomo wesibeletho ezitholakalayo okwamanje. Sincoma ukuya esibhedlela sethu ukuze uthole eminye imininingwane.",
                    "tonga": "Ndausa, kunyina zintu zya kansa ya mulomo wa cibeleko ziliwo lino. Tulamwaambila kuti muyende ku kliniki yesu kuti muzyibe zimwi.",
                    "bemba": "Chabulanda, tapali ifipe fya kansa ya cervix ifyasangwa pali ino nshita. Tulemikonkomesha ukuti mwise kuchipatala chesu pakwishibilapo ifingi.",
                    "lozi": "Ni maswabi, ha ku na swakupila swa kankere ya mulomo wa sibeleko se si fumaneha cwale. Lu ku susueza ku ya kwa kiliniki ya luna.",
                }
                send(no_cerv_map.get(lang, "Sorry, no cervical cancer products are currently available. We recommend visiting our clinic for more information."), sender, phone_id)
        else:
            general_products = extract_products_by_category("General")
            if general_products:
                products_text = format_products_for_display(general_products, lang)
                send(products_text, sender, phone_id)
            else:
                gen_map = {
                    "shona": "Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo.",
                    "ndebele": "Siyabonga! Sizokuthinta emizuzwini embalwa ukuze uthole eminye imininingwane.",
                    "tonga": "Ndalumba! Tulaamwaambila akaindi kaniini kuti mupewe zimwi zyakweelela kuzyiba.",
                    "bemba": "Natotela! Twalalanda naimwe mukashita fye akanono.",
                    "lozi": "Ndalumba! Lu ta ku ama ka nako ye nyinyani kuli lu file litaba ze ñwi.",
                }
                send(gen_map.get(lang, "Thank you! We'll contact you shortly for more details."), sender, phone_id)
       
        proceed_map = {
            "shona": "Ungada here kuenderera mberi nekutenga chimwe chezvigadzirwa izvi? ",
            "ndebele": "Ungathanda ukuqhubeka nokuthenga noma yini yale mikhiqizo? ",
            "chinyanja": "Kodi mukufuna kupitiriza kugula chinthu cha zinthu izi? ",
            "tonga": "Mulakonzya kuyanda kupitilila kuula zimwi zintu eezi? ",
            "bemba": "Ulefwaya ukukonkanyapo ukushita nafimbi ifipe? ",
            "lozi": "Kana u bata ku zwelapili ku landa se si liñwi sa swakupila se? ",
        }
        send(proceed_map.get(lang, "Would you like to proceed with purchasing any of these products? "), sender, phone_id)
       
        state["step"] = "confirm_purchase"
        save_single_user_state(sender)
       
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kutenga zvigadzirwa? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuthenga imikhiqizo? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kugula zinthu? ",
            "tonga": "Ti ndavwa. Ndalomba mupandule: Mulakonzya kuyanda kuula zintu? ",
            "bemba": "Nshishibe pali ifi. Yasuka: Ulefwaya ukukonkanyapo ukushita nafimbi ifipe? ",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku landa swakupila? ",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to purchase products?  "), sender, phone_id)


def handle_purchase_confirmation(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
   
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "ayi", "not really", "cha"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "yebo"]
   
    if _contains_signal(prompt_lower, no_responses):
        alright_map = {
            "shona": "Zvakanaka. Tinokutendai! Kana uine mimwe mibvunzo, tanga patsva nekuti 'hesi'.",
            "ndebele": "Kulungile. Ngiyabonga! Uma uneminye imibuzo, qala kabusha ngokuthi 'unjani'.",
            "chinyanja": "Zikomo! Khalani ndi tsiku labwino. Ngati muli ndi mafunso ena, yambani ponena 'muli bwanji'.",
            "tonga": "Kabotu. Twalumba! Kuti muli amibuzyo imbi, amutalike alimwi akulemba kuti 'mwabuka buti'.",
            "bemba": "Chisuma. Natotela! Nga muli nefipusho nafimbi, lembeni ukuti 'shani'.",
            "lozi": "Ho lokile. Ndalumba! Ha mu na lipuzo le linwi, qalisa ka ku bulela 'mwa bona'.",
        }
        send(alright_map.get(lang, "Alright. Thank you! If you have more questions, start over by saying 'hi'."), sender, phone_id)
        reset_conversation(sender)
       
    elif _contains_signal(prompt_lower, yes_responses):
        thanks_map = {
            "shona": "Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo nezvekutenga.",
            "ndebele": "Siyabonga! Sizokuthinta emizuzwini embalwa ukuze uthole eminye imininingwane ngokuthenga.",
            "chinyanja": "Zikomo! Tidzakuumbanani posachedwapa kuti mupeze zambiri zokhudza kugula.",
            "tonga": "Twalumba! Tula mwaambila akaindi kaniini kuti muzibe zimwi zyakweelela kuzyiba kujatikizya kuula kwanu.",
            "bemba": "Natotela! Twalamitumina ifyebo ifingi pafipe mwa shita nombaline.",
            "lozi": "Ndalumba! Lu ta ku ama ka nako ye nyinyani kuli lu fe litaba ze ñwi ka za ku landa.",
        }
        send(thanks_map.get(lang, "Thank you! We'll contact you shortly for more details about your purchase."), sender, phone_id)
        reset_conversation(sender)
       
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kuenderera mberi nekutenga? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuqhubeka nokuthenga? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kupitiriza kugula? ",
            "tonga": "Ti ndamvwa. Ndausa mupandule: Mulakonzya kuyanda kupitilila kugula? ",
            "bemba": "Nshishibe pali ifi. Yasuka: Bushe ulefyaya ukukonkanyapo ukushita ifipe? ",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku zwelapili ku landa? ",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to proceed with purchasing?  "), sender, phone_id)


def extract_products_by_category(category_name):
    try:
        return products_by_category.get(category_name, [])
    except Exception as e:
        logging.error(f"Error extracting products for category {category_name}: {e}")
        return []

def format_products_for_display(products_list, lang):
    if not products_list:
        empty_map = {
            "shona": "Hapana zvigadzirwa zvazvino onekwa.",
            "ndebele": "Azikho imikhiqizo etholakalayo okwamanje.",
            "chinyanja": "Palibe zinthu zitholakalayo pakali pano.",
            "tonga": "Kunyina zintu zilimo lino.",
            "bemba": "Tapali ifipe pali kano kashita.",
            "lozi": "Ha ku na swakupila se si fumaneha cwale.",
        }
        return empty_map.get(lang, "No products currently available.")
   
    header_map = {
        "shona": "🏥 Zvigadzirwa Zvehutano:\n\n",
        "ndebele": "🏥 Imikhiqizo Yezempilo:\n\n",
        "chinyanja": "🏥 Zinthu za Thanzo:\n\n",
        "tonga": "🏥 Zintu zya Bumi:\n\n",
        "bemba": "🏥 ifipe ya Buumi:\n\n",
        "lozi": "🏥 Swakupila:\n\n",
    }
    products_text = header_map.get(lang, "🏥 Health Products:\n\n")
   
    for i, product in enumerate(products_list, 1):
        name = product.get('name', 'Unknown Product')
        price = product.get('price', 'Price not available')
        availability = product.get('availability', 'Availability not specified')
       
        if lang == "shona":
            products_text += f"{i}. {name}\n   💰 Mutengo: {price}\n   📦 Kuwanikwa: {availability}\n\n"
        elif lang == "ndebele":
            products_text += f"{i}. {name}\n   💰 Inani: {price}\n   📦 Ukutholakala: {availability}\n\n"
        elif lang == "chinyanja":
            products_text += f"{i}. {name}\n   💰 Mtengo: {price}\n   📦 Kupezeka: {availability}\n\n"
        elif lang == "tonga":
            products_text += f"{i}. {name}\n   💰 Mutengo: {price}\n   📦 Kuliko: {availability}\n\n"
        elif lang == "bemba":
            products_text += f"{i}. {name}\n   💰 Imintengo: {price}\n   📦 Ukusangwa: {availability}\n\n"
        elif lang == "lozi":
            products_text += f"{i}. {name}\n   💰 Teko: {price}\n   📦 Ku fumanehanga: {availability}\n\n"
        else:
            products_text += f"{i}. {name}\n   💰 Price: {price}\n   📦 Availability: {availability}\n\n"
   
    select_map = {
        "shona": "Sarudza chirongwa nekuudza nhamba yacho.",
        "ndebele": "Khetha umkhiqizo ngokutshela inombolo yayo.",
        "chinyanja": "Sankhani chinthu ponena nambala yake.",
        "tonga": "Sala cintu akulemba nambala yaco.",
        "bemba": "Sala nambala pakuti usale ichipe.",
        "lozi": "U khethe swakupila ka ku bulela nomolo ya sona.",
    }
    products_text += select_map.get(lang, "Select a product by telling us the number.")
   
    return products_text


# ─────────────────────────────────────────────
#  MAIN CONVERSATION ROUTER
# ─────────────────────────────────────────────

def handle_conversation_state(sender, prompt, phone_id):
    state = user_states[sender]
    current_step = state.get("step")

    if current_step not in ["language_detection", "registration"]:
        maybe_update_language(sender, prompt)
        state = user_states[sender]

    if current_step == "human_agent_chat":
        relayed = relay_user_message_to_agent(sender, prompt, phone_id)
        if not relayed:
            user_states[sender]["step"] = "main_menu"
            user_states[sender].pop("agent_phone", None)
            save_single_user_state(sender)
            lang = user_states[sender].get("language", "english")
            back_map = {
                "shona": "Kubatanidza kwake kwakugumira. Ndinokudzoseredzai kuna Rudo. Ndingakubatsirei?",
                "ndebele": "Uxhumano lwakho luphelile. Sibuyela ku-Rudo. Ngingakusiza?",
                "chinyanja": "Maukumano anu amaliza. Tikubweza kwa Rudo. Ndingakuthandizireni?",
                "bemba": "Ukumana kwenu kwashila. Mwabwela kuli Rudo. Kuti ningamwafwilisha?",
                "lozi": "Puisano ya hao i fela. Ubwela kwa Rudo. Nka ku thusa?",
                "tonga": "Kulumbaana kwanu kumanizya. Mubweza kuli Rudo. Nkugwasya buti?",
            }
            send(back_map.get(lang, "Your agent session has ended. Returning you to Rudo. How can I help?"), sender, phone_id)
        return

    if current_step == "waiting_for_agent":
        if prompt.strip().upper() == "CANCEL":
            if redis_client:
                try:
                    redis_client.delete(_agent_request_key(sender))
                    redis_client.delete(_agent_rejections_key(sender))
                except Exception:
                    pass
            user_states[sender]["step"] = "main_menu"
            save_single_user_state(sender)
            lang = user_states[sender].get("language", "english")
            cancel_map = {
                "shona": "Tapota, chikumbiro chenyu chakanzurwa. Ndinokudzoseredzai kuna Rudo.",
                "ndebele": "Kulungile, isicelo senu sikhansiliwe. Sibuyela ku-Rudo.",
                "chinyanja": "Chabwino, pempho lanu lasanikizidwa. Tikubweza kwa Rudo.",
                "bemba": "Chisuma, icipemba chenu chatanshibwa. Mwabwela kuli Rudo.",
                "lozi": "Ho lokile, kopo ya hao i hamulelwa. Ubwela kwa Rudo.",
                "tonga": "Kabotu, bulombwi bwanu bwasinkiwa. Mubweza kuli Rudo.",
            }
            send(cancel_map.get(lang, "Your request has been cancelled. Returning you to Rudo."), sender, phone_id)
        else:
            check_agent_request_timeout(sender, phone_id)
        return

    if current_step not in ["language_detection", "registration"]:
        if is_human_agent_request(prompt):
            lang = user_states[sender].get("language", "english")
            connecting_map = {
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
            send(
                connecting_map.get(lang, (
                    "🔍 Looking for a human agent to assist you...\n"
                    "Please wait.\n"
                )),
                sender, phone_id
            )
            user_states[sender]["step"] = "waiting_for_agent"
            save_single_user_state(sender)
            notify_agents_of_request(sender, phone_id)
            return

    ALL_GREETINGS = [
        "hi", "hello", "hey", "hie", "hi there",
        "good morning", "good afternoon", "good evening",
        "mhoro", "mhoroi", "hesi", "makadini", "wadini",
        "sawubona", "salibonani",
        "moni", "muli bwanji",
        "mwabuka", "mbuti", "mwalandwa", "mwalandwa buti",
        "mwaiseni", "muli shani", "shani",
        "mwa bona",
    ]
    reset_keywords = ["start over", "restart", "new conversation", "main menu", "reset"]
    prompt_lower = prompt.lower().strip()

    is_greeting = _contains_signal(prompt_lower, ALL_GREETINGS)
    is_reset = _contains_signal(prompt_lower, reset_keywords)

    if (is_greeting or is_reset) and current_step not in ["language_detection", "registration"]:
        reset_conversation(sender)
        lang = user_states[sender]["language"]
        greet_map = {
            "shona": "Mhoroi! Ndingakubatsirei nhasi?",
            "ndebele": "Sawubona! Ngingakusiza ngani namuhla?",
            "chinyanja": "Moni! Ndingakuthandizireni lero?",
            "tonga": "Mwabuka buti! Nga ndamugwasya buti sunu?",
            "bemba": "Shani! Bushe kuti namwafwa shani lelo?",
            "lozi": "Mwa bona! Nka ku thusa ka mini sunu?",
        }
        send(greet_map.get(lang, "Hello! How can I help you today?"), sender, phone_id)
        return

    current_step = state.get("step")

    if current_step == "language_detection" and state.get("first_message", True):
        handle_language_detection(sender, prompt, phone_id)
    elif current_step == "registration":
        handle_registration(sender, prompt, phone_id)
    elif current_step == "ask_another_week":
        handle_another_week(sender, prompt, phone_id)
    elif current_step == "cervical_more_info":
        handle_cervical_more_info(sender, prompt, phone_id)
    elif current_step == "cervical_question_number":
        handle_cervical_question_number(sender, prompt, phone_id)
    elif current_step == "keep_learning":
        handle_keep_learning(sender, prompt, phone_id)
    elif current_step == "follow_up":
        handle_follow_up(sender, prompt, phone_id)
    elif current_step == "product_inquiry":
        handle_purchase_response(sender, prompt, phone_id)
    elif current_step == "confirm_purchase":
        handle_purchase_confirmation(sender, prompt, phone_id)
    elif current_step == "shop_interest":
        handle_shop_interest(sender, prompt, phone_id)
    elif current_step == "shop_browse":
        handle_shop_browse(sender, prompt, phone_id)
    elif current_step == "shop_order_decision":
        handle_shop_order_decision(sender, prompt, phone_id)
    elif current_step == "shop_product_name":
        handle_shop_product_name(sender, prompt, phone_id)
    elif current_step == "shop_more_categories":
        handle_shop_more_categories(sender, prompt, phone_id)
    elif current_step == "shop_add_more":
        handle_shop_add_more(sender, prompt, phone_id)
    elif current_step == "shop_quantity":
        handle_shop_quantity(sender, prompt, phone_id)
    elif current_step == "shop_address":
        handle_shop_address(sender, prompt, phone_id)
    elif current_step == "general_followup":
        handle_general_followup(sender, prompt, phone_id)
    elif current_step == "general_question":
        lang = state["language"]
        reply = ask_gemini_general(prompt, lang, sender=sender)
        send(reply, sender, phone_id)
        _send_more_questions(sender, phone_id, lang)
        state["step"] = "general_followup"
        save_single_user_state(sender)
    else:
        handle_main_menu(sender, prompt, phone_id)


def ask_cervical_more_info(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
   
    more_map = {
        "shona": "Ungada here kuwana rumwe ruzivo rwe cervical cancer? ",
        "ndebele": "Ungathanda ukuthola eminye imininingwane nge-cervical cancer? ",
        "chinyanja": "Kodi mukufuna kupeza zambiri za cervical cancer?",
        "tonga": "Mulakonzya kuyanda kuzyiba zimwi kujatikizya kansa ya mulomo wa cibeleko (ervical cancer)?",
        "bemba": "Bushe kuti wafwaya ukwishibilapo ifingi pali cervical kansa?",
        "lozi": "Kana u bata ku fumana litaba ze ñwi ka za kankere ya mulomo wa sibeleko?",
    }
    send(more_map.get(lang, "Would you like to get more information about cervical cancer?  "), sender, phone_id)
   
    state["step"] = "cervical_more_info"
    save_single_user_state(sender)

def ask_cervical_question_number(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
   
    num_map = {
        "shona": "Pinda nhamba yemubvunzo kubva pa 1 kusvika pa 100:",
        "ndebele": "Faka inombolo yombuzo kusuka ku-1 kuya ku-100:",
        "chinyanja": "Lowetsani nambala ya funso kuchokera pa 1 mpaka 100:",
        "tonga": "Mulembesye nambala ya mubuzyo kuzwa pa 1 kusika pa 100:",
        "bemba": "lemba nambala ya lipusho ukufuma pa 1 ukufika pa 100:",
        "lozi": "Kenya nomolo ya lipuzo ku zwana 1 ku ya ku 100:",
    }
    send(num_map.get(lang, "Enter a question number from 1 to 100:"), sender, phone_id)
   
    state["step"] = "cervical_question_number"
    save_single_user_state(sender)

def ask_keep_learning(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
   
    keep_map = {
        "shona": "Ungada here kuramba uchidzidza zvimwe zvinhu zve cervical cancer? ",
        "ndebele": "Ungathanda ukuqhubeka nokufunda ezinye izindaba ze-cervical cancer? ",
        "chinyanja": "Kodi mukufuna kupitiriza kuphunzira zina zambiri za cervical cancer?",
        "tonga": "Mulakonzya kuyanda kwiya zyinji zimwi kujatikizya kansa ya mulomo wa cibeleko (ervical cancer)?",
        "bemba": "Bushe ulefwaya ukukonkanyapo ukusambililapo ifingi pali cervical cancer?",
        "lozi": "Kana u bata ku zwelapili ku ithuta litaba ze ñwi ka za kankere ya mulomo wa sibeleko?",
    }
    send(keep_map.get(lang, "Would you like to keep learning more about cervical cancer?  "), sender, phone_id)
   
    state["step"] = "keep_learning"
    save_single_user_state(sender)

def handle_cervical_more_info(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
   
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "emukwayi", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi", "iyo", "awe"]
   
    if _contains_signal(prompt_lower, yes_responses):
        ask_cervical_question_number(sender, phone_id)
    elif _contains_signal(prompt_lower, no_responses):
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id)
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kuwana rumwe ruzivo? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuthola eminye imininingwane? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kupeza zambiri?",
            "tonga": "Tindamvwa. Ndapota mupandule: Mulauyanda kuzyiba zimwi?",
            "bemba": "Nshishibe pali ifi. Napapata asuka: Kuti wafyaya ukwishibilapo nafimbi?",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku fumana litaba ze ñwi?",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to get more information?  "), sender, phone_id)

def handle_cervical_question_number(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
   
    try:
        question_num = int(re.sub(r"\D", "", prompt))
        if 1 <= question_num <= 100:
            data_tuple = get_cervical_data(lang)
           
            question_found = False
            for i, item in enumerate(data_tuple):
                if f"*Question {question_num}:" in str(item):
                    question_content = str(item)
                   
                    if i + 1 < len(data_tuple) and "Answer" in str(data_tuple[i + 1]):
                        question_content += "\n" + str(data_tuple[i + 1])
                   
                    send(question_content, sender, phone_id)
                    question_found = True
                    ask_keep_learning(sender, phone_id)
                    break
           
            if not question_found:
                not_found_map = {
                    "shona": f"Ndine urombo, handina kuwana mubvunzo wenhamba {question_num}. Edza imwe nhamba kubva pa 1 kusvika pa 100.",
                    "ndebele": f"Uxolo, angikutholanga umbuzo wenombolo {question_num}. Zama enye inombolo kusuka ku-1 kuya ku-100.",
                    "chinyanja": f"Pepani, sindinapeze funso la nambala {question_num}. Yesani nambala ina kuchokera pa 1 mpaka 100.",
                    "tonga": f"Ndausaa, tindajana mubuzyo wa nambala {question_num}. Amu njisye nambala imbi kuzwa pa 1 kusika pa 100.",
                    "bemba": f"Njelelako, nshi isangile nambala ye lipusho {question_num}. Lembeni nambala imbi ukufuma pa 1 ukufika pa 100.",
                    "lozi": f"Ni maswabi, ha ni fumani lipuzo la nomolo {question_num}. Linge nomolo ye nzwi ku zwana 1 ku ya ku 100.",
                }
                send(not_found_map.get(lang, f"Sorry, I couldn't find question number {question_num}. Please try another number from 1 to 100."), sender, phone_id)
                ask_cervical_question_number(sender, phone_id)
        else:
            range_map = {
                "shona": "Ndapota pinda nhamba kubva pa 1 kusvika pa 100 chete.",
                "ndebele": "Sicela ufake inombolo ephakathi kuka-1 no-100 kuphela.",
                "chinyanja": "Chonde lowetsani nambala kuchokera pa 1 mpaka 100 basi.",
                "tonga": "Ndausa, mulembesye nambala kuzwa pa 1 kusika pa 100 buyo.",
                "bemba": "Napapata, ingisha nambala ukufuma pa 1 ukufika pa 100 fye.",
                "lozi": "Ndapota, kenya nomolo ku zwana 1 ku ya ku 100 feela.",
            }
            send(range_map.get(lang, "Please enter a number between 1 and 100 only."), sender, phone_id)
            ask_cervical_question_number(sender, phone_id)
           
    except ValueError:
        invalid_map = {
            "shona": "Ndapota pinda nhamba chaiyo kubva pa 1 kusvika pa 100.",
            "ndebele": "Sicela ufake inombolo evumelekile ephakathi kuka-1 no-100.",
            "chinyanja": "Chonde lowetsani nambala yoyenera kuchokera pa 1 mpaka 100.",
            "tonga": "Ndapota, mulembesye nambala ilungeme kuzwa pa 1 kusika pa 100.",
            "bemba": "Napapata, ingisha nambala ilingile ukufuma pa 1 ukufika pa 100.",
            "lozi": "Ndapota, kenya nomolo ye nepahezi ku zwana 1 ku ya ku 100.",
        }
        send(invalid_map.get(lang, "Please enter a valid number between 1 and 100."), sender, phone_id)
        ask_cervical_question_number(sender, phone_id)

def handle_keep_learning(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
   
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "emukwayi", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi", "awe", "iyo"]
   
    if _contains_signal(prompt_lower, yes_responses):
        ask_cervical_question_number(sender, phone_id)
    elif _contains_signal(prompt_lower, no_responses):
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id)
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kuramba uchidzidza? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukuqhubeka nokufunda? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kupitiriza kuphunzira?",
            "tonga": "Tindamvwa. Ndausa mupandule: Mulayanda kyiya?",
            "bemba": "Nshishibe pali ifi. Napapata Yasuka: Ulefwaya ukukonkanyapo ukusambilila?",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku zwelapili ku ithuta?",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to keep learning?  "), sender, phone_id)

def ask_another_week(sender, phone_id):
    state = user_states[sender]
    lang = state["language"]
   
    another_map = {
        "shona": "Ungada here kudzidza nezve mamwe mavhiki epamuviri? ",
        "ndebele": "Ungathanda ukufunda ngamanye amaviki okukhulelwa? ",
        "chinyanja": "Kodi mukufuna kudziwa za masabata ena a pakati?",
        "tonga": "Mulakonzya kuyanda kuzyiba zya maweek ambi a bubulemi?",
        "bemba": "Kuti wafwaya ukwishiba pa fya milungu yapabukulu imbi?",
        "lozi": "Kana u bata ku ithuta ka za maviki a manwi a buimana?",
    }
    send(another_map.get(lang, "Would you like to learn about other pregnancy weeks?  "), sender, phone_id)
   
    state["step"] = "ask_another_week"
    save_single_user_state(sender)


def handle_another_week(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
   
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde", "emukwayi", "yebo"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really", "cha", "ayi", "awe", "iyo"]
   
    if _contains_signal(prompt_lower, yes_responses):
        state["step"] = "ask_week"
        week_map = {
            "shona": "Ndapota isa vhiki re pamuviri ",
            "ndebele": "Sicela ufake iviki lokukhulelwa ",
            "chinyanja": "Chonde lowetsani sabata la pakati ",
            "lozi": "Ndapota faka linomolo la viki ya ku imelela mwana ",
            "tonga": "Mulembesye nambala ya week ya bubulemi yenu",
            "bemba": "Chisuma, ingisheni nambala yamilungu mwaba pabukulu ",
        }
        send(week_map.get(lang, "Please enter your pregnancy week number "), sender, phone_id)
        save_single_user_state(sender)
       
    elif _contains_signal(prompt_lower, no_responses):
        state["step"] = "product_inquiry"
        state["topic"] = "maternal"
       
        prod_offer_map = {
            "shona": "Ndatenda! Ungada here kutenga zvigadzirwa zvehutano hwepamuviri? Tinopa:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits",
            "ndebele": "Ngiyabonga! Ungathanda ukuthengwa izinto zokunakekela isisu? Sinakho:\n- Ama-Prenatal Vitamins\n- Izinto zokuhlola isisu\n- Amakhithi okunakekela isisu",
            "chinyanja": "Zikomo! Kodi mukufuna kugula zinthu za Thanzi la Amayi? Tili ndi:\n- Mavitamini a Prenatal\n- Zoyezera pakati\n- Makiti a Thanzi la Amayi",
            "tonga": "Ndalumba! Mulakonzya kuyanda kuula zintu zya buumi bwa mukaintu ulibulemi? Tuli azyo:\n- Mavitamini a Prenatal\n- Zyakupima bubulemi\n- Makiti a Bumi bwa Bakaintu Balibulemi",
            "bemba": "Twatotela! Bushe kuti mwafwaya ukushita ifipe fyapabukulu? Natukwa na:\n- Ama-Prenatal Vitamins\n- Ifyakwishibilako nga muli pabukulu\n- Makiti ya Buumi bwa banamayo",
            "lozi": "Ndalumba! Kana u bata ku landa swakupila swa buimana? Lu na:\n- Mavitamini a Prenatal\n- Swakutatuba buimana\n- Makiti a Buimana",
        }
        send(prod_offer_map.get(lang, "Thank you! Would you like to purchase maternal health products? We offer:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits"), sender, phone_id)
        save_single_user_state(sender)
       
    else:
        unclear_map = {
            "shona": "Handina kunzwisisa. Pindura ndapota: Ungada here kudzidza nezve mamwe mavhiki? ",
            "ndebele": "Angikuzwisisi. Phendula ngicela: Ungathanda ukufunda ngamanye amaviki? ",
            "chinyanja": "Sindinamve. Yankhani chonde: Kodi mukufuna kudziwa za masabata ena?",
            "tonga": "Tindamvwa. Ndapota mupandule: Mulakonzya kuyanda kuzyiba zya maweek ambi?",
            "bemba": "Nshishibe pali ifi. Napapta Yasuka: Kuti mwafwaya ukwishiba pa milungu imbi iyapabukulu?",
            "lozi": "Ha ni utwisisi. Arabela kwa ku ya: Kana u bata ku ithuta ka za maviki a manwi?",
        }
        send(unclear_map.get(lang, "I didn't understand. Please reply: Would you like to learn about other weeks?  "), sender, phone_id)


# ─────────────────────────────────────────────
#  Gemini helper functions (stateless)
# ─────────────────────────────────────────────

def _get_lang_enforce(lang: str) -> str:
    return {
        "shona":     "Pindura muchiShona chete. Usashandise Chirungu.",
        "ndebele":   "Phendula ngesiNdebele kuphela. Ungasebenzisi isiNgisi.",
        "chinyanja": "Yankhani mu Chichewa/Chinyanja basi. Osagwiritsa ntchito Chingerezi.",
        "lozi":      "Arabela ka Silozi feela. U se ke wa sebelisa Siingelesi.",
        "bemba":     "Yasuka mu Cibemba fye. Wibonfya icingeleshi.",
        "tonga":     "Mupandule mu Chitonga buyo. Mutabelezyi Ciingelezi.",
    }.get(lang, "Respond in the user's detected language only.")


def _get_fallback(lang: str) -> str:
    return {
        "shona":     "Pane dambudziko pakupindura mubvunzo wako.",
        "ndebele":   "Kunenkinga ekuphenduleni umbuzo wakho.",
        "chinyanja": "Pali vuto popanga yankho la funso lanu.",
        "tonga":     "Kuli ipenzi mukupandula mubuzyo wanu.",
        "bemba":     "Cabulanda, kuliko ubwafya pakwasuka kuyasuka ilipusho lyobe.",
        "lozi":      "Ku na bothata ka ku arabela lipuzo la hao.",
    }.get(lang, "Sorry, there was a problem getting an answer.")


def build_conversation_context(sender: str, max_turns: int = 6) -> str:
    """
    Returns the last `max_turns` exchanges (user + bot) as a formatted
    string ready to be prepended to a Gemini prompt.
    Skips system-noise messages (short ack-only bot replies).
    """
    history = get_user_conversation(sender)
    if not history:
        return ""

    recent = history[-( max_turns * 2 + 1):-1]

    lines = []
    for entry in recent:
        role    = entry.get("role", "")
        message = entry.get("message", "").strip()
        if not message:
            continue
        if role == "bot" and len(message) < 20:
            continue
        tag = "User" if role == "user" else "Assistant"
        lines.append(f"{tag}: {message}")

    if not lines:
        return ""

    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"



def _try_medgemma(question: str, lang: str, sender: str = None):
    """Answer with the fine-tuned MedGemma server if MEDGEMMA_URL is set, wrapped in safety guardrails
    (danger-sign referral first, safety footer, output cleaning). Returns None to fall back to Gemini."""
    if not medgemma_enabled():
        return None
    history = pairs_from_history(get_user_conversation(sender)) if sender else []
    text = clean_model_output(ask_medgemma(question, history=history))
    if text and looks_like_english_leak(text, lang):
        logging.warning(f"[medgemma] English leak for lang={lang}, discarding model answer")
        text = None
    return compose_reply(question, lang, text)


def ask_gemini(question: str, lang: str = "english", sender: str = None) -> str:
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)

    mg = _try_medgemma(question, lang, sender)
    if mg:
        return mg

    context = build_conversation_context(sender) if sender else ""

    instruction_body = {
        "shona": (
            "Uri mubatsiri wezvehutano hwepamuviri. "
            "Pindura mubvunzo uyu muShona yakajeka, yakapfava, uye ine ruzivo rwezvehutano:\n\n"
        ),
        "ndebele": (
            "Ungumsizi wezempilo yesisu. "
            "Phendula lo mbuzo ngesiNdebele esicacile, esilula, futhi enolwazi lwezempilo:\n\n"
        ),
        "chinyanja": (
            "Ndine mphungu wa Thanzi la Amayi. "
            "Yankhani funso ili m'Chinyanja moyenera, mosavuta, komanso moli ndi umanyambazi wa Thanzi la Amayi:\n\n"
        ),
        "lozi": (
            "Ki muthusi wa za mapilo wa buimana. "
            "Alaba lipuzo le ka Silozi se si nepahezi, se si nolofetse, ni se si na ni bupilo:\n\n"
        ),
        "bemba": (
            "Niwe kafwa wafya bumi bwaba namayo abali pabukulu. "
            "Yasuka ilipusho ilipusho ilyakonkapo bwino bwino, mukwanguka, nechishinka pa fyabumi:\n\n"
        ),
        "tonga": (
            "Ndimwi mweenzinyina wa bupilo bwa bakaintu balibulemi. "
            "Mupandule mubvunzo ooyu mu Chitonga cilimvwisika, cipepu, alimwi cizwide ulwazi lwa bupilo:\n\n"
        ),
    }.get(lang, (
        "You are a maternal health assistant. "
        "Answer the following question clearly, simply, and with accurate health information:\n\n"
    ))

    base_prompt = f"{lang_enforce}\n\n{instruction_body}{context}Current question: {question}\n\n{lang_enforce}"

    def _generate(prompt_text, temperature):
        gen_config = dict(generation_config)
        gen_config["temperature"] = temperature
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
            safety_settings=safety_settings
        )
        response = gemini_model.generate_content(prompt_text)
        try:
            text = response.text
            return text.strip() if text and text.strip() else None
        except (ValueError, AttributeError) as ve:
            logging.warning(f"[ask_gemini] Blocked/empty lang={lang}: {ve}")
            return None

    try:
        text = _generate(base_prompt, temperature=0.5)
        if text is None:
            return fallback

        if looks_like_english_leak(text, lang):
            logging.warning(f"[ask_gemini] English leak detected for lang={lang}, retrying with stricter enforcement")
            strict_prompt = (
                f"CRITICAL: Your entire reply must be written only in {lang.capitalize()}. "
                f"Do not include any English sentences or phrases, even if the question contains English words.\n\n"
                f"{base_prompt}"
            )
            retry_text = _generate(strict_prompt, temperature=0.3)
            if retry_text and not looks_like_english_leak(retry_text, lang):
                return retry_text
            return text  # fall back to first attempt if retry didn't help

        return text
    except Exception as e:
        logging.error(f"[ask_gemini Error] {type(e).__name__}: {e}")
        return fallback
        

def ask_gemini_cancer(question: str, lang: str = "english", sender: str = None) -> str:
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)

    mg = _try_medgemma(question, lang, sender)
    if mg:
        return mg

    context = build_conversation_context(sender) if sender else ""

    instruction_body = {
        "shona": (
            "Uri mubatsiri wezvehutano hwegomarara rechibereko. "
            "Pindura mubvunzo uyu muShona yakajeka uye yakapfava:\n\n"
        ),
        "ndebele": (
            "Ungumsizi wezempilo yomhlaza wesibeletho. "
            "Phendula lo mbuzo ngesiNdebele esicacile futhi esilula:\n\n"
        ),
        "chinyanja": (
            "Ndine mphungu wa thanzi la kansa ya chibereko. "
            "Yankhani funso ili mu Chinyanja momveka bwino komanso mwaulemu:\n\n"
        ),
        "lozi": (
            "Ki muthusi wa za kansa ya mulomo wa sibeleko. "
            "Alaba lipuzo le ka Silozi se si bonahala hande ni se si nolofetse:\n\n"
        ),
        "bemba": (
            "Niwe kafwa wabufya bumi pali cervical kansa. "
            "Yasuka ilipusho ilipusho ilyakonkapo bwino bwino, mukwanguka, nechishinka Mucibemba:\n\n"
        ),
        "tonga": (
            "Ndimwi mweenzinyina wa bupilo kujatikizya kansa ya mulomo wa cibeleko. "
            "Mupandule mubvunzo ooyu mu Chitonga cilimvwisika alimwi cipepu:\n\n"
        ),
    }.get(lang, (
        "You are a cervical cancer health assistant. "
        "Answer the following question clearly and simply in English:\n\n"
    ))

    prompt = f"{instruction_body}{context}Current question: {question}\n\n{lang_enforce}"

    try:
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = gemini_model.generate_content(prompt)
        try:
            text = response.text
            if text and text.strip():
                return text.strip()
        except (ValueError, AttributeError) as ve:
            logging.warning(f"[ask_gemini_cancer] Blocked/empty lang={lang}: {ve}")
        return fallback
    except Exception as e:
        logging.error(f"[ask_gemini_cancer Error] {type(e).__name__}: {e}")
        return fallback



def ask_gemini_general(question: str, lang: str, sender: str = None) -> str:
    lang_enforce = _get_lang_enforce(lang)
    fallback = _get_fallback(lang)
    context = build_conversation_context(sender) if sender else ""

    company_address = "No. 50 Lunsemfwa Rd, Kalundu, Lusaka, Zambia"
    company_email   = "hello@dawa-health.com"
    company_website = "https://dawa-health.com/"
    company_phone   = "+260 571 376 677"

    instruction_body = {
        "shona": (
            "Uri mubatsiri wezvehutano ane hunyanzvi muhutano hwevakadzi vane pamuviri uye gomarara remuromo wechibereko. "
            "Pindura mubvunzo wemushandisi uchishandisa ruzivo rwechokwadi. "
            "USATANGE nemitsara yakaita sekuti Zvakanaka, Hongu, Hezvino. "
            "Tanga zvakananga nemhinduro. "
            "Pedzisa nekuyambira kupfupi kunoti ruzivo urwu harutsivi kuongororwa nachiremba.\n\n"
        ),
        "ndebele": (
            "Ungumsizi wezempilo ochwepheshile ogxile kwezempilo yabomama abakhulelweyo kanye lomdlavuza womlomo wesibeletho. "
            "Phendula umbuzo womsebenzisi usebenzisa ulwazi lwezempilo oluqondileyo. "
            "UNGAKALI ngemisho efana lokuthi Kulungile, Yebo, Nakhu. "
            "Qalisa masinyane ngempendulo uqobo. "
            "Qedisa ngesexwayiso esifitshane esithi ulwazi lolu aluthathi indawo yokuhlolwa ngudokotela.\n\n"
        ),
        "chinyanja": (
            "Ndinu mthandizi wa zaumoyo wa akatswiri pa zaumoyo wa amayi apakati komanso khansa ya chiberekero. "
            "Yankhani funso la wogwiritsa ntchito pogwiritsa ntchito chidziwitso cholondola. "
            "MUSAYAMBE ndi mawu ngati Chabwino, Inde, Nazi. "
            "Yambani mwachindunji ndi yankho. "
            "Malizitsani ndi chenjezo chachidule chonena kuti chidziwitsochi sichimalowa m'malo mwa kuyezetsa kwa dokotala.\n\n"
        ),
        "tonga": (
            "Muli mweenzinyina wa bumi uzyiba zya bumi bwa bakaintu bali ada alimwi a kansa ya mulomo wa cibeleko wa Dawa Health. "
            "Mupandule mubbuzyo wa muntu uyu mukubelesya mulaga ulungeme. "
            "Mutatali amazwi aakuti Kabotu, Inde, naa Leka ndikonzyesye. "
            "Talikila kubelesya kumupandulo."
            "Kuti babuzya zyakuyanda kuti ba Dawa Health babole kung'anda, mwaambile kuti ba Dawa Health clinicians balaya kung'anda. "
            f"Contact: email={company_email}, phone={company_phone}, address={company_address}, website={company_website}. "
            "Manizya akubasinsimuna kufwaafwi kuti ezi zyondamwabila tazichilili dokotala zyo amba\n\n"
        ),
        "bemba": (
            "Uli kapyunga wa buumi uwashintilila pa buumi bwa banakashi abali ne fumo pamo ne kansa ya cervix mu kampani ka Dawa Health. "
            "Asuka amepusho ukubonfya amasuko ayalingile, ifishininkisho fyapafyabumi."
            "Witampa ukwasuka na emukwayi, ehe nangu leka nondolole."
            "Asuka ukwabula ukupita mumbali nangula mukulungam."
            "Nga baipusha ukuti bushe ba dawa kuti baisa ku n'ganda?"
            "Pwisheni nokuti ifyebo namyeba tafilefuma kuli dokota nagula ukupyana dokota.\n\n"
        ),
        "lozi": (
            "Mu muthusi wa za mapilo wa bucwani ya iketile hahulu ku mapilo a basali baimana ni kankere ya mulomo wa sibeleko. "
            "Qalisa hanghang ka karabo. "
            "Felelisa ka temoso ye nyinyani ye e re ziboho ze ha zi nkeleli sibaka sa ku lekolwa ki dokota.\n\n"
        ),
    }.get(lang, (
        "You are a professional health assistant specializing in maternal health, sexual reproductive health and cervical cancer for Dawa Health. "
        "Answer the user question using correct and evidence-based health information. "
        "DO NOT start with phrases like Okay, Sure, or Let me explain. "
        "Start directly with the answer. "
        "Include a brief disclaimer at the end stating that this information does not replace a doctor evaluation. "
        "If asked about home visits, Dawa Health clinicians do home visits. "
        f"Contact: email={company_email}, phone={company_phone}, address={company_address}, website={company_website}.\n\n"
    ))

    prompt = f"{instruction_body}{context}Current question: {question}\n\n{lang_enforce}"

    try:
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        response = gemini_model.generate_content(prompt)
        try:
            text = response.text
            if text and text.strip():
                return text.strip()
        except (ValueError, AttributeError) as ve:
            logging.info(f"[LANG DEBUG] sender={sender} | "f"prompt={prompt!r} | "f"detected_lang={lang}")
            logging.warning(f"[ask_gemini_general] Blocked/empty lang={lang}: {ve}")
            try:
                finish = response.candidates[0].finish_reason
                logging.warning(f"[ask_gemini_general] Finish reason: {finish}")
            except Exception:
                pass
        return fallback
    except Exception as e:
        logging.error(f"[ask_gemini_general Error] {type(e).__name__}: {e}")
        return fallback


    base_prompt = f"{lang_enforce}\n\n{instruction_body}{context}Current question: {question}\n\n{lang_enforce}"

    def _generate(prompt_text, temperature):
        gen_config = dict(generation_config)
        gen_config["temperature"] = temperature
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=gen_config,
            safety_settings=safety_settings
        )
        response = gemini_model.generate_content(prompt_text)
        try:
            text = response.text
            return text.strip() if text and text.strip() else None
        except (ValueError, AttributeError) as ve:
            logging.warning(f"[ask_gemini_general] Blocked/empty lang={lang}: {ve}")
            return None

    try:
        text = _generate(base_prompt, temperature=0.5)
        if text is None:
            return fallback

        if _looks_like_english_leak(text, lang):
            logging.warning(f"[ask_gemini_general] English leak detected for lang={lang}, retrying")
            strict_prompt = (
                f"CRITICAL: Your entire reply must be written only in {lang.capitalize()}. "
                f"Do not include any English sentences or phrases, even if the question contains English words.\n\n"
                f"{base_prompt}"
            )
            retry_text = _generate(strict_prompt, temperature=0.3)
            if retry_text and not _looks_like_english_leak(retry_text, lang):
                return retry_text
            return text

        return text
    except Exception as e:
        logging.error(f"[ask_gemini_general Error] {type(e).__name__}: {e}")
        return fallback


def handle_ask_week(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
   
    try:
        week_num = int(re.sub(r"\D", "", prompt))
        if 1 <= week_num <= 40:
            pregnancy_info = get_pregnancy_data(lang)
            info_text = str(pregnancy_info)
           
            logging.info(f"Searching for week {week_num} in {lang} data")
           
            if lang == "ndebele":
                week_found = False
                lines = info_text.split('\n')
                response_lines = []
               
                ndebele_week_names = {
                    1: "yokuqala", 2: "yesibili", 3: "yesithathu", 4: "yesine",
                    5: "yesihlanu", 6: "yesithupha", 7: "yesikhombisa", 8: "yesishiyagalombili",
                    9: "yesishiyagalolunye", 10: "yetshumi", 11: "yetshumi nanye", 12: "yetshumi nambili",
                    13: "yetshumi nantathu", 14: "yetshumi nane", 15: "yetshumi nanhlanu",
                    16: "yetshumi nesithupha", 17: "yetshumi nesikhombisa", 18: "yetshumi nesishiyagalombili",
                    19: "yetshumi nesishiyagalolunye", 20: "lamashumi amabili", 21: "lamashumi amabili nesinye",
                    22: "lamashumi amabili nambili", 23: "lamashumi amabili nantathu", 24: "lamashumi amabili nane",
                    25: "lamashumi amabili nanhlanu", 26: "lamashumi amabili nesithupha", 27: "lamashumi amabili nesikhombisa",
                    28: "lamashumi amabili nesishiyagalombili", 29: "lamashumi amabili nesishiyagalolunye", 30: "lamashumi amathathu",
                    31: "lamashumi amathathu nesinye", 32: "lamashumi amathathu nambili", 33: "lamashumi amathathu nantathu",
                    34: "lamashumi amathathu nane", 35: "lamashumi amathathu nanhlanu", 36: "lamashumi amathathu nesithupha",
                    37: "lamashumi amathathu nesikhombisa", 38: "lamashumi amathathu nesishiyagalombili",
                    39: "lamashumi amathathu nesishiyagalolunye", 40: "lamashumi amane"
                }
               
                week_name = ndebele_week_names.get(week_num, f"#{week_num}")
               
                patterns = [
                    f"*Iviki {week_name}:",
                    f"*Iviki {week_name} :",
                    f"*Iviki {week_name}*",
                    f"*Iviki {week_name} (",
                    f"*Iviki {week_name}(",
                    f"*Iviki {week_name} :*"
                ]
               
                found_start = False
                for i, line in enumerate(lines):
                    if not found_start:
                        for pattern in patterns:
                            if pattern in line:
                                found_start = True
                                response_lines.append(line)
                                break
                    else:
                        if i + 1 < len(lines) and any(
                            f"*Iviki {ndebele_week_names.get(w, '')}" in lines[i + 1]
                            for w in range(week_num + 1, min(week_num + 5, 41))
                        ):
                            break
                        elif line.strip().startswith('*Iviki ') and week_name not in line:
                            break
                        elif line.strip() and not line.strip().startswith('***'):
                            response_lines.append(line)
               
                if response_lines:
                    week_info = '\n'.join(response_lines)
                    week_info = re.sub(r'\*Iviki [^*]*$', '', week_info)
                    send(f"Ulwazi lwe *Iviki {week_num}:*\n\n{week_info}", sender, phone_id)
                    week_found = True
                    ask_another_week(sender, phone_id)
               
                if not week_found:
                    week_patterns = [f"Week {week_num}", f"({week_num})", f": {week_num}:"]
                    for pattern in week_patterns:
                        if pattern in info_text:
                            start_idx = info_text.find(pattern)
                            end_idx = min(start_idx + 2000, len(info_text))
                            next_week = min([info_text.find(f"Week {w}", start_idx + 10) for w in range(week_num + 1, 41) if info_text.find(f"Week {w}", start_idx + 10) != -1] or [end_idx])
                            section = info_text[start_idx:next_week].strip()
                            send(f"Ulwazi lwe Iviki {week_num}:\n\n{section}", sender, phone_id)
                            week_found = True
                            ask_another_week(sender, phone_id)
                            break
               
                if not week_found:
                    send(f"Uxolo, angikutholanga ulwazi lweviki {week_num}. Zama elinye iviki kusuka ku-1 kuya ku-40.", sender, phone_id)
           
            elif lang == "chinyanja":
                pattern = rf"\*Sabata {week_num}:.*?(?=\*Sabata {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Zambiri za *Sabata {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Pepani, sindinapeze zambiri za sabata {week_num}. Yesani sabata lina kuchokera pa 1 mpaka 40.", sender, phone_id)
           
            elif lang == "shona":
                pattern = rf"\*Vhiki {week_num}:.*?(?=\*Vhiki {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Ruzivo rwe *Vhiki {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ndine urombo, handina kuwana ruzivo rwevhiki {week_num}. Edza imwe vhiki kubva pa 1 kusvika pa 40.", sender, phone_id)

            elif lang == "lozi":
                pattern = rf"\*Sunda {week_num}:.*?(?=\*Sunda {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Liziba la *Sunda {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ni maswabi, ha ni a fumana liziba la vhiki {week_num}. Linge sunda ye n'wi ku zwana 1 ku ya ku 40.", sender, phone_id)

            elif lang == "bemba":
                pattern = rf"\*Umulungu {week_num}:.*?(?=\*Umulungu {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Ifingi pa *Mulungu {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Natapa, nshasangile ifingi pa uyu mulungu {week_num}. Esheni mulungu ubi ukufuma pa 1 ukufika ku 40.", sender, phone_id)

            elif lang == "tonga":
                pattern = rf"\*Nhwiiiki {week_num}:.*?(?=\*Nhwiiiki {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Cibeela ca *Nhwiiiki {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Ndatola, tana kuwana cibeela ca nhwiiiki {week_num}. Lingenya vhiki linzwi kuzwa 1 kusika 40.", sender, phone_id)
           
            else:
                pattern = rf"\*Week {week_num}:.*?(?=\*Week {week_num+1}:|\*Question|\Z)"
                match = re.search(pattern, info_text, re.S | re.I)
                if match:
                    send(f"Here's information for *Week {week_num}:*\n\n{match.group(0).strip()}", sender, phone_id)
                    ask_another_week(sender, phone_id)
                else:
                    send(f"Sorry, I couldn't find information for week {week_num}. Please try another week from 1 to 40.", sender, phone_id)
       
        else:
            range_map = {
                "shona": "Ndapota isa vhiki kubva pa 1 kusvika pa 40 chete.",
                "ndebele": "Sicela ufake iviki eliphakathi kuka-1 no-40 kuphela.",
                "bemba": "Chisuma, ingisha mulungu ukufuma pa 1 ukufika pa 40 fye.",
                "chinyanja": "Chonde lowetsani sabata kuyambira pa 1 mpaka pa 40 basi.",
                "tonga": "Ndakomba, mulembesye week kuzwa pa 1 kusika pa 40 buyo.",
                "lozi": "Ndapota, kenisa vhiki ku zwana 1 ku ya ku 40 feela.",
            }
            send(range_map.get(lang, "Please enter a week between 1 and 40 only."), sender, phone_id)
           
    except ValueError:
        range_map = {
            "shona": "Ndapota isa vhiki kubva pa 1 kusvika pa 40 chete.",
            "ndebele": "Sicela ufake iviki eliphakathi kuka-1 no-40 kuphela.",
            "bemba": "Napapata, ingisha mulungu ukufuma pa 1 ukufika pa 40 fye.",
            "chinyanja": "Chonde lowetsani sabata kuyambira pa 1 mpaka pa 40 basi.",
            "tonga": "Ndakomba, mulembesye week kuzwa pa 1 kusika pa 40 buyo.",
            "lozi": "Ndapota, kenisa vhiki ku zwana 1 ku ya ku 40 feela.",
        }
        send(range_map.get(lang, "Please enter a week between 1 and 40 only."), sender, phone_id)
           

@app.route("/agent-timeout", methods=["POST"])
def agent_timeout():
    """
    Called by a scheduled job (e.g. Upstash QStash) after AGENT_TIMEOUT_SECONDS.
    Body JSON: { "user_number": "+260...", "phone_id": "..." }
    If the request is still pending, it fires 'no agents available'.
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
                request_data = json.loads(raw)
                if request_data.get("status") == "pending":
                    ensure_user_state(user_number)
                    _handle_no_agents_available(user_number, current_phone_id)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logging.error(f"Error in agent_timeout endpoint: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return render_template("connected.html")

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == "BOT":
            return challenge, 200
        else:
            return "Failed", 403

    elif request.method == "POST":
        try:
            data = request.get_json()
            logging.info(f"Received webhook data: {data}")
           
            if "entry" in data:
                for entry in data["entry"]:
                    if "changes" in entry:
                        for change in entry["changes"]:
                            if "value" in change:
                                value = change["value"]
                               
                                if "messages" in value:
                                    for message in value["messages"]:
                                        sender = message["from"]
                                        phone_id = value["metadata"]["phone_number_id"]

                                        if message.get("type") == "interactive":
                                            interactive = message.get("interactive", {})
                                            if interactive.get("type") == "button_reply":
                                                btn_id = interactive["button_reply"]["id"]
                                                logging.info(f"Button reply from {sender}: {btn_id}")
                                                if btn_id.startswith("agent_accept:"):
                                                    user_num = btn_id.split("agent_accept:", 1)[1]
                                                    handle_agent_accept(sender, user_num, phone_id)
                                                elif btn_id.startswith("agent_reject:"):
                                                    user_num = btn_id.split("agent_reject:", 1)[1]
                                                    handle_agent_reject(sender, user_num, phone_id)
                                            continue

                                        if "text" in message:
                                            prompt = message["text"]["body"]
                                            logging.info(f"Processing message from {sender}: {prompt}")

                                            _agent_phones_norm = {normalize_phone(p) for p in AGENTS.values()}
                                            if normalize_phone(sender) in _agent_phones_norm:
                                                relayed = relay_agent_message_to_user(sender, prompt, phone_id)
                                                if relayed:
                                                    continue

                                            is_new = ensure_user_state(sender)
                                            if not is_new:
                                                user_states[sender]["first_message"] = False
                                            
                                            save_user_conversation(sender, "user", prompt)
                                            
                                            referral = extract_referral_source(prompt)
                                            if referral:
                                                save_referral_source(sender, referral)
                                                logging.info(f"Referral detected for {sender}: {referral}")
                                            
                                            handle_conversation_state(sender, prompt, phone_id)

                                        else:
                                            logging.info(f"Non-text message received from {sender}")
                                            ensure_user_state(sender)
                                            state = user_states.get(sender, {})
                                            lang = state.get("language", "english")
                                            non_text_map = {
                                                "shona": "Ndine urombo, handigoni kugamuchira mameseji asiri mavara chete. Ndapota tumira meseji yemavara.",
                                                "ndebele": "Uxolo, angikwazi ukwamukela imilayezo engeyona imibhalo kuphela. Sicela uthumele umlayezo wombhalo.",
                                                "bemba": "Njelelako, Dekwanisha fye ukumona ama text meseji. Napapata, lemba meseji.",
                                                "chinyanja": "Pepani, sindingathe kulandira mameseji enama osati a zilembo. Chonde tumirani meseji ya zilembo.",
                                                "tonga": "Ndausa, tandikonzya kubelesya mameseji aali kunze a mabbala. Ndalomba tumizya meseji ya mabbala.",
                                                "lozi": "Ni maswabi, ha na kona kuzwela miiala yeng'wi kufita feela ya mangolo. Ndapota, lumeza molaala wa mangolo.",
                                            }
                                            send(non_text_map.get(lang, "Sorry, I can only process text messages. Please send a text message."), sender, phone_id)
                               
                                elif "statuses" in value:
                                    logging.info("Message status update received, ignoring.")
                               
                                else:
                                    logging.info("Webhook received non-message event, ignoring.")

        except Exception as e:
            logging.error(f"Error in webhook: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500
       
        return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    load_user_states()
    app.run(host="0.0.0.0", port=5000, debug=True)
