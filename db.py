

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_TIMEOUT = float(os.environ.get("SUPABASE_TIMEOUT", "8"))

_session = requests.Session()


def enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _headers(prefer=None):
    headers = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    # New sb_secret_ keys are not JWTs and must only travel in the apikey header.
    # The legacy service_role key is a JWT and may be sent as a bearer token as well.
    if SUPABASE_KEY.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {SUPABASE_KEY}"
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _now():
    return datetime.now(timezone.utc)


def _ts(dt=None):
    """Timestamp format that is safe inside PostgREST filters (no '+', no fractions)."""
    return (dt or _now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _call(method, table, params=None, body=None, prefer=None):
    """One PostgREST request. Returns parsed JSON (usually a list) or None. Raises on HTTP errors."""
    resp = _session.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{table}",
        params=params,
        json=body,
        headers=_headers(prefer),
        timeout=SUPABASE_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Supabase {method} {table} -> {resp.status_code}: {resp.text[:300]}")
    if resp.content:
        try:
            return resp.json()
        except ValueError:
            return None
    return None


# ─────────────────────────────────────────────
#  KEY / VALUE with expiry (agent sessions, de-duplication)
# ─────────────────────────────────────────────

class SupabaseKV:
    """Minimal stand-in for the Redis calls the bot uses: get, set(ex, nx), delete, ping."""

    def ping(self):
        _call("GET", "kv", {"select": "key", "limit": "1"})
        return True

    def get(self, key):
        rows = _call("GET", "kv", {
            "key": f"eq.{key}",
            "select": "value",
            "or": f"(expires_at.is.null,expires_at.gt.{_ts()})",
            "limit": "1",
        })
        return rows[0]["value"] if rows else None

    def set(self, key, value, ex=None, nx=False):
        row = {
            "key": key,
            "value": value if isinstance(value, str) else json.dumps(value),
            "expires_at": _ts(_now() + timedelta(seconds=ex)) if ex else None,
        }
        if nx:
            # An expired row must not block a fresh insert
            _call("DELETE", "kv", {"key": f"eq.{key}", "expires_at": f"lt.{_ts()}"})
            rows = _call("POST", "kv", {"on_conflict": "key"}, row,
                         "resolution=ignore-duplicates,return=representation")
            return True if rows else None
        _call("POST", "kv", {"on_conflict": "key"}, row, "resolution=merge-duplicates,return=minimal")
        return True

    def delete(self, key):
        _call("DELETE", "kv", {"key": f"eq.{key}"})
        return 1


kv = SupabaseKV()


# ─────────────────────────────────────────────
#  USERS
# ─────────────────────────────────────────────

def get_user_state(phone):
    if not enabled():
        return None
    try:
        rows = _call("GET", "users", {"phone": f"eq.{phone}", "select": "state", "limit": "1"})
        return rows[0]["state"] if rows else None
    except Exception as e:
        logging.error(f"[db] get_user_state failed for {phone}: {e}")
        return None


def save_user_state(phone, state):
    if not enabled():
        return
    try:
        _call("POST", "users", {"on_conflict": "phone"}, {
            "phone": phone,
            "user_id": state.get("user_id"),
            "language": state.get("language"),
            "step": state.get("step"),
            "state": state,
            "updated_at": _now().isoformat(),
        }, "resolution=merge-duplicates,return=minimal")
    except Exception as e:
        logging.error(f"[db] save_user_state failed for {phone}: {e}")


# ─────────────────────────────────────────────
#  MESSAGES (conversation history)
# ─────────────────────────────────────────────

def get_messages(phone, limit=30):
    """Newest `limit` messages, oldest first, shaped like the old Redis history entries."""
    if not enabled():
        return []
    try:
        rows = _call("GET", "messages", {
            "phone": f"eq.{phone}",
            "select": "id,role,message,message_en,created_at",
            "order": "id.desc",
            "limit": str(limit),
        }) or []
        rows.reverse()
        return [
            {
                "role": r["role"],
                "message": r["message"],
                "message_en": r.get("message_en"),
                "timestamp": r["created_at"],
            }
            for r in rows
        ]
    except Exception as e:
        logging.error(f"[db] get_messages failed for {phone}: {e}")
        return []


def add_message(phone, role, message, message_en=None):
    if not enabled():
        return
    try:
        _call("POST", "messages", None, {
            "phone": phone,
            "role": role,
            "message": message,
            "message_en": message_en,
        }, "return=minimal")
    except Exception as e:
        logging.error(f"[db] add_message failed for {phone}: {e}")


def set_last_message_en(phone, role, text_en):
    """Attach an English translation to the newest message with this role."""
    if not enabled() or not text_en:
        return
    try:
        rows = _call("GET", "messages", {
            "phone": f"eq.{phone}", "role": f"eq.{role}",
            "select": "id", "order": "id.desc", "limit": "1",
        })
        if rows:
            _call("PATCH", "messages", {"id": f"eq.{rows[0]['id']}"}, {"message_en": text_en}, "return=minimal")
    except Exception as e:
        logging.error(f"[db] set_last_message_en failed for {phone}: {e}")


def delete_messages(phone):
    if not enabled():
        return
    try:
        _call("DELETE", "messages", {"phone": f"eq.{phone}"})
    except Exception as e:
        logging.error(f"[db] delete_messages failed for {phone}: {e}")


def delete_last_message_if_role(phone, role):
    """Remove the newest message, but only if it has the given role."""
    if not enabled():
        return
    try:
        rows = _call("GET", "messages", {
            "phone": f"eq.{phone}", "select": "id,role", "order": "id.desc", "limit": "1",
        })
        if rows and rows[0]["role"] == role:
            _call("DELETE", "messages", {"id": f"eq.{rows[0]['id']}"})
    except Exception as e:
        logging.error(f"[db] delete_last_message_if_role failed for {phone}: {e}")


# ─────────────────────────────────────────────
#  REFERRALS AND ORDERS
# ─────────────────────────────────────────────

def add_referral(phone, source, user_id):
    if not enabled():
        logging.warning("[db] Supabase not configured, referral not saved")
        return
    try:
        _call("POST", "referrals", None,
              {"phone": phone, "source": source, "user_id": user_id}, "return=minimal")
    except Exception as e:
        logging.error(f"[db] add_referral failed: {e}")


def add_order(phone, user_id, product, price, quantity, address):
    if not enabled():
        logging.warning("[db] Supabase not configured, order not saved")
        return
    try:
        _call("POST", "orders", None, {
            "phone": phone,
            "user_id": user_id,
            "product": product,
            "price": str(price),
            "quantity": int(quantity),
            "address": address,
            "status": "pending",
        }, "return=minimal")
    except Exception as e:
        logging.error(f"[db] add_order failed: {e}")
