"""
auth_utils.py — Supabase Auth helpers for Underdawg.

Auth state is stored in st.session_state (never on disk).
Browser cookies (via extra-streamlit-components) hold the Supabase
access_token + refresh_token so that auth survives hard browser reloads.
Passwords are never stored.
"""
import streamlit as st
from datetime import datetime, timedelta, timezone


# ── Secrets check ─────────────────────────────────────────────────────────────

def secrets_configured() -> bool:
    """Return True if [supabase] url and anon_key are present in Streamlit secrets."""
    try:
        _ = st.secrets["supabase"]["url"]
        _ = st.secrets["supabase"]["anon_key"]
        return True
    except (KeyError, FileNotFoundError, AttributeError):
        return False


# ── Supabase client singleton ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _build_client():
    """
    Create and cache the Supabase client for the lifetime of the Streamlit process.
    Returns None if the package is missing or secrets are not configured.
    """
    try:
        from supabase import create_client  # type: ignore
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]
        return create_client(url, key)
    except ImportError:
        return None
    except (KeyError, FileNotFoundError, AttributeError):
        return None
    except Exception:
        return None


def get_supabase_client():
    """Return the cached Supabase client, or None if unavailable."""
    return _build_client()


# ── Session state accessors ───────────────────────────────────────────────────

def get_current_user() -> dict | None:
    """Return {id, email} for the signed-in user, or None."""
    return st.session_state.get("supabase_user")


def is_authenticated() -> bool:
    """True if a user is currently signed in."""
    return st.session_state.get("supabase_user") is not None


def require_auth() -> dict:
    """
    Return the current user dict if authenticated.
    Raises RuntimeError if not authenticated — callers should check is_authenticated() first
    and redirect to the login page rather than letting this raise.
    """
    user = get_current_user()
    if user is None:
        raise RuntimeError("Not authenticated.")
    return user


# ── Auth operations ───────────────────────────────────────────────────────────

def sign_in(email: str, password: str) -> dict:
    """
    Sign in with email + password via Supabase Auth.

    Returns:
        {"ok": True,  "user": {id, email}, "session": {access_token, refresh_token}}
        {"ok": False, "error": "<human-readable message>"}
    """
    client = _build_client()
    if client is None:
        return {"ok": False, "error": _client_unavailable_msg()}
    try:
        resp = client.auth.sign_in_with_password(
            {"email": email.strip(), "password": password}
        )
        user    = getattr(resp, "user", None)
        session = getattr(resp, "session", None)
        if user is None:
            return {"ok": False, "error": "Sign-in failed. Check your email and password."}
        return {
            "ok":      True,
            "user":    _user_to_dict(user),
            "session": _session_to_dict(session),
        }
    except Exception as exc:
        return {"ok": False, "error": _friendly_error(str(exc))}


def sign_up(email: str, password: str) -> dict:
    """
    Create a new account via Supabase Auth.

    Returns:
        {"ok": True,  "needs_confirmation": bool, "user": {...}, "session": {...} | None}
        {"ok": False, "error": "<human-readable message>"}

    If needs_confirmation is True, the user must click the confirmation link in
    their email before they can sign in. Session will be None in that case.
    """
    client = _build_client()
    if client is None:
        return {"ok": False, "error": _client_unavailable_msg()}
    try:
        resp = client.auth.sign_up(
            {"email": email.strip(), "password": password}
        )
        user    = getattr(resp, "user", None)
        session = getattr(resp, "session", None)
        if user is None:
            return {"ok": False, "error": "Sign-up failed. Please try again."}

        # If identities list is empty, this email already has an account.
        identities = getattr(user, "identities", None) or []
        if len(identities) == 0:
            return {
                "ok":    False,
                "error": "An account with this email already exists. Please sign in instead.",
            }

        # session is None when email confirmation is required
        needs_confirmation = session is None
        return {
            "ok":                 True,
            "needs_confirmation": needs_confirmation,
            "user":               _user_to_dict(user),
            "session":            _session_to_dict(session) if session else None,
        }
    except Exception as exc:
        return {"ok": False, "error": _friendly_error(str(exc))}


def sign_out() -> None:
    """
    Sign out of Supabase and clear auth session state.
    Safe to call even when not authenticated.
    """
    client = _build_client()
    if client is not None:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    st.session_state.pop("supabase_user",    None)
    st.session_state.pop("supabase_session", None)


# ── Cookie-based token persistence ───────────────────────────────────────────
# `cm` is an extra_streamlit_components.CookieManager instance (or None if the
# package is not installed).  All three functions are no-ops when cm is None.

def save_auth_cookies(
    cm,
    access_token: str,
    refresh_token: str,
    days: int = 7,
    *,
    debug: bool = False,
) -> None:
    """Store Supabase tokens in browser cookies so auth survives page reloads.

    Must be called during a normal Streamlit render cycle — NOT immediately
    before st.rerun(), because st.rerun() can interrupt the CookieManager
    component's JS before it writes the cookie.  Use the _pending_cookie_save
    session-state flag pattern (see app.py) to defer this call to the next render.
    """
    if cm is None or not access_token or not refresh_token:
        if debug:
            print(
                f"[auth] save_auth_cookies: skipped "
                f"(cm={cm is not None} at={bool(access_token)} rt={bool(refresh_token)})"
            )
        return
    if debug:
        print(
            f"[auth] save_auth_cookies: writing "
            f"ud_access_token (len={len(access_token)}) "
            f"ud_refresh_token (len={len(refresh_token)})"
        )
    try:
        expiry = datetime.now(tz=timezone.utc) + timedelta(days=days)
        cm.set("ud_access_token",  access_token,  expires_at=expiry, key="save_at")
        cm.set("ud_refresh_token", refresh_token, expires_at=expiry, key="save_rt")
        if debug:
            print("[auth] save_auth_cookies: cm.set() calls completed")
    except Exception as exc:
        if debug:
            print(f"[auth] save_auth_cookies: FAILED — {type(exc).__name__}: {exc}")


def clear_auth_cookies(cm, *, debug: bool = False) -> None:
    """Remove auth cookies on sign-out."""
    if cm is None:
        return
    if debug:
        print("[auth] clear_auth_cookies: deleting ud_access_token ud_refresh_token")
    try:
        cm.delete("ud_access_token",  key="del_at")
        cm.delete("ud_refresh_token", key="del_rt")
    except Exception as exc:
        if debug:
            print(f"[auth] clear_auth_cookies: FAILED — {type(exc).__name__}: {exc}")


def restore_session_if_possible(cm, *, debug: bool = False) -> bool:
    """
    Read Supabase tokens from cookies and restore the session into session_state.

    Called on every unauthenticated render.  Returns True if the session was
    restored.  Safe to call even when cookies are absent or tokens are stale.
    When debug=True, status lines are printed to stdout (never to the Streamlit UI
    and never with token values).
    """
    if is_authenticated():
        return True
    if cm is None:
        return False
    try:
        if debug:
            try:
                _keys = list((getattr(cm, "cookies", None) or {}).keys())
                print(f"[auth] restore: cookie keys = {_keys}")
            except Exception:
                print("[auth] restore: could not list cookie keys")
        access_token  = str(cm.get("ud_access_token")  or "")
        refresh_token = str(cm.get("ud_refresh_token") or "")
        if not access_token or not refresh_token:
            if debug:
                print(
                    f"[auth] restore: no tokens "
                    f"(at={bool(access_token)} rt={bool(refresh_token)})"
                )
            return False
        if debug:
            print(
                f"[auth] restore: token lengths "
                f"access={len(access_token)} refresh={len(refresh_token)}"
            )
        client = _build_client()
        if client is None:
            if debug:
                print("[auth] restore: supabase client unavailable")
            return False
        # set_session handles expired access tokens automatically by using the
        # refresh_token.  When it does, Supabase returns ROTATED tokens — we must
        # write them back to cookies so the next reload can restore again.
        resp    = client.auth.set_session(access_token, refresh_token)
        user    = getattr(resp, "user",    None)
        session = getattr(resp, "session", None)
        if user is None:
            if debug:
                print("[auth] restore: set_session returned no user")
            return False
        st.session_state["supabase_user"]    = _user_to_dict(user)
        st.session_state["supabase_session"] = _session_to_dict(session)
        # Persist rotated tokens so future reloads can restore correctly.
        if session:
            new_at = str(getattr(session, "access_token",  "") or "")
            new_rt = str(getattr(session, "refresh_token", "") or "")
            if new_at and new_rt and (new_at != access_token or new_rt != refresh_token):
                if debug:
                    print("[auth] restore: writing rotated tokens back to cookies")
                save_auth_cookies(cm, new_at, new_rt, debug=debug)
        if debug:
            print("[auth] restore: succeeded")
        return True
    except Exception as exc:
        if debug:
            import re as _re
            exc_type = type(exc).__name__
            # Scrub any long alphanum sequences that might be token fragments.
            exc_msg  = _re.sub(r"[A-Za-z0-9_\-]{40,}", "[token]", str(exc))
            print(f"[auth] restore: FAILED — {exc_type}: {exc_msg}")
        return False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _user_to_dict(user) -> dict:
    if user is None:
        return {}
    return {
        "id":    str(getattr(user, "id",    "")),
        "email": str(getattr(user, "email", "")),
    }


def _session_to_dict(session) -> dict | None:
    if session is None:
        return None
    return {
        "access_token":  str(getattr(session, "access_token",  "")),
        "refresh_token": str(getattr(session, "refresh_token", "")),
    }


def _client_unavailable_msg() -> str:
    if not secrets_configured():
        return (
            "Supabase is not configured. "
            "Add [supabase] url and anon_key to your Streamlit secrets."
        )
    return (
        "The supabase package is not installed. "
        "Run: pip install supabase"
    )


def _friendly_error(raw: str) -> str:
    """Convert Supabase/GoTrue error strings to human-readable messages."""
    low = raw.lower()
    if "invalid login credentials" in low or "invalid_credentials" in low:
        return "Incorrect email or password. Please try again."
    if "email not confirmed" in low:
        return (
            "Your email address has not been confirmed. "
            "Check your inbox and click the confirmation link, then sign in."
        )
    if "user already registered" in low or "already registered" in low:
        return "An account with this email already exists. Please sign in instead."
    if "password should be" in low or "password is too short" in low or "at least 6" in low:
        return "Password must be at least 6 characters."
    if "rate limit" in low or "too many requests" in low or "over_email_send_rate_limit" in low:
        return "Too many attempts. Please wait a minute and try again."
    if "network" in low or "connection" in low or "unable to connect" in low:
        return "Network error. Check your internet connection and try again."
    if "invalid email" in low or "unable to validate email" in low:
        return "Please enter a valid email address."
    # Return first non-empty line, capped at 200 chars, stripping any stack trace
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return "An unexpected error occurred. Please try again."
