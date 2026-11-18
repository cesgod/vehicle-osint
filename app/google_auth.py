"""Google OAuth 2.0 sign-in (Advanced Mode) and authorized service factories.

Sign in once; the refresh token is stored in data/google_token.json and reused, so
Cloud Vision, Drive and Sheets work in every later session without pasting keys.

Scopes are deliberately narrow:
  * cloud-vision  -- Vision API only (not all of Google Cloud)
  * drive.file    -- only files this app creates; it can never read the rest of your Drive.
                     This also covers spreadsheets the app creates through the Sheets API.

This is a single-user, run-on-your-own-machine design: there is one token on disk.
Do not expose the server publicly while signed in.
"""
from __future__ import annotations

import json
import os

from . import config

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-vision",
    "https://www.googleapis.com/auth/drive.file",
]

# Google may return scopes in a different order / with aliases; don't treat that as an error.
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

_pending_flows: dict[str, object] = {}  # state -> Flow (keeps the PKCE verifier between redirect and callback)


class AuthError(Exception):
    pass


def client_configured() -> bool:
    return config.CLIENT_SECRET_PATH.exists()


def save_client_secret(raw: bytes) -> str:
    """Validate and store an uploaded client_secret.json. Returns the client type."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthError("That file is not valid JSON.") from exc
    kind = "installed" if "installed" in data else "web" if "web" in data else None
    if not kind or "client_id" not in data[kind] or "client_secret" not in data[kind]:
        raise AuthError('Not an OAuth client file. Download it from Google Cloud Console > Credentials > '
                        '"OAuth client ID" (Desktop app or Web application).')
    config.CLIENT_SECRET_PATH.write_bytes(raw)
    return kind


def start_login(redirect_uri: str) -> str:
    from google_auth_oauthlib.flow import Flow

    if not client_configured():
        raise AuthError("client_secret.json is missing. Upload it in the Settings tab.")
    if redirect_uri.startswith(("http://localhost", "http://127.0.0.1")):
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # loopback redirect is allowed over http
    flow = Flow.from_client_secrets_file(str(config.CLIENT_SECRET_PATH), scopes=SCOPES, redirect_uri=redirect_uri)
    url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
    _pending_flows.clear()
    _pending_flows[state] = flow
    return url


def finish_login(state: str, authorization_response: str) -> str:
    flow = _pending_flows.pop(state, None)
    if flow is None:
        raise AuthError("Sign-in session expired. Start again from the app.")
    try:
        flow.fetch_token(authorization_response=authorization_response)
    except Exception as exc:  # oauthlib raises many types
        raise AuthError(f"Google sign-in failed: {exc}") from exc
    creds = flow.credentials
    email = ""
    try:
        from googleapiclient.discovery import build
        email = build("oauth2", "v2", credentials=creds, cache_discovery=False).userinfo().get().execute().get("email", "")
    except Exception:
        pass
    payload = json.loads(creds.to_json())
    payload["email"] = email
    config.TOKEN_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return email


def get_credentials():
    """Stored user credentials, refreshed if expired. None when signed out."""
    if not config.TOKEN_PATH.exists():
        return None
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    try:
        payload = json.loads(config.TOKEN_PATH.read_text(encoding="utf-8"))
        creds = Credentials.from_authorized_user_info(payload)
    except (ValueError, KeyError):
        return None
    if not creds.valid:
        if not creds.refresh_token:
            return None
        try:
            creds.refresh(Request())
        except RefreshError:
            sign_out()
            return None
        fresh = json.loads(creds.to_json())
        fresh["email"] = payload.get("email", "")
        config.TOKEN_PATH.write_text(json.dumps(fresh), encoding="utf-8")
    return creds


def signed_in_email() -> str | None:
    """Email of the signed-in account, or None. Does not hit the network."""
    if not config.TOKEN_PATH.exists():
        return None
    try:
        payload = json.loads(config.TOKEN_PATH.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return payload.get("email") or "Google account" if payload.get("refresh_token") else None


def sign_out() -> None:
    config.TOKEN_PATH.unlink(missing_ok=True)


def require_credentials():
    creds = get_credentials()
    if creds is None:
        raise AuthError("Sign in with Google to use this feature (Advanced mode).")
    return creds


def sheets_service():
    from googleapiclient.discovery import build
    return build("sheets", "v4", credentials=require_credentials(), cache_discovery=False)


def drive_service():
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=require_credentials(), cache_discovery=False)


def vision_client():
    from google.cloud import vision
    return vision.ImageAnnotatorClient(credentials=require_credentials(), transport="rest")
