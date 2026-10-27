"""Where a connection to the provider comes from (design §7).

- Standalone: the harness reads the provider key from its environment ("your keys, your infrastructure").
- Behind a HarnessRouter server: the harness never holds a provider key. For each connection it asks the
  server for a short-lived credential scoped to one call; the server holds the key and meters the call.

Either way the result is the same thing: a WebSocket URL on the OpenAI realtime wire and the headers to
open it with. Measured on Azure OpenAI (design §1.7): a client secret minted by the server opens the
WebSocket as a bearer token, not only WebRTC.
"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass

import httpx


class CredentialError(RuntimeError):
    """The provider connection could not be prepared. The message names the cause, never a secret."""


@dataclass
class Connection:
    url: str
    headers: dict[str, str]
    model: str                      # what the provider runs (a deployment name on Azure)
    ref: dict                       # ids the server gave the connection (for ending it), never a secret


class Credentials:
    async def connect(self, model: str) -> Connection:  # pragma: no cover - interface
        raise NotImplementedError

    async def release(self, conn: Connection) -> None:
        """Tell the server a connection is over (it stops the meter). Nothing to do standalone."""
        return None


class AzureKey(Credentials):
    """Azure OpenAI, the resource key from the environment: AZURE_OPENAI_ENDPOINT (https://<resource>.openai.azure.com)
    and AZURE_OPENAI_API_KEY. The model id names the deployment unless RTH_AZURE_DEPLOYMENT maps it."""

    def __init__(self, endpoint: str | None = None, key: str | None = None):
        self.endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
        self._key = key or os.environ.get("AZURE_OPENAI_API_KEY") or ""
        if not self.endpoint or not self._key:
            raise CredentialError("Azure OpenAI needs AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        host = urllib.parse.urlsplit(self.endpoint).hostname or ""
        if not host:
            raise CredentialError("AZURE_OPENAI_ENDPOINT is not a URL")
        self.host = host

    async def connect(self, model: str) -> Connection:
        deployment = os.environ.get("RTH_AZURE_DEPLOYMENT") or model
        url = f"wss://{self.host}/openai/v1/realtime?model={urllib.parse.quote(deployment)}"
        return Connection(url=url, headers={"api-key": self._key}, model=deployment, ref={})


class OpenAIKey(Credentials):
    """OpenAI, OPENAI_API_KEY from the environment."""

    def __init__(self, key: str | None = None):
        self._key = key or os.environ.get("OPENAI_API_KEY") or ""
        if not self._key:
            raise CredentialError("OpenAI needs OPENAI_API_KEY")

    async def connect(self, model: str) -> Connection:
        url = f"wss://api.openai.com/v1/realtime?model={urllib.parse.quote(model)}"
        return Connection(url=url, headers={"Authorization": f"Bearer {self._key}"}, model=model, ref={})


class HarnessRouterMint(Credentials):
    """A HarnessRouter server's realtime mint: POST {base}/v1/harnesses/{harness}/realtime/sessions with the
    server's own token returns a client secret for one call on the server's provider connection, billed to
    the caller's organization. The secret is used once, to open this connection, and never stored or logged.

    RTH_SERVER_URL (the server's API base, e.g. https://app.example/api/harness), RTH_SERVER_HARNESS (the
    server's harness the calls run under) and RTH_SERVER_TOKEN (or RTH_SERVER_TOKEN_FILE)."""

    def __init__(self, base: str | None = None, harness: str | None = None, token: str | None = None,
                 max_minutes: int = 30):
        self.base = (base or os.environ.get("RTH_SERVER_URL") or "").rstrip("/")
        self.harness = harness or os.environ.get("RTH_SERVER_HARNESS") or ""
        tok = token or os.environ.get("RTH_SERVER_TOKEN") or ""
        if not tok and os.environ.get("RTH_SERVER_TOKEN_FILE"):
            with open(os.environ["RTH_SERVER_TOKEN_FILE"], encoding="utf-8") as f:
                tok = f.read().strip()
        if not (self.base and self.harness and tok):
            raise CredentialError("a HarnessRouter server's mint needs RTH_SERVER_URL, RTH_SERVER_HARNESS and a token")
        self._token = tok
        self.max_minutes = max_minutes

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    async def connect(self, model: str) -> Connection:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(f"{self.base}/v1/harnesses/{urllib.parse.quote(self.harness)}/realtime/sessions",
                             headers=self._headers(), json={"max_minutes": self.max_minutes})
        if r.status_code != 201:
            try:
                err = (r.json() or {}).get("error") or {}
                why = f"{err.get('code') or ''} {err.get('message') or ''}".strip()
            except ValueError:
                why = ""
            raise CredentialError(f"the server did not mint a credential (HTTP {r.status_code}{', ' + why if why else ''})")
        d = r.json()
        host = urllib.parse.urlsplit(str(d.get("webrtc_url") or "")).hostname or ""
        secret = str(d.get("client_secret") or "")
        deployment = str(d.get("deployment") or "")
        if not (host and secret and deployment):
            raise CredentialError("the server's mint answered without a usable credential")
        url = f"wss://{host}/openai/v1/realtime?model={urllib.parse.quote(deployment)}"
        return Connection(url=url, headers={"Authorization": f"Bearer {secret}"}, model=deployment,
                          ref={"server_session": str(d.get("session_id") or "")})

    async def release(self, conn: Connection) -> None:
        sid = conn.ref.get("server_session")
        if not sid:
            return
        try:
            async with httpx.AsyncClient(timeout=20) as c:
                await c.post(f"{self.base}/v1/realtime/sessions/{urllib.parse.quote(sid)}/end", headers=self._headers())
        except httpx.HTTPError:
            pass        # the server's sweep meters a call nobody ended to its max_minutes


def from_env() -> Credentials:
    """The credentials this process was given: a server's mint, else Azure, else OpenAI."""
    if os.environ.get("RTH_SERVER_URL"):
        return HarnessRouterMint()
    if os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureKey()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIKey()
    raise CredentialError("no provider credentials: set RTH_SERVER_URL (a HarnessRouter server), "
                          "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or OPENAI_API_KEY")
