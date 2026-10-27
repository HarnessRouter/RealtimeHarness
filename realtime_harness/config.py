"""A configured harness: what a call runs with, decided by its owner and never by the caller.

A harness file (YAML) holds one harness or a list of them:

    id: support
    name: Store support
    instructions: You answer questions about the store.
    voice: marin
    tool_timeout_seconds: 10
    max_call_minutes: 20
    mcp_servers:
      - name: orders
        url: https://orders.example/mcp
        transport: http            # http (Streamable HTTP) or sse
        headers: {Authorization: "Bearer ${ORDERS_TOKEN}"}
        disabled_tools: [refund]
      - name: weather
        url: https://weather.example/mcp
        enabled: false             # never contacted
    disabled_tools: [delete_account]

`${NAME}` in a header value reads the environment when the file is loaded, so a secret is never
written in the file. Everything here holds for every call on the harness (design §2.1): the person on
the call can narrow a call (mute, cancel, hang up) and never widen it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import yaml

DEFAULT_TOOL_TIMEOUT_S = 10            # design §13 decision 7, from the Azure measurements in §1.7
DEFAULT_MAX_CALL_MINUTES = 30
DEFAULT_VOICE = "marin"
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    pass


def _env(value: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in os.environ:
            raise ConfigError(f"the environment has no {name} (named in a header)")
        return os.environ[name]
    return _ENV_REF.sub(sub, value)


@dataclass
class McpServer:
    name: str
    url: str
    transport: str = "http"
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    disabled_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "McpServer":
        name = str(d.get("name") or "").strip()
        url = str(d.get("url") or "").strip()
        if not name or not url:
            raise ConfigError("an MCP server needs a name and a url")
        transport = str(d.get("transport") or "http").lower()
        if transport not in ("http", "sse"):
            raise ConfigError(f"MCP server {name!r}: transport is http or sse, not {transport!r}")
        headers = {str(k): _env(str(v)) for k, v in (d.get("headers") or {}).items()}
        return cls(name=name, url=url, transport=transport, headers=headers,
                   enabled=bool(d.get("enabled", True)),
                   disabled_tools=[str(t) for t in d.get("disabled_tools") or d.get("disabledTools") or []])


@dataclass
class HarnessConfig:
    id: str
    name: str
    instructions: str = ""
    voice: str = DEFAULT_VOICE
    mcp_servers: list[McpServer] = field(default_factory=list)
    disabled_tools: list[str] = field(default_factory=list)
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_S
    max_call_minutes: float = DEFAULT_MAX_CALL_MINUTES
    default_model: str = ""
    # the provider's model (on Azure, the deployment) that transcribes the caller's speech for the record;
    # empty leaves the provider session's own setting
    transcription_model: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "HarnessConfig":
        hid = str(d.get("id") or "").strip()
        if not hid:
            raise ConfigError("a harness needs an id")
        timeout = float(d.get("tool_timeout_seconds") or DEFAULT_TOOL_TIMEOUT_S)
        minutes = float(d.get("max_call_minutes") or DEFAULT_MAX_CALL_MINUTES)
        if timeout <= 0 or minutes <= 0:
            raise ConfigError(f"harness {hid!r}: tool_timeout_seconds and max_call_minutes are positive")
        return cls(id=hid, name=str(d.get("name") or hid), instructions=str(d.get("instructions") or ""),
                   voice=str(d.get("voice") or DEFAULT_VOICE),
                   mcp_servers=[McpServer.from_dict(s) for s in d.get("mcp_servers") or []],
                   disabled_tools=[str(t) for t in d.get("disabled_tools") or []],
                   tool_timeout_seconds=timeout, max_call_minutes=minutes,
                   default_model=str(d.get("default_model") or ""),
                   transcription_model=str(d.get("transcription_model") or ""))


def load(path: str) -> list[HarnessConfig]:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    items = doc if isinstance(doc, list) else (doc.get("harnesses") if isinstance(doc, dict) and "harnesses" in doc else [doc])
    harnesses = [HarnessConfig.from_dict(d or {}) for d in items]
    ids = [h.id for h in harnesses]
    if len(set(ids)) != len(ids):
        raise ConfigError("two harnesses share an id")
    return harnesses
