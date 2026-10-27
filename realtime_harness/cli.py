"""`rth`: the Realtime Harness command line.

    rth serve --config harness.yaml [--host 127.0.0.1] [--port 8790]

The server's API keys come from RTH_API_KEYS or RTH_API_KEYS_FILE; the provider connection from the
environment (realtime_harness/credentials.py): a HarnessRouter server's mint, Azure OpenAI, or OpenAI.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import ConfigError, load
from .credentials import CredentialError, from_env
from .uhp import Server, api_keys_from_env, create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rth", description="Realtime Harness: a harness for realtime speech models, over UHP.")
    ap.add_argument("--version", action="version", version=f"rth {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve the harness over UHP")
    s.add_argument("--config", required=True, help="the harness file (YAML)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8790)
    s.add_argument("--model", action="append", help="a model this server runs (repeat); default gpt-realtime-2.1")
    args = ap.parse_args(argv)
    try:
        harnesses = load(args.config)
    except (OSError, ConfigError) as e:
        print(f"rth: {e}", file=sys.stderr)
        return 2
    keys = api_keys_from_env()
    if not keys:
        print("rth: set RTH_API_KEYS or RTH_API_KEYS_FILE: the server answers only bearer-authenticated calls", file=sys.stderr)
        return 2
    try:
        creds = from_env()
    except CredentialError as e:
        print(f"rth: {e}", file=sys.stderr)
        return 2
    import uvicorn
    app = create_app(Server(harnesses, creds, keys, models=args.model))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
