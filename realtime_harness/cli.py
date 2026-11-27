"""`rth`: the Realtime Harness command line.

    rth serve --config harness.yaml [--host 127.0.0.1] [--port 8790]

The server's API keys come from RTH_API_KEYS or RTH_API_KEYS_FILE; the provider connection from the
environment (realtime_harness/credentials.py): a HarnessRouter server's mint, Azure OpenAI, or OpenAI.

Behind a server that places the calls (HarnessRouter), RTH_OPERATOR_KEYS or RTH_OPERATOR_KEYS_FILE holds
that server's key. Each call it creates brings its own configuration and its own provider connection, so
such a harness needs neither a harness file nor provider credentials of its own:

    rth serve [--host 127.0.0.1] [--port 8790]
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import ConfigError, load
from .credentials import CredentialError, from_env
from .uhp import Server, api_keys_from_env, create_app, operator_keys_from_env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rth", description="Realtime Harness: a harness for realtime speech models, over UHP.")
    ap.add_argument("--version", action="version", version=f"rth {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve the harness over UHP")
    s.add_argument("--config", help="the harness file (YAML); optional with an operator key")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8790)
    s.add_argument("--model", action="append", help="a model this server runs (repeat); default gpt-realtime-2.1")
    args = ap.parse_args(argv)
    operators = operator_keys_from_env()
    if not args.config and not operators:
        print("rth: --config names the harness file; only a server with RTH_OPERATOR_KEYS runs without one", file=sys.stderr)
        return 2
    try:
        harnesses = load(args.config) if args.config else []
    except (OSError, ConfigError) as e:
        print(f"rth: {e}", file=sys.stderr)
        return 2
    keys = api_keys_from_env()
    if not keys and not operators:
        print("rth: set RTH_API_KEYS or RTH_API_KEYS_FILE: the server answers only bearer-authenticated calls", file=sys.stderr)
        return 2
    try:
        creds = from_env()
    except CredentialError as e:
        if not operators:
            print(f"rth: {e}", file=sys.stderr)
            return 2
        creds = None        # every call an operator creates brings its own connection
    import uvicorn
    app = create_app(Server(harnesses, creds, keys, models=args.model, operator_keys=operators))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
