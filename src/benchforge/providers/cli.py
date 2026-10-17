"""Offline discovery and explicit activation diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from benchforge.core.provider_config import ProviderRef
from benchforge.providers.runtime import discover, resolve


def add_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("plugins", help="discover or inspect installed providers")
    commands = parser.add_subparsers(dest="plugin_command", required=True)
    commands.add_parser("list", help="list package metadata without importing provider code")
    inspect = commands.add_parser(
        "inspect", help="activate a trusted provider and validate contracts"
    )
    inspect.add_argument("provider")
    inspect.add_argument(
        "--version", required=True, help="expected provider implementation version"
    )


def dispatch(args: argparse.Namespace) -> int:
    if args.plugin_command == "list":
        print(json.dumps([asdict(item) for item in discover()], indent=2, sort_keys=True))
    else:
        resolved = resolve((ProviderRef(id=args.provider, version=args.version),))
        print(
            json.dumps(
                {
                    "manifests": [m.canonical_dict() for m in resolved.manifests],
                    "distributions": [asdict(item) for item in resolved.provenance],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0
