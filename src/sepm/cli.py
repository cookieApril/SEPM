"""Read-only command-line inspection tools included with the SEPM package.

``retrieve`` queries published SOPs in natural language, and ``candidates``
inspects the candidate queue. The CLI intentionally exposes no write operation;
production writes should use the ``SEPMService`` Python API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .service import SEPMService


def main() -> None:
    """Parse a subcommand, read an SQLite database, and emit UTF-8 JSON."""
    parser = argparse.ArgumentParser(description="Inspect a SEPM database")
    parser.add_argument("--db", type=Path, default=Path("sepm.db"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("query")
    retrieve.add_argument("--limit", type=int, default=5)
    candidates = subparsers.add_parser("candidates")
    candidates.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    service = SEPMService(args.db)
    # argparse restricts command to two choices, so the else branch is candidates.
    if args.command == "retrieve":
        payload = [item.model_dump(mode="json") for item in service.retrieve_sops(args.query, args.limit)]
    else:
        items, total = service.store.list_candidates(None, args.limit, 0)
        payload = {"total": total, "items": [item.model_dump(mode="json") for item in items]}
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
