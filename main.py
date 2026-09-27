#!/usr/bin/env python3
"""
arXiv Digest Agent — CLI Entry Point

Usage:
    python main.py digest "recent work on KV-cache compression for LLMs"
    python main.py digest 2401.12345
    python main.py digest https://arxiv.org/abs/2401.12345 --no-qa
    python main.py qa 2401.12345
    python main.py sessions
    python main.py demo
"""
from __future__ import annotations

import argparse
import sys

from src.cli import cmd_digest, cmd_qa, cmd_sessions, cmd_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="🔬 Autonomous arXiv Paper Digest & QA Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py digest "KV-cache compression for LLMs"\n'
            "  python main.py digest 2401.12345\n"
            "  python main.py qa 2401.12345\n"
            "  python main.py sessions\n"
            "  python main.py demo"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── digest ──
    p_digest = sub.add_parser(
        "digest",
        help="Fetch a paper (by topic or arXiv ID/URL), summarize, and enter QA",
    )
    p_digest.add_argument("query", help="Research topic, arXiv ID, or arXiv URL")
    p_digest.add_argument(
        "--no-qa",
        action="store_true",
        help="Print the briefing and exit (skip interactive QA)",
    )

    # ── qa ──
    p_qa = sub.add_parser(
        "qa",
        help="Resume the QA loop for a previously ingested paper",
    )
    p_qa.add_argument("arxiv_id", help="arXiv ID of a saved session (see `sessions`)")

    # ── sessions ──
    sub.add_parser("sessions", help="List all saved sessions")

    # ── demo ──
    sub.add_parser(
        "demo",
        help="Demo using the bundled synthetic paper (needs API keys, no arXiv fetch)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "digest":
        cmd_digest(args.query, skip_qa=args.no_qa)
    elif args.command == "qa":
        cmd_qa(args.arxiv_id)
    elif args.command == "sessions":
        cmd_sessions()
    elif args.command == "demo":
        cmd_demo()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
