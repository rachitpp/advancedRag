"""
CLI entry point for interactive CRAG queries.

Run with:  python -m crag
With user memory: CRAG_USER_MEMORY_ENABLED=true python -m crag --user alice

Special commands at the prompt:
  /json    — toggle telemetry JSON dump after each query
  /reset   — start a fresh conversation session (forgets short-term context)
  /forget  — wipe long-term user memory (requires --user flag)
  /info    — show current session/user state
  0        — exit
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid

from dotenv import load_dotenv

from .config import CRAGConfig
from .pipeline import CRAGPipeline


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,   # logs -> stderr; answer streaming -> stdout
    )


def _stream_to_stdout(token: str) -> None:
    print(token, end="", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="crag", description="CRAG interactive CLI")
    p.add_argument(
        "--user", default=None,
        help="User ID for long-term memory (requires CRAG_USER_MEMORY_ENABLED=true).",
    )
    p.add_argument(
        "--no-memory", action="store_true",
        help="Disable conversational memory for this run.",
    )
    return p.parse_args()


def main() -> int:
    load_dotenv()
    _setup_logging()
    args = _parse_args()

    cfg = CRAGConfig.from_env()
    if args.no_memory:
        # Override config for this run via a quick clone.
        from dataclasses import replace
        cfg = replace(cfg, conversation_enabled=False)

    user_id = args.user
    session_id = str(uuid.uuid4())

    print("=" * 60)
    print("CRAG System Ready")
    print(f"  Main model      : {cfg.main_model}")
    print(f"  Fast model      : {cfg.fast_model}")
    print(f"  HyDE            : {cfg.hyde_enabled}")
    print(f"  Conversation    : {cfg.conversation_enabled}")
    print(f"  User memory     : {cfg.user_memory_enabled and bool(user_id)}")
    print(f"  Session ID      : {session_id}")
    if user_id:
        print(f"  User ID         : {user_id}")
    print("  Commands: /json /reset /forget /info  |  '0' to exit")
    print("=" * 60)

    show_json = False

    with CRAGPipeline(cfg) as pipeline:
        while True:
            try:
                user_input = input("\nUser:\n").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return 0

            if not user_input:
                continue
            if user_input == "0":
                return 0
            if user_input == "/json":
                show_json = not show_json
                print(f"[telemetry dump: {'on' if show_json else 'off'}]")
                continue
            if user_input == "/reset":
                pipeline.end_session(session_id)
                session_id = str(uuid.uuid4())
                print(f"[new session: {session_id}]")
                continue
            if user_input == "/forget":
                if not user_id:
                    print("[no --user provided; nothing to forget]")
                    continue
                pipeline.forget_user(user_id)
                print(f"[wiped user memory for: {user_id}]")
                continue
            if user_input == "/info":
                print(f"  session_id : {session_id}")
                print(f"  user_id    : {user_id or '(none)'}")
                continue

            print("\nAssistant:")
            result = pipeline.run_query(
                user_input,
                session_id=session_id,
                user_id=user_id,
                on_token=_stream_to_stdout,
            )
            print()  # newline after streamed answer

            print("\n--- Result ---")
            print(f"  Faithfulness     : {result.faithfulness_score:.0%}")
            print(f"  Web search       : {result.used_web_search}")
            print(f"  Self-corrected   : {result.self_corrected}")
            print(f"  Regressed        : {result.correction_regressed}")
            print(f"  Fallback         : {result.fallback_reason.value}")
            print(f"  Source docs      : {len(result.source_docs)}")
            if result.standalone_question:
                print(f"  Rewritten as     : {result.standalone_question[:80]}")
            if result.user_facts_used:
                print(f"  User facts used  : {len(result.user_facts_used)}")
                for fact in result.user_facts_used:
                    print(f"    - {fact}")

            if show_json:
                print("\n--- Telemetry ---")
                print(json.dumps(result.telemetry, indent=2))


if __name__ == "__main__":
    sys.exit(main())
