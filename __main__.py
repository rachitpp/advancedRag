"""
CLI entry point for interactive CRAG queries.

Run with:  python -m crag
With user memory: CRAG_USER_MEMORY_ENABLED=true python -m crag --user alice
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid

from dotenv import load_dotenv

from .config import CRAGConfig
from .pipeline import CRAGPipeline, CRAGResult


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="crag", description="CRAG interactive CLI")
    p.add_argument("--user", default=None)
    p.add_argument("--tenant", default=None)
    p.add_argument("--no-memory", action="store_true")
    return p.parse_args()


async def _amain(args: argparse.Namespace) -> int:
    cfg = CRAGConfig.from_env()
    if args.no_memory:
        from dataclasses import replace
        cfg = replace(cfg, conversation_enabled=False)

    user_id = args.user
    tenant_id = args.tenant
    session_id = str(uuid.uuid4())

    print("=" * 60)
    print("CRAG System Ready")
    print(f"  Main model      : {cfg.main_model}")
    print(f"  Fast model      : {cfg.fast_model}")
    print(f"  Qdrant          : {cfg.qdrant_url} / {cfg.qdrant_collection}")
    print(f"  Cache           : {'on' if cfg.cache_enabled else 'off'}")
    print(f"  HyDE            : {cfg.hyde_enabled}")
    print(f"  User memory     : {cfg.user_memory_enabled and bool(user_id)}")
    print(f"  Session ID      : {session_id}")
    if user_id:
        print(f"  User ID         : {user_id}")
    if tenant_id:
        print(f"  Tenant ID       : {tenant_id}")
    print("  Commands: /json /reset /forget /info  |  '0' to exit")
    print("=" * 60)

    show_json = False

    async with CRAGPipeline(cfg) as pipeline:
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = (await loop.run_in_executor(None, input, "\nUser:\n")).strip()
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
                await pipeline.end_session(session_id)
                session_id = str(uuid.uuid4())
                print(f"[new session: {session_id}]")
                continue
            if user_input == "/forget":
                if not user_id:
                    print("[no --user provided; nothing to forget]")
                    continue
                await pipeline.forget_user(user_id, tenant_id)
                print(f"[wiped user memory for: {user_id}]")
                continue
            if user_input == "/info":
                print(f"  session_id : {session_id}")
                print(f"  user_id    : {user_id or '(none)'}")
                print(f"  tenant_id  : {tenant_id or '(none)'}")
                continue

            print("\nAssistant:")
            result: CRAGResult | None = None
            async for chunk in pipeline.astream_query(
                user_input,
                session_id=session_id,
                user_id=user_id,
                tenant_id=tenant_id,
            ):
                if isinstance(chunk, CRAGResult):
                    result = chunk
                else:
                    print(chunk, end="", flush=True)
            print()
            assert result is not None

            if result.is_fallback or result.self_corrected:
                print(f"\n[Final answer (post-verification)]:\n{result.answer}")

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

            if show_json:
                print("\n--- Telemetry ---")
                print(json.dumps(result.telemetry, indent=2))


def main() -> int:
    load_dotenv()
    _setup_logging()
    args = _parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
