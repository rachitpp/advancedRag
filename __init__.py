"""
CRAG — Corrective RAG pipeline with two-tier memory.

Public API:
  - CRAGConfig: typed configuration
  - CRAGPipeline: pipeline orchestrator (use as context manager)
  - CRAGResult: structured result
  - FallbackReason: enum of fallback causes
  - Conversation, ConversationStore: conversational memory primitives
  - InMemoryConversationStore: default backend (swap with Redis in prod)

Typical usage (stateless):

    from crag import CRAGConfig, CRAGPipeline

    cfg = CRAGConfig.from_env()
    with CRAGPipeline(cfg) as pipeline:
        result = pipeline.run_query("What is X?")
        print(result.answer)

With conversational memory:

    with CRAGPipeline(cfg) as pipeline:
        sid = "session-123"
        r1 = pipeline.run_query("Capital of France?", session_id=sid)
        r2 = pipeline.run_query("What's its population?", session_id=sid)
        # ^ Pipeline rewrites to "What is the population of Paris, France?"

With user memory (requires CRAG_USER_MEMORY_ENABLED=true):

    with CRAGPipeline(cfg) as pipeline:
        pipeline.run_query(
            "I'm vegetarian. Suggest a pasta dish.",
            session_id="s1", user_id="alice",
        )
        # Days later, fresh session:
        pipeline.run_query(
            "Recommend a restaurant.",
            session_id="s2", user_id="alice",
        )
        # ^ User memory injects: "User is vegetarian"
"""

from .config import CRAGConfig
from .conversation import (
    Conversation,
    ConversationStore,
    InMemoryConversationStore,
)
from .observability import FallbackReason
from .pipeline import CRAGPipeline, CRAGResult

__all__ = [
    "CRAGConfig",
    "CRAGPipeline",
    "CRAGResult",
    "FallbackReason",
    "Conversation",
    "ConversationStore",
    "InMemoryConversationStore",
]
