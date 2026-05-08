"""
All prompts in one place.

Why one file: prompt drift is a top-3 source of regression in RAG systems.
Having every prompt visible in one ~100-line file makes audit trivial — you
can diff against last week and see exactly what changed in the system's
behavior. The moment we have >15 prompts or per-prompt versioning needs,
this splits into a directory. Not yet.

Each prompt is a ChatPromptTemplate factory function — they take no args
because we want template-time errors (missing variable) to surface at module
import, not at first query.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------- Generation

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise, helpful assistant. Answer ONLY from the provided context. "
     "Be concise and structured. If the answer is not in the context, say: "
     "\"I could not find the answer in the provided documents.\" "
     "Do not hallucinate. Cite chunk numbers (e.g., [Chunk 3]) where useful."),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

SELF_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant performing a factual correction. Your previous "
     "answer contained claims NOT supported by the provided context. Rewrite the "
     "answer so every statement is directly supported. For each unsupported claim "
     "below, replace it with what the context says, or omit it. Do not invent."),
    ("human",
     "CONTEXT:\n{context}\n\n"
     "QUESTION:\n{question}\n\n"
     "PREVIOUS ANSWER:\n{previous_answer}\n\n"
     "UNSUPPORTED CLAIMS:\n{unsupported_claims}\n\n"
     "Corrected answer:"),
])


# ---------------------------------------------------------------- Grading

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance grader. Answer 'yes' only if the chunk directly "
     "helps answer the question, 'no' if off-topic or only tangentially related."),
    ("human",
     "Document chunk:\n{document}\n\nQuestion: {question}\n\nIs this chunk relevant?"),
])


# ---------------------------------------------------------------- Query rewriting / HyDE

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query optimizer. Rewrite the user's question into a "
     "concise, keyword-rich web search query. Output ONLY the rewritten query."),
    ("human", "Original: {question}\n\nRewritten:"),
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Write a brief, plausible passage (3-5 sentences) that would directly answer "
     "the user's question, as if extracted from a reference document. Do not hedge, "
     "do not say 'I don't know' — write the passage as if the answer is known. "
     "This is for retrieval, not for the user, so confident tone is required."),
    ("human", "Question: {question}\n\nPassage:"),
])

QUERY_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the question into exactly one category:\n"
     "  factual    — asks for a specific fact, name, date, number\n"
     "  conceptual — asks to explain, compare, or describe a concept\n"
     "  procedural — asks how to do something or for steps\n"
     "Output ONLY the single word: factual, conceptual, or procedural."),
    ("human", "{question}"),
])


# ---------------------------------------------------------------- Verification

ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Decide if the answer addresses the question. Output exactly one word: "
     "yes (addresses it), no (off-topic or refuses), or partial (touches it but "
     "misses the core ask)."),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nVerdict:"),
])

CLAIM_EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract every individual factual claim from the answer. Each claim must be "
     "a single, atomic, verifiable proposition. Exclude hedges, meta-phrases, and "
     "phrases like 'according to the context'. Reject claims shorter than 3 words."),
    ("human", "Answer:\n{answer}\n\nList all factual claims:"),
])

# Batched verifier — N claims per call, returns N verdicts in order.
CLAIM_BATCH_VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fact verifier. For each numbered claim, decide if the CONTEXT "
     "explicitly supports it. supported=true only if the context clearly states "
     "the claim. supported=false if absent, contradicted, or only implied. "
     "Return one verdict per claim, in the same order, matching by index."),
    ("human",
     "CONTEXT:\n{context}\n\n"
     "CLAIMS:\n{numbered_claims}\n\n"
     "Verdicts (one per claim, in order):"),
])