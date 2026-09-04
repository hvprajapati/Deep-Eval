# Day 15 — The full RAG report

> **Module 3 · RAG evaluation.** All five RAG metrics on the whole pipeline, with the production-grade configs — then a written diagnosis per case.

## Code

### 15.1 · Run the pipeline across the golden set

# %%
# --- setup (from Day 03): load course/.env and recreate the course judge ---
import os

from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.models import LocalModel

judge = LocalModel(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",  # Groq speaks the OpenAI protocol
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)


# %%
# --- RAG recap (from Day 11): the corpus, the naive retriever, the grounded answerer ---
from groq import Groq

CORPUS = [
    {"title": "Tokens", "content": (
        "Large language models split text into tokens, common character sequences roughly 4 "
        "characters or three-quarters of a word long. Both the prompt and the output are counted "
        "in tokens, and pricing and context limits are measured in tokens.")},
    {"title": "Embeddings", "content": (
        "An embedding is a fixed-length vector that represents the meaning of a piece of text. "
        "Texts with similar meaning have vectors that are close together, usually measured by "
        "cosine similarity, which is what lets a system do semantic search.")},
    {"title": "Retrieval-Augmented Generation", "content": (
        "RAG grounds an LLM's answer in external documents. At query time the system retrieves "
        "the most relevant chunks and passes them to the model as context, which reduces "
        "hallucination and lets you update knowledge without retraining.")},
    {"title": "Hallucination", "content": (
        "A hallucination is fluent, confident text that is factually wrong or unsupported by its "
        "sources. It happens because the model predicts likely text rather than looking facts up. "
        "Grounding answers in retrieved context is the main defense.")},
    {"title": "AI agents", "content": (
        "An AI agent is an LLM given a goal, tools, and a loop: plan, call a tool, observe the "
        "result, decide the next step, until the task is done.")},
]

RAG_SYSTEM = (
    "Answer ONLY using the provided context. If the context doesn't contain the answer, say you "
    "don't have that information. Be concise (1-2 sentences)."
)


def retrieve(query: str, k: int = 2) -> list[str]:
    """Naive retriever: rank docs by keyword overlap with the query."""
    q_words = set(query.lower().split())
    scored = []
    for doc in CORPUS:
        doc_words = set((doc["title"] + " " + doc["content"]).lower().split())
        overlap = len(q_words & doc_words)
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc["content"] for _, doc in scored[:k]]


def answer_with_groq(question: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    resp = Groq().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def ask_rag(question: str, k: int = 2) -> tuple[str, list[str]]:
    """Full pipeline: retrieve, then answer. Returns (answer, passages)."""
    passages = retrieve(question, k)
    return answer_with_groq(question, passages), passages


QUESTIONS = [
    {"input": "What is a token in an LLM?",
     "expected_output": "A token is a common character sequence, roughly 4 characters, used "
                        "to measure both prompt and output length."},
    {"input": "How does RAG reduce hallucination?",
     "expected_output": "RAG retrieves relevant chunks and passes them as context, grounding "
                        "the answer in real documents."},
    {"input": "How do I containerize a model for deployment with Docker?"},
]

rag_rows = []
for q in QUESTIONS:
    answer, passages = ask_rag(q["input"])
    rag_rows.append({**q, "actual_output": answer, "retrieval_context": passages})

### 15.2 · All five RAG metrics, throttled and error-tolerant

# %%
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

test_cases = [
    LLMTestCase(
        input=r["input"],
        actual_output=r["actual_output"],
        expected_output=r.get("expected_output"),
        retrieval_context=r["retrieval_context"],
    )
    for r in rag_rows
]

metrics = [
    FaithfulnessMetric(model=judge),
    AnswerRelevancyMetric(model=judge),
    ContextualRelevancyMetric(model=judge),
    ContextualPrecisionMetric(model=judge),
    ContextualRecallMetric(model=judge),
]

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=1, throttle_value=2.0),  # ~40 judge calls — spread the burst
    display_config=DisplayConfig(print_results=True),
    error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
)

### 15.3 · Per-case failure analysis

# %%
for i, tr in enumerate(results.test_results, 1):
    print(f"\n=============== CASE {i}: {tr.input[:60]}")
    for md in tr.metrics_data:
        flag = "PASS" if md.success else "FAIL"
        print(f"  [{flag}] {md.name:<24} score={md.score}")
        print(f"         {md.reason[:160]}")
    # -- your diagnosis goes here --
    print("  diagnosis: <write one sentence: retriever or generator, and what to change>")

# %% [markdown]

## Theory

### Reading a full RAG report
Five metrics, three cases, roughly forty judge calls — and every failure has a name and an address:

- **Faithfulness / Answer Relevancy** fail → the *generator* (prompt, model, context assembly).
- **Contextual Precision / Recall / Relevancy** fail → the *retriever* (ranking, coverage, noise).
- **Recall fails but precision passes** → right docs ranked well, but key docs never returned → chunking or index coverage.
- **Precision fails but recall passes** → everything is in there, ranked badly → ranking/scoring layer.

Write the diagnosis sentence in 15.3 by hand — the deliberate pause turns a score table into an engineering decision. Day 16 acts on it.

### Why the token question fails
The naive retriever counts keyword overlap. "What is a token in an LLM?" shares almost no words with the Tokens document ("Large language models split text into tokens...") — no "LLM", no "what". Meanwhile the RAG document literally contains "LLM" and wins the ranking. The retriever returned mostly the wrong context, so the generator said "I don't have that information." That is the day's lesson in miniature: **a failing answer often has a healthy generator and a sick retriever** — the metrics are what let you tell the difference.

### The configs, explained
- `AsyncConfig(max_concurrent=1, throttle_value=2.0)` — runs metrics asynchronously but space out judge calls, staying under Groq's rate limits while ~40 calls finish quickly.
- `ErrorConfig(ignore_errors=True, skip_on_missing_params=True)` — a case missing a field a metric needs is *skipped* for that metric instead of crashing the run. Essential when cases are heterogeneous (the Docker case has no `expected_output`).
- `DisplayConfig(print_results=True)` — the full pass/fail table in the notebook.
