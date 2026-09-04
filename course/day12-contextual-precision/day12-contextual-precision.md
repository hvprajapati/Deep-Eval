# Day 12 — Contextual Precision, is the ranking right?

> **Module 3 · RAG evaluation.** First retriever metric: the right document in the wrong position is a retrieval failure.

## Code

### 12.1 · What the naive retriever returns

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
# --- RAG recap (from Day 11): the corpus and the naive retriever ---
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


q = "What is a token in an LLM?"
nodes = retrieve(q, k=3)
for i, n in enumerate(nodes, 1):
    print(f"{i}. {n[:90]}...")

### 12.2 · Two rankings: correct first vs correct buried

# %%
from deepeval.test_case import LLMTestCase

GOOD_ANSWER = "A token is a common character sequence, roughly 4 characters, used to measure prompt and output length."

# case A: the Tokens doc ranks first
case_first = LLMTestCase(
    input=q,
    actual_output=GOOD_ANSWER,
    retrieval_context=[
        "Large language models split text into tokens, common character sequences roughly 4 characters...",
        "An embedding is a fixed-length vector that represents the meaning of a piece of text...",
        "RAG grounds an LLM's answer in external documents...",
    ],
)

# case B: same docs, Tokens buried in last position
case_buried = LLMTestCase(
    input=q,
    actual_output=GOOD_ANSWER,
    retrieval_context=[
        "An embedding is a fixed-length vector that represents the meaning of a piece of text...",
        "RAG grounds an LLM's answer in external documents...",
        "Large language models split text into tokens, common character sequences roughly 4 characters...",
    ],
)

### 12.3 · Measure precision and inspect the judge's walk

# %%
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric

metric = ContextualPrecisionMetric(model=judge, threshold=0.5)
results = evaluate(test_cases=[case_first, case_buried], metrics=[metric])
for tr in results.test_results:
    md = tr.metrics_data[0]
    print(f"{tr.name}: score={md.score} success={md.success}")
    print(f"  reason: {md.reason[:160]}\n")

print("=== judge's node-by-node walk (last case) ===")
print(metric.verbose_logs[:1200])

# %% [markdown]

## Theory

### Precision is about order
Contextual Precision answers: *of the nodes that are relevant, how early do they appear?* The same three documents score differently depending on position. Case A passes; case B — identical content, reordered — fails. Precision is the metric for ranking quality, and ranking is what a retriever *is*.

### How the judge scores it
The judge walks the retrieved list node by node and asks: is this node relevant to the question, and does it add information the earlier nodes didn't have? Each node earns a verdict that decays with position — relevant info found at position 3 is worth less than the same info at position 1. `verbose_logs` shows the whole walk, one verdict per node. Read it when a precision score surprises you.

### Why ranking matters — top-k truncation
Real pipelines keep only the top-k chunks and throw the rest away. A document at position 4 in a `k=3` pipeline functionally does not exist. Precision captures exactly this: it is the metric that answers "will the correct chunk survive truncation?"

### Precision vs the metrics tomorrow
- **Precision** — right docs, right order (today).
- **Recall** — right docs present *anywhere* in the list (Day 13).
- **Relevancy** — how much of the list is noise (Day 13).

A retriever with high recall but low precision finds everything but buries it; high precision but low recall ranks beautifully what little it finds. Day 13 completes the diagnostic picture.
