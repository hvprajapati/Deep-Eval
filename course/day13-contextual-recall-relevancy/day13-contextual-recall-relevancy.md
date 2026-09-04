# Day 13 — Contextual Recall & Relevancy

> **Module 3 · RAG evaluation.** Complete the retriever diagnostic: recall (is the needed info anywhere?) and relevancy (how much noise came along?).

## Code

### 13.1 · Build the three diagnostic cases

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
from deepeval.test_case import LLMTestCase

q = "How does RAG reduce hallucination?"
EXPECTED = ("RAG reduces hallucination by retrieving relevant chunks and passing them as context, "
            "grounding the answer in real documents.")

# Case A: right docs, right order
case_a = LLMTestCase(
    input=q, actual_output=EXPECTED, expected_output=EXPECTED,
    retrieval_context=[
        "RAG grounds an LLM's answer in external documents. At query time the system retrieves "
        "the most relevant chunks and passes them to the model as context, which reduces "
        "hallucination and lets you update knowledge without retraining.",
    ],
)

# Case B: the answer-relevant doc is MISSING entirely
case_b = LLMTestCase(
    input=q, actual_output=EXPECTED, expected_output=EXPECTED,
    retrieval_context=[
        "An embedding is a fixed-length vector that represents the meaning of a piece of text. "
        "Texts with similar meaning have vectors that are close together, usually measured by "
        "cosine similarity.",
    ],
)

# Case C: right doc present, but buried under noise
case_c = LLMTestCase(
    input=q, actual_output=EXPECTED, expected_output=EXPECTED,
    retrieval_context=[
        "Llamas are domesticated South American camelids, related to the alpaca.",
        "An embedding is a fixed-length vector that represents the meaning of a piece of text.",
        "The capital of France is Paris.",
        "RAG grounds an LLM's answer in external documents and passes retrieved chunks to the "
        "model as context, which reduces hallucination.",
    ],
)

### 13.2 · Measure all three retriever metrics together

# %%
from deepeval import evaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

metrics = [
    ContextualPrecisionMetric(model=judge),
    ContextualRecallMetric(model=judge),
    ContextualRelevancyMetric(model=judge),
]
results = evaluate(test_cases=[case_a, case_b, case_c], metrics=metrics)

for i, tr in enumerate(results.test_results, 1):
    line = f"case {i}"
    for md in tr.metrics_data:
        line += f"  |  {md.name}={md.score}"
    print(line)

for tr in results.test_results:
    print(f"\n--- {tr.name} ---")
    for md in tr.metrics_data:
        print(f"  {md.name:<24} score={md.score}")
        print(f"    {md.reason[:120]}")

# %% [markdown]

## Theory

### Recall: is the needed information anywhere?
Contextual Recall checks: could someone produce the `expected_output` from the retrieved nodes alone? It walks the nodes and asks whether each sentence of the expected answer is covered by *some* node — position does not matter. Case B fails recall because the one document that explains the answer never came back. Recall is the metric for *coverage*: chunking that splits facts across documents, indexes that miss synonyms, embeddings too small for the domain — all show up here.

### Relevancy: how much noise came along?
Contextual Relevancy counts how many retrieved nodes the answer actually needed. Case C has the right doc at position 4 — but three irrelevant nodes drowned it, so relevancy tanks while recall is fine. Relevancy is the metric for *signal-to-noise*: over-eager retrieval, broken filters, queries that match too much.

### Reading the three together
| precision | recall | relevancy | diagnosis |
|---|---|---|---|
| high | high | high | retriever healthy |
| high | low | high | right stuff ranked, but key docs missing → chunking / index coverage |
| low | high | high | everything present but buried → ranking model, not the index |
| high | high | low | correct docs plus garbage → filters, query expansion, or k too big |
| low | low | low | retriever is guessing → start over |

Case A should score high on all three; B fails recall; C fails precision and relevancy. If your run shows something else, read the judge reasons — the walks are printed above — before concluding the metric is wrong: the judge usually found something real.
