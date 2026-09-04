# Day 14 — Generator metrics, Faithfulness & Answer Relevancy on real RAG

> **Module 3 · RAG evaluation.** Now judge the *answers* — grounded in whatever the retriever actually returned, including its misses.

## Code

### 14.1 · Run the pipeline on three questions

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
    {"input": "How do I containerize a model for deployment with Docker?"},  # not in the corpus
]

rag_rows = []
for q in QUESTIONS:
    answer, passages = ask_rag(q["input"])
    rag_rows.append({**q, "actual_output": answer, "retrieval_context": passages})
    print("Q:", q["input"])
    print("A:", answer)
    print("-" * 80)

### 14.2 · Build cases and measure the generator

# %%
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
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

metrics = [FaithfulnessMetric(model=judge), AnswerRelevancyMetric(model=judge)]
results = evaluate(test_cases=test_cases, metrics=metrics)
for i, tr in enumerate(results.test_results, 1):
    print(f"\n=== Q{i}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"  {md.name:<22} score={md.score}")
        print(f"    reason: {md.reason[:140]}")

# %% [markdown]

## Theory

### Grounding vs helpfulness, measured separately
- **Faithfulness** — is the answer supported by the retrieved context? It says nothing about whether the answer is *good*.
- **Answer Relevancy** — does the answer address the question? It says nothing about whether it is *true*.

A RAG answer needs both: grounded in the chunks AND on-topic. The two scores together describe the generator; the retriever metrics from Days 12–13 describe the retriever. When something fails, the first question is always: which half is broken?

### The interesting case: the Docker question
"How do I containerize a model?" is not answerable from the five-doc corpus. Watch what the pipeline does:

1. The retriever returns its best (irrelevant) chunks — retrieval failed, silently.
2. The generator, bound by "answer ONLY using the provided context", refuses: "I don't have that information."

That refusal is *correct* behavior, and the metrics should agree: faithfulness may pass (the refusal contradicts nothing — it adds no unsupported claims), while relevancy fails (no answer was given). This is the honest outcome: a RAG should fail loudly on out-of-scope questions, and your eval pack should record that failure. If the generator instead invents a Docker answer, faithfulness catches the invention — that is the regression the pack exists to prevent.

### Expected output is optional here
Faithfulness and Answer Relevancy never read `expected_output` — the judge works from context and question alone. The golden answers in `QUESTIONS` are for the reference-based metrics (Day 15's suite includes them for free on other metrics, and Day 16 uses them to diff retrievers).
