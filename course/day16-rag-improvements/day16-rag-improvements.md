# Day 16 — Evaluation-driven RAG improvements

> **Module 3 · RAG evaluation.** The payoff day: change the retriever, re-run the pack, watch the scores move. This loop is the job.

## Code

### 16.1 · A better retriever: TF-IDF cosine similarity

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- RAG recap (from Day 11): the corpus ---
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

docs = [d["title"] + ". " + d["content"] for d in CORPUS]
vectorizer = TfidfVectorizer(stop_words="english")
doc_vectors = vectorizer.fit_transform(docs)


def retrieve_tfidf(query: str, k: int = 2) -> list[str]:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vectors)[0]
    ranked = sorted(zip(sims, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


for q in ["What is a token in an LLM?", "How does RAG reduce hallucination?"]:
    print("Q:", q)
    for i, d in enumerate(retrieve_tfidf(q), 1):
        print(f"  {i}. {d[:80]}...")

### 16.2 · Rebuild the pipeline on the new retriever

# %%
# --- RAG recap (from Day 11): the grounded answerer ---
from groq import Groq

RAG_SYSTEM = (
    "Answer ONLY using the provided context. If the context doesn't contain the answer, say you "
    "don't have that information. Be concise (1-2 sentences)."
)


def answer_with_groq(question: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    resp = Groq().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


QUESTIONS = [
    {"input": "What is a token in an LLM?",
     "expected_output": "A token is a common character sequence, roughly 4 characters, used "
                        "to measure both prompt and output length."},
    {"input": "How does RAG reduce hallucination?",
     "expected_output": "RAG retrieves relevant chunks and passes them as context, grounding "
                        "the answer in real documents."},
    {"input": "How do I containerize a model for deployment with Docker?"},
]

rows = []
for q in QUESTIONS:
    passages = retrieve_tfidf(q["input"])
    answer = answer_with_groq(q["input"], passages)
    rows.append({**q, "actual_output": answer, "retrieval_context": passages})
    print("Q:", q["input"], "\nA:", answer, "\n" + "-" * 80)

### 16.3 · Re-run the full suite and diff

# %%
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
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
    for r in rows
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
    async_config=AsyncConfig(max_concurrent=1, throttle_value=2.0),
    error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
)

# Compare with Day 15's numbers by hand: which scores moved, and why?
for i, tr in enumerate(results.test_results, 1):
    print(f"case {i}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"   {md.name:<24} {md.score}")

### 16.4 · Grow the golden set from your own docs (synthetic generation)

# %%
# A template-based generator: turn each corpus doc into 1-2 questions.
TEMPLATES = [
    lambda t: f"What is {t.lower()}?",
    lambda t: f"Why does {t.lower()} matter?",
]

new_cases = []
for doc in CORPUS:
    for make_q in TEMPLATES:
        q = make_q(doc["title"])
        answer, passages = answer_with_groq(q, retrieve_tfidf(q)), retrieve_tfidf(q)
        new_cases.append({"input": q, "actual_output": answer,
                          "retrieval_context": passages})

print(f"Generated {len(new_cases)} new cases; add the good ones to the golden set "
      f"and re-run 16.3 — the pack is now {len(test_cases) + len(new_cases)} strong.")

# %% [markdown]

## Theory

### The optimize loop
Hypothesis → change → re-evaluate → keep or revert. Today's hypothesis: *keyword overlap fails on semantic queries; TF-IDF cosine will surface the Tokens doc for "What is a token in an LLM?"*. The pack decides. Notice the discipline: the **suite did not change** between Day 15 and Day 16 — the retriever did. Changing both at once makes score deltas unreadable.

### What TF-IDF fixed and what it didn't
TF-IDF matches on statistically distinctive terms, so "token" and "hallucination" now connect to their documents even when the query rephrases. But it still has no notion of *meaning* — "containerize a model" still finds nothing relevant (there is nothing to find). Score movements are specific: precision and recall on the token question should jump; the Docker case should still (correctly) refuse.

### Synthetic test generation, honestly
The 16.4 templates generate questions mechanically from your own documents. Real teams go further — rewriting questions with an LLM for variety, and (with the Confident AI platform) `generate_goldens_from_docs`. The rules stay the same:
1. **Never trust auto-generated cases blindly** — spot-check them before they join the golden set.
2. **The golden set only grows** — old cases are the regression contract; new cases raise the bar.

### The regression pack
The five-question suite is now the project's safety net. Every future change — chunk size, embedding model, prompt wording, LLM swap — runs against it first. That is the entire point of the module: the pack is the product's memory of what "working" means.
