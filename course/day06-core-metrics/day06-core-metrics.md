# Day 06 — Faithfulness, Hallucination & Answer Relevancy

> **Module 2 · The metric toolkit.** The "big three" generator metrics: is the answer supported, does it invent facts, and does it actually answer?

## Code

### 6.1 · Three answers, one supporting context

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
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)

CONTEXT = [
    "The refund window is 30 days from the purchase date.",
    "Digital products are refundable only if never downloaded.",
]

test_cases = [
    LLMTestCase(
        input="Can I get a refund on an ebook I never downloaded?",
        actual_output="Yes — refunds are available within 30 days, and digital products "
                      "are refundable if they were never downloaded.",
        retrieval_context=CONTEXT,
    ),
    LLMTestCase(
        input="Can I get a refund on an ebook I never downloaded?",
        actual_output="Yes, and as a thank-you we also give you a free physical copy, "
                      "free shipping, and a lifetime subscription.",
        retrieval_context=CONTEXT,
    ),
    LLMTestCase(
        input="Can I get a refund on an ebook I never downloaded?",
        actual_output="The weather in Portugal is lovely this time of year. "
                      "Our office has four floors and a great espresso machine.",
        retrieval_context=CONTEXT,
    ),
]

### 6.2 · Measure all three metrics

# %%
metrics = [
    FaithfulnessMetric(model=judge),
    HallucinationMetric(model=judge),
    AnswerRelevancyMetric(model=judge),
]

from deepeval import evaluate

results = evaluate(test_cases=test_cases, metrics=metrics)
for i, tr in enumerate(results.test_results, 1):
    print(f"\n=== case {i}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"  {md.name:<22} score={md.score}")
        print(f"    reason: {md.reason[:130]}")

# %% [markdown]

## Theory

### How Faithfulness works
The judge extracts every factual *claim* from the answer, then checks each claim against the retrieval context. Score = the fraction of claims the context supports. Notice what it does **not** do: it does not check whether the answer answered the question, and it does not check whether the context itself is right. Faithfulness measures *grounding*, nothing else. An answer that faithfully repeats a wrong document scores 1.0 — that is Contextual Recall's job (Day 13).

### How Hallucination works
A stricter cousin: it hunts for claims that *contradict* the context. Case 2 above — "free physical copy, lifetime subscription" — is unfaithful (unsupported) and, depending on the context, hallucinatory (contradictory). The two scores usually move together, but contradiction is the more severe failure, so Hallucination is the metric you gate on when your users can be hurt by invented facts.

### How Answer Relevancy works
The judge extracts the claims of the answer and asks: how many of them actually answer the question? Case 3 is *faithful* (it contradicts nothing) but entirely irrelevant. Relevancy is the only one of the three that needs no context — just question + answer. For a chatbot, low relevancy with high faithfulness = the pipeline works but the model rambles.

### Reading the three together
| Faithfulness | Hallucination | Relevancy | diagnosis |
|---|---|---|---|
| low | low | high | answer goes beyond the context — loosen context or fix grounding prompt |
| high | high | high | healthy |
| high | high | low | correct but unhelpful — prompt/temperature issue |
| high | low | high | contains contradictions — inspect the context quality itself |

Day 14 brings these three onto a real RAG pipeline, where the context comes from a retriever instead of hand-written strings.
