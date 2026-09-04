# Day 07 — Summarization, JSON & score-only mode

> **Module 2 · The metric toolkit.** Reference-free metrics, structured-output checks, and two knobs that decide *who* fails: score-only and flaky metrics.

## Code

### 7.1 · Summarization without a golden summary

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
from deepeval import evaluate
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

ARTICLE = (
    "Retrieval-Augmented Generation grounds an LLM's answer in external documents. "
    "At query time the system retrieves the most relevant chunks and passes them to the "
    "model as context, which reduces hallucination and lets you update knowledge without "
    "retraining. The two moving parts — the retriever and the generator — fail in "
    "different ways, so they are evaluated with different metrics."
)

test_cases = [
    LLMTestCase(
        input=ARTICLE,
        actual_output="RAG grounds LLM answers in retrieved documents, cutting "
                      "hallucination and enabling knowledge updates without retraining.",
    ),
    LLMTestCase(
        input=ARTICLE,
        actual_output="AI is cool. RAG is an acronym. Thanks for reading!",
    ),
]

metric = SummarizationMetric(model=judge, threshold=0.5)
results = evaluate(test_cases=test_cases, metrics=[metric])
for tr in results.test_results:
    md = tr.metrics_data[0]
    print(f"{tr.name}: score={md.score} success={md.success}")
    print(f"  reason: {md.reason[:150]}")

### 7.2 · JSON correctness against a schema

# %%
from pydantic import BaseModel
from deepeval.metrics import JsonCorrectnessMetric


class Answer(BaseModel):
    answer: str
    confidence: float


json_cases = [
    LLMTestCase(
        input="What is RAG?",
        actual_output='{"answer": "Retrieval-Augmented Generation", "confidence": 0.95}',
    ),
    LLMTestCase(
        input="What is RAG?",
        actual_output='{"answer": "Retrieval-Augmented Generation"}',  # missing field
    ),
]

json_metric = JsonCorrectnessMetric(expected_schema=Answer, model=judge, threshold=0.5)
results = evaluate(test_cases=json_cases, metrics=[json_metric])
for tr in results.test_results:
    md = tr.metrics_data[0]
    print(f"{tr.name}: score={md.score} success={md.success}")
    print(f"  reason: {md.reason[:150]}")

### 7.3 · Score-only mode: measure without verdicts

# %%
score_only = SummarizationMetric(model=judge, threshold=None)  # no gate, just a number
results = evaluate(test_cases=test_cases, metrics=[score_only])
for tr in results.test_results:
    md = tr.metrics_data[0]
    print(f"{tr.name}: score={md.score}  success={md.success}  <-- success is None")

### 7.4 · Flaky metrics: score, but never fail the case

# %%
flaky_metric = SummarizationMetric(model=judge, threshold=0.5, flaky=True)
results = evaluate(test_cases=test_cases, metrics=[flaky_metric])
for tr in results.test_results:
    md = tr.metrics_data[0]
    print(f"{tr.name}: score={md.score}  success={md.success}  <-- flaky never fails")

# %% [markdown]

## Theory

### Reference-free metrics
Summarization (and Answer Relevancy from Day 6) judge quality without a golden output. Summarization extracts the key truths of the source text and checks how many the summary preserved. That makes it usable anywhere you have source documents but no hand-written summaries — the normal state of real projects.

### Structured output checks
`JsonCorrectnessMetric` validates against a **pydantic schema**: missing fields, wrong types, and malformed JSON all fail. Under the hood it is stricter than the others (`strict_mode=True` by default) because structured output is binary — a key with the wrong type is broken, not "0.7 broken". Any LLM app that emits JSON (agents, function-calling, parsers) should have this in its pack.

### Score-only mode (`threshold=None`)
With no threshold the metric computes a score and reason but no pass/fail. Why would you want that?
- **Dashboards**: you want the number on the wall, not a gate.
- **Research**: comparing 10 prompt variants — you need scores, not verdicts.
- **Mixed suites**: gate the metrics you care about and keep the rest observational.

### Flaky metrics (`flaky=True`)
A flaky metric still scores and still gets reported — it just never decides a case's pass/fail. Use it for a judge you don't fully trust yet (cheap model, new metric) so it cannot break the build while you calibrate it. Day 27 uses this in CI.
