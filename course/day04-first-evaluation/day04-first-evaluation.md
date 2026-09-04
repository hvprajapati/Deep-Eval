# Day 04 — Running your first evaluation

> **Module 1 · Foundations.** The eval loop becomes real: dataset + metric + threshold + verdict — printed as a report, then parsed as Python objects.

## Code

### 4.1 · One metric, two cases

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
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(model=judge, threshold=0.5)

test_cases = [
    LLMTestCase(
        input="What's your refund policy?",
        actual_output="We offer full refunds within 30 days of purchase, no questions asked.",
    ),
    LLMTestCase(
        input="What's your refund policy?",
        actual_output="Our mascot is a llama. Llamas are great. Anyway, shipping takes 2 days.",
    ),
]

### 4.2 · Run evaluate()

# %%
from deepeval import evaluate

results = evaluate(test_cases=test_cases, metrics=[metric])

### 4.3 · Read the results as data

# %%
for tr in results.test_results:
    print(f"\n=== {tr.name}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"  {md.name:<24} score={md.score}  success={md.success}")
        print(f"    reason: {md.reason[:120]}")

### 4.4 · Move the threshold and watch verdicts flip

# %%
metric.threshold = 0.8
results = evaluate(test_cases=test_cases, metrics=[metric])
for tr in results.test_results:
    print(tr.name, "success =", tr.success, "| score =", tr.metrics_data[0].score)

# %% [markdown]

## Theory

### Score, reason, verdict
Every DeepEval metric returns three things:

- **score** — a float, 0 to 1.
- **reason** — the judge's written justification.
- **verdict** — pass/fail: `score >= threshold`.

The reason is the most valuable of the three: it is the judge telling you *what* is wrong. Build the habit — never fix a failing case without reading the reason first.

### Thresholds are product decisions
0.5 is a default, not a truth. A medical triage bot wants answer relevancy at 0.9; a meme generator can ship at 0.4. The threshold encodes how much failure your use case tolerates, and it is *yours* to set, per metric, per product.

### The results objects
- `evaluate(...)` → `EvaluationResult` with `.test_results` (one `TestResult` per case).
- `TestResult` → `.name`, `.success`, `.metrics_data`, `.input`, `.actual_output`.
- `MetricData` → `.name`, `.score`, `.reason`, `.success`, plus cost and token fields.

These are plain objects: filter them, average them, write them into reports and CI checks. Day 8 puts this entire flow inside pytest, and Day 27 puts pytest inside GitHub Actions.
