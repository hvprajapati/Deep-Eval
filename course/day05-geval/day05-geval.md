# Day 05 — G-Eval, metrics you write in English

> **Module 1 · Foundations.** Define metrics with plain-language criteria instead of code. By the end of today you can score anything you can describe.

## Code

### 5.1 · The refund-policy cases (one good, one bad)

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

test_cases = [
    LLMTestCase(
        input="What's your refund policy?",
        actual_output=(
            "Happy to help! We offer full refunds within 30 days of purchase, no questions "
            "asked. Just reply here with your order number."
        ),
        expected_output="Refunds are available within 30 days of purchase.",
    ),
    LLMTestCase(
        input="What's your refund policy?",
        actual_output="No refunds. Read the policy page next time.",
        expected_output="Refunds are available within 30 days of purchase.",
    ),
]

### 5.2 · A steps-based G-Eval (tone)

# %%
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams

tone = GEval(
    name="Professional Tone",
    evaluation_steps=[
        "Check whether the response is polite and professional.",
        "Penalize sarcasm, rudeness, or dismissive language.",
        "Reward clear, respectful phrasing even if brief.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge,
)

### 5.3 · A rubric-based G-Eval (correctness)

# %%
from deepeval.metrics.g_eval import Rubric

correctness = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct given the expected output.",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually wrong or contradicts the expected output."),
        Rubric(score_range=(3, 5), expected_outcome="Partially correct -- misses or garbles a key fact."),
        Rubric(score_range=(6, 8), expected_outcome="Mostly correct with only minor omissions."),
        Rubric(score_range=(9, 10), expected_outcome="Fully correct and complete."),
    ],
    model=judge,
)

### 5.4 · Run both side by side

# %%
from deepeval import evaluate

results = evaluate(test_cases=test_cases, metrics=[correctness, tone])
for tr in results.test_results:
    print(f"\n=== {tr.name}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"  {md.name:<22} score={md.score}")
        print(f"    reason: {md.reason[:140]}")

# %% [markdown]

## Theory

### Generation-based evaluation
G-Eval asks the judge to *generate* a chain-of-thought evaluation and finish with a score. That is why it is the most flexible metric in DeepEval: the "metric" is a natural-language description of quality, compiled into a judge prompt. Anything you can write down — tone, formatting, empathy, brand voice — becomes a metric in minutes.

### The three dials
- **`criteria`** — a one-line definition of the quality being measured.
- **`evaluation_steps`** — the checklist the judge must walk through before scoring. Steps make the score *explainable* and *stable*: two judges following the same steps agree more often.
- **`rubric`** — score bands (0–10 internally) with a described outcome per band. Rubrics give graded scores instead of a rough 0–1, useful when "partially correct" is a real grade.
- **`evaluation_params`** (`SingleTurnParams.*`) — which test-case fields the judge may look at. Restricting scope (e.g. tone ignores `expected_output`) keeps the judge focused.

### When G-Eval is the right tool
- You have a quality dimension with no built-in metric (brand voice, empathy, humor).
- You need a metric *today* and can refine it later.
- The alternative — writing a custom metric class — buys speed and full control (Day 9), but G-Eval buys speed of authoring. Start with G-Eval; graduate to custom when you need determinism or cost control.

### `strict_mode` (preview for later)
`strict_mode=True` forces the judge to only return scores 0 or 1 — pass or fail, no gray zone. Useful for gate-style checks ("did it refuse the medical question?") where a 0.7 is meaningless. Safety metrics (Day 25) are where strictness shines.
