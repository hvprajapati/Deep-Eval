# Day 09 — Custom metrics

> **Module 2 · The metric toolkit.** Three ways to build your own scoring: deterministic rules, classic NLP overlap scores, and judge-based classes.

## Code

### 9.1 · A deterministic custom metric (no LLM needed)

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
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class MaxWordsMetric(BaseMetric):
    """Score 1.0 when the answer is short enough, sliding to 0 past the cap."""

    def __init__(self, threshold: float = 0.5, max_words: int = 40):
        self.threshold = threshold          # BaseMetric reads this for is_successful()
        self.max_words = max_words
        super().__init__()

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        n = len(test_case.actual_output.split())
        self.score = max(0.0, 1.0 - n / self.max_words)
        self.reason = f"{n} words (cap {self.max_words})."
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)


metric = MaxWordsMetric(max_words=10)
metric.measure(LLMTestCase(input="hi", actual_output="one two three four five six"))
print("score:", metric.score, "| reason:", metric.reason, "| success:", metric.is_successful())

### 9.2 · Classic overlap scores with Scorer

# %%
from deepeval.scorer import Scorer

scorer = Scorer()
print("exact_match:      ", scorer.exact_match_score(target="hello world", prediction="hello world"))
print("exact_match (off):", scorer.exact_match_score(target="hello world", prediction="hello there world"))
print("quasi_exact_match:", scorer.quasi_exact_match_score(target="hello world", prediction="Hello World!!"))
print("rougeL:           ", scorer.rouge_score(target="the cat sat on the mat",
                                               prediction="the cat sat on the rug", score_type="rougeL"))
print("sentence_bleu1:   ", scorer.sentence_bleu_score(references=["the cat sat on the mat"],
                                                       prediction="the cat sat on the mat"))

### 9.3 · A judge-based custom metric (async)

# %%
import asyncio


class DirectlyAnswersMetric(BaseMetric):
    """Ask the judge a yes/no question; turn the answer into a 0/1 score."""

    def __init__(self, threshold: float = 0.5, model=judge):
        self.threshold = threshold
        self.model = model
        super().__init__()

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        prompt = (
            "Does this answer directly address the question?\n"
            f"Question: {test_case.input}\n"
            f"Answer: {test_case.actual_output}\n"
            'Reply with only "yes" or "no".'
        )
        verdict, _cost = self.model.generate(prompt)   # (text, cost)
        self.score = 1.0 if str(verdict).strip().lower().startswith("yes") else 0.0
        self.reason = f"Judge verdict: {verdict}"
        return self.score

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return asyncio.run(self.a_measure(test_case))  # fallback for sync use


cases = [
    LLMTestCase(input="What's the refund policy?", actual_output="Refunds within 30 days."),
    LLMTestCase(input="What's the refund policy?", actual_output="I like trains."),
]
metric = DirectlyAnswersMetric()
from deepeval import evaluate

results = evaluate(test_cases=cases, metrics=[metric])
for tr in results.test_results:
    md = tr.metrics_data[0]
    print(f"{tr.name}: score={md.score} success={md.success} | {md.reason}")

# %% [markdown]

## Theory

### The BaseMetric contract
A custom metric owes DeepEval three things:

1. `measure()` / `a_measure()` — compute the score and store it in `self.score`, with `self.reason` explaining it.
2. `self.threshold` — set in `__init__`; the inherited `is_successful()` compares `score >= threshold` for you.
3. Async support — `a_measure` is what `evaluate()` calls by default; `measure` is the sync fallback.

Because verdicts come from the inherited `is_successful()`, your only real job is the scoring logic. Everything else (progress bars, aggregation, pytest integration) comes free.

### Three kinds of custom metrics
- **Deterministic** (9.1): regexes, word counts, schema checks. Free, instant, and perfectly reproducible — always prefer them when they capture the criterion.
- **Scorer-based** (9.2): classic NLP overlap (exact/quasi match, ROUGE, BLEU, SQuAD). Still no LLM, but fuzzier: good for "close enough to the golden answer" checks where wording varies.
- **Judge-based** (9.3): wrap any custom prompt around `self.model.generate(prompt)` and map the reply to a score. This is G-Eval with the training wheels off — full control over the prompt, the parsing, and the cost.

### When to graduate from G-Eval to custom
G-Eval (Day 5) is fastest to write. Custom metrics win when you need determinism (no LLM), strict parsing (yes/no gates), shared logic across many metrics, or cost control (no chain-of-thought). A common trajectory: prototype the criterion with G-Eval, promote it to a custom metric once it stabilizes.
