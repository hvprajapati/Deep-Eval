# Day 27 — Evals in production

> **Module 5 · Production.** Datasets that round-trip, hyperparameters that travel with every score, and CI that refuses the merge when the pack fails.

## Code

### 27.1 · Export and re-import the pack as JSON

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
import json

from deepeval.test_case import LLMTestCase

pack = [
    LLMTestCase(
        input="What's your refund policy?",
        actual_output="We offer full refunds within 30 days of purchase.",
        expected_output="Refunds are available within 30 days of purchase.",
    ),
    LLMTestCase(
        input="How long does shipping take?",
        actual_output="Orders ship within 2 business days.",
        expected_output="Orders ship within 2 business days.",
    ),
]

# export
with open("golden_pack.json", "w") as f:
    json.dump([tc.model_dump(mode="json") for tc in pack], f, indent=2)
print("exported", len(pack), "cases")

# import (a different day, a different machine — same pack)
with open("golden_pack.json") as f:
    restored = [LLMTestCase(**row) for row in json.load(f)]
print("restored", len(restored), "cases:", restored[0].input)

### 27.2 · Log hyperparameters with every run

# %%
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

results = evaluate(
    test_cases=restored,
    metrics=[AnswerRelevancyMetric(model=judge)],
    hyperparameters={
        "model": "openai/gpt-oss-120b",
        "prompt_template": "You are a polite shop assistant.",
        "temperature": 0,
        "retriever": "tfidf-v2",
    },
)
print("run logged with hyperparameters:", results.test_results[0].success)

### 27.3 · A CI job that gates on the pack

# %%
# Write the CI workflow into course/.github/workflows/evals.yml — the place
# GitHub Actions looks when this repo is pushed. Jupyter's working directory is
# the folder it was launched from (course/), so ".github" resolves to course/.github.
# The GROQ_API_KEY secret is set in the repository settings, never in the file.
from pathlib import Path

workflows_dir = Path(".github") / "workflows"
workflows_dir.mkdir(parents=True, exist_ok=True)

yml = '''name: evals
on: [pull_request]
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - name: Run the eval pack
        run: deepeval test run tests/
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          DEEPEVAL_TELEMETRY_OPT_OUT: "YES"
'''

(workflows_dir / "evals.yml").write_text(yml, encoding="utf-8")
print("wrote", workflows_dir / "evals.yml")

# %% [markdown]

## Theory

### Datasets are contracts, so they must round-trip
`model_dump(mode="json")` → `json.dump` → `json.load` → `LLMTestCase(**row)` loses nothing. That loop is how packs move between machines, git repos, and teammates. On the Confident AI platform (Day 28) the same pack becomes a *dataset* with versions, shared across the team — but the local JSON loop is the lowest common denominator, and every team needs it working.

### Hyperparameters travel with scores
A score of 0.8 means nothing without its context: which model, which prompt, which retriever, which temperature. `evaluate(hyperparameters={...})` stamps every run with that context, so a score from last month and a score from today can be compared honestly. (DeepEval also offers a `@log_hyperparameters` decorator for capturing this from the function that builds the test cases.) Rule of thumb: if two runs differ in score but you can't see what differed in config, the comparison is noise.

### CI gates with LLM judges
The workflow above runs `deepeval test run tests/` on every PR — the Day 8 suite, plus whatever packs the team has added since. Three realities of judging in CI:

1. **Cost** — every run spends judge tokens. Curate the pack (small, high-signal), and run the full suite on merges, not every commit.
2. **Flakiness** — a judge can disagree with itself. Metrics marked `flaky=True` (Day 7) score but never fail the build; promote them to gating once they prove stable.
3. **Secrets** — the judge key lives in GitHub secrets, never in the repo.

### The production loop, complete
Build → evaluate (local) → evaluate (CI gate) → ship → watch traces → new failures become new golden cases → the pack grows. That is the whole course in one sentence. Day 28 assembles the capstone version of it.
