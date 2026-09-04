# Day 08 — Evals as tests: pytest + deepeval test run

> **Module 2 · The metric toolkit.** Turn eval packs into real pytest suites that a CI server can run and gate on.

## Code

### 8.1 · Load the environment (the test runner inherits the key from here)

# %%
from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder

### 8.2 · Write a pytest-based eval file

# %%
# The test file lives in course/tests/ (created by this cell), so the CI runner
# in Day 27 can find it. Jupyter's working directory is the folder it was
# launched from — the course/ folder — so "tests" resolves to course/tests.
from pathlib import Path

tests_dir = Path("tests")
tests_dir.mkdir(exist_ok=True)

test_file = '''"""Evals as pytest tests. Run with: deepeval test run tests/"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # course/.env — or GROQ_API_KEY arrives from CI secrets (Day 27)

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

# The course judge, defined here so this suite is self-contained.
judge = LocalModel(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

GOLDEN = [
    {"input": "What's your refund policy?",
     "actual_output": "We offer full refunds within 30 days of purchase.",
     "expected_output": "Refunds are available within 30 days of purchase."},
    {"input": "How long does shipping take?",
     "actual_output": "Orders ship within 2 business days.",
     "expected_output": "Orders ship within 2 business days."},
]


@pytest.fixture(scope="module")
def relevancy():
    return AnswerRelevancyMetric(model=judge, threshold=0.5)


@pytest.mark.parametrize(
    "case",
    GOLDEN,
    ids=[c["input"] for c in GOLDEN],
)
def test_relevant_answers(case, relevancy):
    test_case = LLMTestCase(**case)
    assert_test(test_case, metrics=[relevancy])


def test_professional_tone():
    tone = GEval(
        name="Professional Tone",
        criteria="Is the response polite and professional?",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="What's your refund policy?",
        actual_output="No refunds. Read the policy page next time.",
    )
    assert_test(test_case, metrics=[tone])
'''

(tests_dir / "test_day08.py").write_text(test_file, encoding="utf-8")
print("wrote", tests_dir / "test_day08.py")

### 8.3 · Run the suite from the CLI

# %%
# The `deepeval` CLI discovers test_*.py files and runs them under pytest.
# (Equivalent: run `deepeval test run tests/` in a terminal, from course/.)
import subprocess

subprocess.run(["deepeval", "test", "run", "tests"])

# %% [markdown]

## Theory

### Why evals belong in CI
`deepeval test run` is pytest wearing an eval coat: normal discovery, normal fixtures, normal parametrize — but each "assert" is a judge scoring an LLM output against a threshold. Once the suite is a CLI command, it plugs into GitHub Actions like any test suite (Day 27 does exactly that). The contract: *every prompt, model, or pipeline change must pass the pack before merge*.

### Fixtures keep judges cheap and consistent
`@pytest.fixture(scope="module")` builds the judge once per test session instead of per test. Judge objects are stateless and safe to share — sharing them avoids re-initialization overhead and guarantees every test used the same judge configuration.

### Test hygiene for LLM suites
- **Deterministic inputs only** — golden cases, not random generations.
- **Small packs** — each assert is a paid judge call; 5 cases x 2 metrics x every PR adds up. Curate.
- **Readable ids** — `ids=[...]` names each parametrized case by its question, so a red dot in CI tells you *which question* broke.
- **`assert_test` runs metrics asynchronously by default** — fine for small packs; add `run_async=False` only when a metric requires it.

### The day's takeaway
Everything since Day 4 was script-style (`evaluate(...)`). From today the same metrics live in test files, so quality is enforced by the same machinery that enforces correctness — the build.
