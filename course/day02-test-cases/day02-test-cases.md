# Day 02 — LLMTestCase, the unit of evaluation

> **Module 1 · Foundations.** Everything in DeepEval flows through one object: the test case. Build it by hand today; every later day builds it from a live app.

## Code

### 2.1 · Load the environment (every day starts here)

# %%
from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder

### 2.2 · Your first test case

# %%
from deepeval.test_case import LLMTestCase

tc = LLMTestCase(
    input="What's your refund policy?",
    actual_output="We offer full refunds within 30 days of purchase.",
    expected_output="Refunds are available within 30 days of purchase.",
)
print("input:   ", tc.input)
print("actual:  ", tc.actual_output)
print("expected:", tc.expected_output)

### 2.3 · The same case with context fields

# %%
grounded = LLMTestCase(
    input="What's your refund policy?",
    actual_output="We offer full refunds within 30 days of purchase.",
    expected_output="Refunds are available within 30 days of purchase.",
    context=["Customer support policy page: full refunds within 30 days."],
    retrieval_context=[
        "Doc 1: Refunds are available within 30 days of purchase.",
        "Doc 2: Refund requests are processed within 3 business days.",
    ],
)
print("context:           ", grounded.context)
print("retrieval_context: ", grounded.retrieval_context)

### 2.4 · Load a golden dataset from JSON

# %%
import json

GOLDEN = [
    {"input": "What's your refund policy?",
     "expected_output": "Refunds are available within 30 days of purchase."},
    {"input": "How long does shipping take?",
     "expected_output": "Orders ship within 2 business days."},
    {"input": "Can I change my delivery address?",
     "expected_output": "Addresses can be changed before the order ships."},
]

test_cases = [LLMTestCase(**row) for row in GOLDEN]
print("Loaded", len(test_cases), "golden cases:")
for t in test_cases:
    print("-", t.input, "→", t.expected_output)

# %% [markdown]

## Theory

### Which field feeds which metric
| fields consumed | metrics | what it checks |
|---|---|---|
| `input` + `actual_output` | Answer Relevancy, Summarization | did the answer actually answer the question? |
| `actual_output` + `retrieval_context` | Faithfulness, Hallucination | is the answer supported by what was retrieved? |
| `actual_output` + `expected_output` | G-Eval Correctness, Scorer overlap scores | does it match the golden answer? |
| `input` + `actual_output` + `retrieval_context` | Contextual Precision | are the retrieved docs *ranked* usefully? |
| `expected_output` + `retrieval_context` | Contextual Recall | do the docs contain everything the golden answer needs? |
| `input` + `actual_output` + `tools_called` + trace | agent metrics (Days 17–22) | did it take the right actions? |

A test case only needs the fields its metrics consume. That is the flexibility — and the trap: a metric that errors or gets skipped on a missing field is the framework telling you the case is under-specified. Day 15 shows the `skip_on_missing_params` safety valve.

### Golden datasets
A golden dataset is a curated, version-controlled list of input/expected pairs. Rules that survive contact with production:

- write the first cases by hand, expand later with synthetic generation (Day 16);
- they live in the repo next to the app — never in someone's head;
- every prompt or model change re-runs the pack (Days 8 and 27).

### When `expected_output` is optional
Reference-free metrics (Summarization, Answer Relevancy, Faithfulness) never look at `expected_output` — the judge decides quality from the question and the context alone. Golden answers matter for reference-based metrics (Correctness G-Eval, Scorer overlap scores). Knowing which is which is what lets you evaluate apps where no single "right answer" exists.
