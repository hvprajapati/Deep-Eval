# Day 10 — DAG metrics, decision-tree evaluation

> **Module 2 · The metric toolkit.** Some criteria are procedures, not prose. DAG metrics encode branching logic: extract → gate → score.

## Code

### 10.1 · Build the tree top-down

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
from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    DeepAcyclicGraph,
    NonBinaryJudgementNode,
    TaskNode,
)
from deepeval.test_case import SingleTurnParams

# Leaf: are the headings in the right order?
order = NonBinaryJudgementNode(
    criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
)
order.add_verdict("Yes", score=10)
order.add_verdict("Two are out of order", score=4)
order.add_verdict("All out of order", score=2)

# Gate: do all three headings exist at all?
headings = BinaryJudgementNode(
    criteria="Do the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
)
headings.add_verdict(True, then=order)  # pass -> continue into the order check
headings.add_verdict(False, score=0)    # fail -> stop here, score 0

# Root: pull the headings out of the output first
extract = TaskNode(
    instructions="Extract all headings in `actual_output`",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    output_label="Summary headings",
)
extract.add_node(headings)

dag = DeepAcyclicGraph(root_nodes=[extract])
metric = DAGMetric(name="Format Correctness", dag=dag, model=judge, async_mode=False)

### 10.2 · Two meeting summaries — one good, one missing a section

# %%
from deepeval.test_case import LLMTestCase

meeting = (
    'Alice: "Today\'s agenda: product update, blockers, marketing timeline. Bob, updates?"\n'
    'Bob: "Core features done; optimizing performance. Fixes by Friday."\n'
    'Alice: "Plan: fixes by Friday, sync next Wednesday. Thanks, everyone!"'
)

test_cases = [
    LLMTestCase(
        input=meeting,
        actual_output=(
            "Intro:\nAlice outlined the agenda.\n\nBody:\nBob reported performance "
            "optimizations, fixes expected by Friday.\n\nConclusion:\nThe team aligned "
            "on fixes by Friday and a sync next Wednesday."
        ),
    ),
    LLMTestCase(
        input=meeting,
        actual_output=(
            "Intro:\nAlice outlined the agenda.\n\nBody:\nBob reported fixes by Friday."
        ),  # no Conclusion -> the gate fails and the order check is skipped
    ),
]

### 10.3 · Run and inspect the trace of decisions

# %%
for i, tc in enumerate(test_cases, 1):
    metric = DAGMetric(name="Format Correctness", dag=dag, model=judge, async_mode=False)
    metric.measure(tc)
    print(f"=== case {i}: score={metric.score:.2f} success={metric.is_successful()}")
    print(metric.verbose_logs)
    print()

# %% [markdown]

## Theory

### When criteria are a procedure, not prose
G-Eval hands the judge a description and hopes. A DAG hands the judge a *program*:

- **TaskNode** — "extract the headings" (a task the judge performs, like a tiny prompt).
- **BinaryJudgementNode** — a yes/no gate; each `add_verdict(True/False)` attaches a score or continues down a branch (`then=`).
- **NonBinaryJudgementNode** — a multiple-choice gate with string verdicts ("Yes" / "All out of order").
- **VerdictNode** — the leaves, carrying raw scores.

In the tree above, the wrong order can never be judged before we know the headings exist — the gate *controls the path*. That is the difference from G-Eval: structure, not persuasion.

### How scores flow
Each run walks one path from root to a leaf verdict; the leaf's raw score (0–10 here) is normalized to 0–1 by `DAGMetric`, then compared to the threshold. `verbose_logs` prints the full walk — every node, every judge verdict, every branch taken. Read it like a traceback for quality.

### DAG vs G-Eval
- **DAG** — objective or mixed criteria with explicit branching (formats, checklists, conditional checks). Deterministic where it branches; only the judgement nodes call the judge.
- **G-Eval** — open-ended quality dimensions (tone, style, correctness) that resist being encoded as a tree.

Most suites use both: G-Eval for the fuzzy, DAG for the structural.

> **Notebook note:** `notebook.ipynb` builds its DAG with the older bottom-up `children=[...]` style and carries two monkey-patches for bugs in that DeepEval version. DeepEval 4.2.1 needs neither the patches nor the bottom-up style — this day uses the current top-down API.
