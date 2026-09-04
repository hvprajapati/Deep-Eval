# Day 24 — Multi-agent systems

> **Module 4 · Agents.** A supervisor routes to specialist workers. Evaluate the whole team end-to-end — then attribute blame per agent via traces.

## Code

### 24.1 · Two specialists and a supervisor

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
import os

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from deepeval.tracing import observe

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0)


@tool
@observe(type="tool")
def lookup_docs(query: str) -> str:
    """Search the documentation library for a concept or API."""
    return f"Docs: '{query}' is explained with two examples in the developer guide."


@tool
@observe(type="tool")
def run_python(code: str) -> str:
    """Execute Python in a sandbox and return stdout."""
    return "stdout: 42" if "6*7" in code else "stdout: (result)"


docs_agent = create_agent(
    llm, tools=[lookup_docs],
    system_prompt="You are the documentation specialist. Only answer from tool results.",
)
code_agent = create_agent(
    llm, tools=[run_python],
    system_prompt="You are the coding specialist. Run code rather than guessing its output.",
)

SUPERVISOR_PROMPT = (
    "Classify the user request. Reply with exactly one word: 'docs' or 'code'."
)


def classify(query: str) -> str:
    reply = llm.invoke([("system", SUPERVISOR_PROMPT), ("user", query)])
    word = reply.content.strip().lower()
    return "docs" if word.startswith("docs") else "code"


### 24.2 · Run the team under a trace

# %%
import json

from deepeval.tracing import observe, trace, trace_manager


@observe(type="agent")
def run_team(query: str) -> dict:
    kind = classify(query)  # supervisor decision (no tools)
    worker = docs_agent if kind == "docs" else code_agent
    state = worker.invoke({"messages": [{"role": "user", "content": query}]})
    return {"actual_output": state["messages"][-1].content.strip(), "route": kind}


CASES = [
    {"input": "What do the docs say about caching embeddings?",
     "expected_tools": ["lookup_docs"]},
    {"input": "Run print(6*7) and tell me the output",
     "expected_tools": ["run_python"]},
]

rows = []
for c in CASES:
    with trace() as current_trace:
        res = run_team(c["input"])
    root = current_trace.root_spans[0]
    rows.append({**c, **res,
                 "tools_called": root.tools_called or [],
                 "trace": trace_manager.create_nested_spans_dict(root)})
    print("route:", res["route"], "| tools:", [t.name for t in rows[-1]["tools_called"]])
    print("answer:", res["actual_output"][:100])
    print("-" * 80)

### 24.3 · Evaluate the team

# %%
from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

test_cases = [
    LLMTestCase(
        input=r["input"],
        actual_output=r["actual_output"],
        tools_called=r["tools_called"],
        expected_tools=[ToolCall(name=n) for n in r["expected_tools"]],
    )
    for r in rows
]
for tc, r in zip(test_cases, rows):
    tc._trace_dict = r["trace"]

metrics = [TaskCompletionMetric(model=judge), ToolCorrectnessMetric(model=judge)]
results = evaluate(test_cases=test_cases, metrics=metrics)
for tr in results.test_results:
    print(f"\n=== {tr.name}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"  {md.name:<20} score={md.score}")
        print(f"    reason: {md.reason[:140]}")

# %% [markdown]

## Theory

### What multi-agent adds — and breaks
Splitting one agent into specialists buys focus (each has fewer tools, shorter prompts, fewer mistakes) and parallelism — at the cost of a new failure surface: **the handoffs**. The supervisor can route wrong; a worker can answer a task meant for the other; context can be lost at the boundary. Evaluating a multi-agent system means evaluating the handoffs, not just the workers.

### Two levels of evaluation
- **End-to-end** (24.3): TaskCompletion and ToolCorrectness on the whole run — is the *team* behaving? This catches routing errors: "what do the docs say" must reach `docs_agent` and call `lookup_docs`.
- **Per-agent isolation**: evaluate each worker against its own task suite (its tools, its expected calls) exactly as Day 22 did for a single agent. A worker that passes alone but the team fails means the *handoff* is the bug — and that is a supervisor problem, not a worker problem.

### Blame attribution via traces
The trace tree shows which agent owned which span: `run_team` at the root, the worker's LLM and tool calls below. When the team fails, read the trace before reading the prompts — the failing span names the failing agent. Component metrics (Day 20) attach to individual spans in production tracing, so a misbehaving sub-agent can be scored continuously, not just on golden days.

### Scaling up
This day's supervisor is a one-word classifier — deliberately small. Real systems replace it with LangGraph state machines, CrewAI crews, or fully traced orchestrators. The evaluation layer does not change: run under `trace()`, capture `tools_called` + the trace dict, score with the same six metrics, per agent and end-to-end.
