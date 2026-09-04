# Day 21 — Trajectory metrics, judging the whole run

> **Module 4 · Agents.** Zoom out from single tool calls to the complete ordered chain of decisions: did it finish, was the plan good, was it followed, was it efficient?

## Code

### 21.1 · Rebuild the three traced cases (Days 19–20 pattern)

# %%
# --- setup (from Day 03): load course/.env and recreate the course judge ---
import os

from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")

from deepeval.models import LocalModel

judge = LocalModel(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",  # Groq speaks the OpenAI protocol
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)


# %%
from deepeval.test_case import LLMTestCase, ToolCall

# --- agent recap (from Day 18): the three tools, the agent, and the traced runner ---
import random
import time

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from deepeval.tracing import observe, trace, trace_manager, update_current_span


@tool
@observe(type="tool")
def search_docs(query: str) -> str:
    """Search internal documentation for a concept or policy."""
    return f"Docs result for '{query}': found a section with a definition and a short example."


@tool
@observe(type="tool")
def run_code(code: str) -> str:
    """Execute a Python snippet in a sandbox and return its stdout."""
    if "2**10" in code:
        return "Executed in a sandbox. stdout: 1024"
    return "Executed in a sandbox. stdout: (result)"


@tool
@observe(type="tool")
def escalate_to_human(reason: str) -> str:
    """Escalate the issue to a human engineer."""
    return f"Escalated to a human engineer. Ticket AI-{random.randint(1000, 9999)}. Reason: {reason}"


TOOLS = [search_docs, run_code, escalate_to_human]

AGENT_SYSTEM = (
    "You are a GenAI developer assistant. Use the available tools to fulfil the user's request. "
    "Call the tool(s) that actually accomplish the task -- do not guess something you can look up "
    "or run."
)

CASES = [
    {"input": "What do the docs say about the difference between fine-tuning and prompting?",
     "expected_tools": ["search_docs"]},
    {"input": "Run this and tell me the output: print(2**10)",
     "expected_tools": ["run_code"]},
    {"input": "My fine-tuning job keeps failing silently. Check the docs for known issues, and if "
              "that doesn't explain it, get a human to look at it.",
     "expected_tools": ["search_docs", "escalate_to_human"]},
]


def build_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0)
    return create_agent(llm, tools=TOOLS, system_prompt=AGENT_SYSTEM)


@observe(type="agent")
def _invoke_agent(agent, query: str, max_retries: int = 3) -> str:
    from groq import BadRequestError

    for attempt in range(max_retries):
        try:
            state = agent.invoke({"messages": [{"role": "user", "content": query}]})
            break
        except BadRequestError as e:
            code = (getattr(e, "body", None) or {}).get("error", {}).get("code")
            if code != "tool_use_failed" or attempt == max_retries - 1:
                raise
            time.sleep(1)

    final_output = state["messages"][-1].content.strip()
    update_current_span(input=query, output=final_output)
    return final_output


def run_agent(agent, query: str) -> dict:
    """Run the agent under a trace and return output + tools + the trace dict."""
    with trace() as current_trace:
        actual_output = _invoke_agent(agent, query)
    root_span = current_trace.root_spans[0]
    tools_called = root_span.tools_called or []
    return {
        "actual_output": actual_output,
        "tools_called": tools_called,
        "tool_names": [tc.name for tc in tools_called],
        "trace": trace_manager.create_nested_spans_dict(root_span),
    }


agent = build_agent()
agent_rows = []
for c in CASES:
    result = run_agent(agent, c["input"])
    agent_rows.append({**c, **result})

test_cases = [
    LLMTestCase(
        input=r["input"],
        actual_output=r["actual_output"],
        tools_called=r["tools_called"],
        expected_tools=[ToolCall(name=n) for n in r["expected_tools"]],
    )
    for r in agent_rows
]
for test_case, r in zip(test_cases, agent_rows):
    test_case._trace_dict = r["trace"]

### 21.2 · Measure the four trajectory metrics

# %%
from deepeval import evaluate
from deepeval.metrics import (
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
)

metrics = [
    TaskCompletionMetric(model=judge),
    PlanQualityMetric(model=judge),
    PlanAdherenceMetric(model=judge),
    StepEfficiencyMetric(model=judge),
]

results = evaluate(test_cases=test_cases, metrics=metrics)
for i, tr in enumerate(results.test_results, 1):
    print(f"\n=== case {i}: {tr.input[:60]}")
    for md in tr.metrics_data:
        print(f"  {md.name:<20} score={md.score}")
        print(f"    reason: {md.reason[:150]}")

# %% [markdown]

## Theory

### What a trajectory is
A trajectory is the **ordered chain of decisions** — every tool call, its arguments, and the reasoning between them, read from the trace. An agent that calls the right tools in the wrong order has a bad trajectory and a possibly good answer. Trajectory metrics read the `_trace_dict` attached to each case (Day 19) and judge the chain, not just the outcome.

### The four metrics, one lens each
- **TaskCompletion** — did the goal get reached? The destination.
- **PlanQuality** — was the *plan* (before any tool call) a good one for this task? The roadmap.
- **PlanAdherence** — did the run stick to its plan? The discipline.
- **StepEfficiency** — were the steps economical — no repeats, no detours? The cost.

An agent can pass Completion and fail all three path metrics; it can also have a beautiful plan it abandoned midway. Reading all four together separates *what* went wrong from *where* in the run it went wrong.

### The loop case, through these lenses
Case 1's agent called `search_docs` nine times for one question. Expect the verdicts to split: TaskCompletion high (it eventually produced a fine answer), PlanQuality decent, StepEfficiency **low** (nine calls where one would do). That split is the diagnostic signature of a loop bug — and Day 22 fixes it and re-measures.

### Trajectory vs action metrics
Days 20 and 21 are complementary, not rivals: action metrics (Tool/Argument Correctness) judge individual calls in isolation; trajectory metrics judge the sequence they form. Real agent suites run both — actions catch bad decisions, trajectories catch bad *orderings* and waste.
