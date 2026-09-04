# Day 22 — The full agent report

> **Module 4 · Agents.** All six agent metrics on all three tasks — then the payoff: fix the loop bug and watch the scores move.

## Code

### 22.1 · Run the full six-metric harness

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


from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

agent = build_agent()
agent_rows = []
for c in CASES:
    result = run_agent(agent, c["input"])
    agent_rows.append({**c, **result})
    print("case:", c["input"][:50], "->", result["tool_names"])

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

metrics = [
    ToolCorrectnessMetric(model=judge, evaluation_params=[ToolCallParams.INPUT_PARAMETERS]),
    ArgumentCorrectnessMetric(model=judge),
    StepEfficiencyMetric(model=judge),
    PlanAdherenceMetric(model=judge),
    PlanQualityMetric(model=judge),
    TaskCompletionMetric(model=judge),
]

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=2, throttle_value=1.0),
    error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
)

for i, tr in enumerate(results.test_results, 1):
    print(f"\n=== case {i}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"  {md.name:<20} {md.score}")

### 22.2 · Diagnose the loop bug, then fix it

# %%
import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq

FIXED_SYSTEM = (
    "You are a GenAI developer assistant. Use the available tools to fulfil the user's request. "
    "Call search_docs AT MOST TWICE per question, then answer from what you found; if the docs "
    "do not explain the issue, escalate to a human. Never repeat the same tool call."
)
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0)
fixed_agent = create_agent(llm, tools=TOOLS, system_prompt=FIXED_SYSTEM)

# Re-run ONLY the failing case through the fixed agent
from deepeval.test_case import LLMTestCase, ToolCall

row = run_agent(fixed_agent, CASES[0]["input"])
print("tools now:", row["tool_names"])

fixed_case = LLMTestCase(
    input=CASES[0]["input"],
    actual_output=row["actual_output"],
    tools_called=row["tools_called"],
    expected_tools=[ToolCall(name="search_docs")],
)
fixed_case._trace_dict = row["trace"]

### 22.3 · Re-measure and diff

# %%
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import StepEfficiencyMetric, TaskCompletionMetric

metrics = [StepEfficiencyMetric(model=judge), TaskCompletionMetric(model=judge)]
results = evaluate(
    test_cases=[fixed_case],
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=2, throttle_value=1.0),
    error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
)
for md in results.test_results[0].metrics_data:
    print(f"{md.name:<20} score={md.score}   (Day 22.1 showed the before-numbers)")
    print(f"  reason: {md.reason[:150]}")

# %% [markdown]

## Theory

### Reading an agent failure report
Six metrics, three tasks — read them like this:

1. **Which case failed?** → open its trace.
2. **Which metrics failed on it?** → the metric names localize the failure:
   - Tool/Argument Correctness → bad individual decisions (Day 20 territory).
   - Plan metrics → bad planning or planning ignored.
   - StepEfficiency alone → the plan was fine, the execution was wasteful — the classic loop bug.
3. **Read the judge reasons** — they name the exact calls.

### The loop bug, and why prompts fix it
The agent repeated `search_docs` because nothing told it to stop: the task was answerable, but each response gave the model an excuse to "verify" again. The fix was not a code change — it was a **behavioral contract in the system prompt** ("at most twice, then answer or escalate"). This is the most important lesson of agent evaluation: most agent bugs are prompt-level policy bugs, and the metrics tell you *which* policy is missing.

### The harness is a template
Nothing in 22.1 is specific to these three tools. Swap `build_agent()` for your own agent, `CASES` for your task suite, and the six metrics run unchanged. From here on, "improving the agent" means: change one thing, re-run 22.1, diff the table. The harness is the loop, and the loop is the product.
