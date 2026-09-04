# Day 20 — Action metrics, judging single tool calls

> **Module 4 · Agents.** Component-level evaluation: score each tool decision on its own — the right tool for the intent, with the right arguments.

## Code

### 20.1 · Rebuild the three cases (from Day 19)

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

### 20.2 · Measure tool choices and arguments

# %%
from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCallParams

metrics = [
    ToolCorrectnessMetric(
        model=judge,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],  # judge the args, not just names
    ),
    ArgumentCorrectnessMetric(model=judge),
]

results = evaluate(test_cases=test_cases, metrics=metrics)
for i, tr in enumerate(results.test_results, 1):
    print(f"\n=== case {i}: {tr.input[:60]}")
    for md in tr.metrics_data:
        print(f"  {md.name:<24} score={md.score}")
        print(f"    reason: {md.reason[:150]}")

# %% [markdown]

## Theory

### Component-level vs trajectory evaluation
A trajectory metric (Day 21) scores the whole run. A component-level metric scores **one decision**: this tool call, this moment. Component metrics attach to individual spans — in production tracing setups each `@observe`d tool call can carry its own score — so a failure can be pinned to the exact call, not just "somewhere in this run".

### ToolCorrectness — the right tool for the intent
The judge reads the task and the agent's tool calls and asks: *is this the tool the task demands?* Case 2 ("run this code") should only pass if `run_code` was chosen; an agent that "answers" `2**10` from memory gets the output right but fails ToolCorrectness — a behavioral bug no output metric can see. `evaluation_params=[ToolCallParams.INPUT_PARAMETERS]` widens the check from tool names to the arguments passed, catching right-tool-wrong-args.

### ArgumentCorrectness — the right arguments for that tool
Same decisions, different question: *given the tool it chose, were the arguments correct?* An agent calling `escalate_to_human(reason="idk")` chose the right tool with a useless argument. ToolCorrectness passes; ArgumentCorrectness fails. The two metrics answer different questions about the same span — run both.

### Actions before outcomes
Both metrics deliberately ignore the final answer. A correct outcome reached through wrong actions is a bug waiting for a slightly different input: the path *is* the product when the path is tool calls. Days 21–22 scale this idea up from single calls to the whole trajectory.
