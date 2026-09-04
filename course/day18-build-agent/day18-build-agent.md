# Day 18 — Build a tool-using agent

> **Module 4 · Agents.** The app under test for the rest of the module: three tools, one LangChain agent, and a deliberate flaw we will catch with metrics — all built in this notebook.

## Code

### 18.1 · Load the environment

# %%
import os

from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")  # silence Confident AI dev logging

### 18.2 · The three tools (every tool call is a trace span)

# %%
import random

from langchain_core.tools import tool

from deepeval.tracing import observe


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

### 18.3 · The agent: system prompt + task cases

# %%
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

### 18.4 · Build the agent (app model, not the judge)

# %%
from langchain.agents import create_agent
from langchain_groq import ChatGroq


def build_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"], temperature=0)
    return create_agent(llm, tools=TOOLS, system_prompt=AGENT_SYSTEM)

### 18.5 · Run it under a trace and capture everything

# %%
import time

from deepeval.tracing import trace, trace_manager, update_current_span


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

### 18.6 · Try the agent raw (no evals yet)

# %%
agent = build_agent()
for c in CASES:
    result = run_agent(agent, c["input"])
    print("Q:", c["input"])
    print("Tools called:", result["tool_names"], "(expected:", c["expected_tools"], ")")
    print("Answer:", result["actual_output"][:120], "...")
    print("-" * 80)

# %% [markdown]

## Theory

### The agent loop
An agent is an LLM with an action space: **plan → act (call a tool) → observe (read the result) → decide → repeat**. The tool schemas *are* the action space — the model chooses tools by name and description and fills their arguments. Everything an agent does is a sequence of tool calls, which is why traces (Day 17) and tool-level metrics (Days 20–21) are the right evaluation instruments.

### Tool design is prompt design
The three tools here are stubs — but look at what they model: documentation lookup (`search_docs`), execution (`run_code`), and an escape hatch (`escalate_to_human`). A real agent's quality depends on tool *descriptions*: the model reads nothing else. Vague descriptions produce wrong tool choices — which is exactly what `ToolCorrectnessMetric` will catch on Day 20.

### The deliberate flaw
The system prompt says "use the tools", but nothing limits *how many times* an agent may call a tool. On the first case, the agent often calls `search_docs` again and again instead of answering. Day 22 measures that loop with `StepEfficiencyMetric` and fixes it — for now, just watch it happen.

### Why the agent lives in cells, not a file
Days 19–22 each start with a recap cell that rebuilds these same tools and this runner. By the time you fix the loop bug on Day 22, every line is yours — the harness is a template you can carry to any real agent.

### Frameworks
LangChain's `create_agent` is convenient for teaching, but DeepEval is framework-agnostic: LangGraph, Pydantic AI, CrewAI — anything you can wrap in `@observe` and run under `trace()` is evaluable. Day 24 assembles a two-agent system the same way.
