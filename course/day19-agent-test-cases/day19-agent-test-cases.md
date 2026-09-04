# Day 19 — Traces → agent test cases

> **Module 4 · Agents.** Convert raw agent runs into `LLMTestCase`s: output, tool calls, and the full trace attached. This is the harness every agent metric runs on.

## Code

### 19.1 · Run the three tasks and capture everything

# %%
# --- setup: load course/.env (the agent reads GROQ_API_KEY from it) ---
import os

from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")


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


agent = build_agent()

agent_rows = []
for c in CASES:
    result = run_agent(agent, c["input"])
    agent_rows.append({**c, **result})
    print("Q:", c["input"])
    print("Tools called:", result["tool_names"], " (expected:", c["expected_tools"], ")")
    print("-" * 80)

### 19.2 · Build LLMTestCases with expected tools and traces

# %%
from deepeval.test_case import LLMTestCase, ToolCall

test_cases = [
    LLMTestCase(
        input=r["input"],
        actual_output=r["actual_output"],
        tools_called=r["tools_called"],
        expected_tools=[ToolCall(name=n) for n in r["expected_tools"]],
    )
    for r in agent_rows
]

# trajectory metrics need the trace tree attached to the case
for test_case, r in zip(test_cases, agent_rows):
    test_case._trace_dict = r["trace"]

print("cases ready:", len(test_cases))
for tc in test_cases:
    print("-", tc.input[:60], "| expected tools:", [t.name for t in tc.expected_tools])

# %% [markdown]

## Theory

### What a trajectory test case needs
A chatbot case needs input and output. An agent case needs three more things:

1. **`tools_called`** — the actual `ToolCall` objects captured from the trace root.
2. **`expected_tools`** — the tool calls the task *should* require (`ToolCall(name=...)`). This encodes intent: "this question should need a docs search", not "the agent should say exactly these words".
3. **`_trace_dict`** — the full span tree, attached as a dict. The trajectory metrics (Day 21) read the ordered chain of decisions from it. (The underscore marks it as internal plumbing — the supported path is letting `run_agent` capture the trace and assigning it here.)

### Capturing traces programmatically
Everything in `run_agent` is plain code — no notebook magic: wrap the agent call in `trace()`, read `root_spans`, and dump `trace_manager.create_nested_spans_dict(...)`. The same function runs in a script, a pytest fixture, or a CI job. The harness is now *portable*, which matters because Day 27 will run it in CI.

### expected_tools encodes intent, not strings
The third case expects `["search_docs", "escalate_to_human"]` — an escalation *policy*, not a scripted answer. This is the key shift from chatbot evals: we assert on the *behavior* (which tools, in what order) and let the final wording be free. Behavior assertions survive prompt rewrites that would break string-based golden answers.

### Why the loop matters
Case 1's output is a beautiful table of fine-tuning vs prompting — and its tool log is nine `search_docs` calls. A chatbot metric would call that a pass. Today's cases preserve the trace so that Days 20–22 can score what actually happened. Keep this case in mind: it is the course's best example of an output-good, behavior-bad agent.
