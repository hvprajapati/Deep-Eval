# Day 17 — LLM tracing

> **Module 4 · Agents.** Agents are non-deterministic loops; traces are the evidence. Capture every step before you score anything.

## Code

### 17.1 · Observe two functions and trace a run

# %%
# %%
import os

os.environ["CONFIDENT_TRACE_VERBOSE"] = "0"  # silence Confident AI dev logging

from deepeval.tracing import observe, trace, trace_manager


@observe(type="tool")
def search_docs(query: str) -> str:
    """Search internal documentation for a concept or policy."""
    return f"Docs result for '{query}': found a section with a definition and a short example."


@observe(type="llm")
def summarize(text: str) -> str:
    return "Summary: " + text[:60] + "..."


with trace() as current_trace:
    result = search_docs("tokens")
    summary = summarize(result)

### 17.2 · Read the trace back

# %%
root = current_trace.root_spans[0]
print("root span:", root.name, "| type:", root.type)
print("tools called inside the trace:", [t.name for t in (root.tools_called or [])])

import json

nested = trace_manager.create_nested_spans_dict(root)
print(json.dumps(nested, indent=2, default=str)[:1500])

# %% [markdown]

## Theory

### Spans and traces
- A **span** is one observed unit of work — a tool call, an LLM call, a retriever lookup — with input, output, and a type (`"tool"`, `"llm"`, `"agent"`, `"retriever"`).
- A **trace** is the tree of spans for one end-to-end run. `trace()` opens the root; `@observe` decorators attach spans as functions execute.
- `trace_manager` keeps the current traces in memory and can post them to the Confident AI platform (Day 28).

### Why agents need traces before metrics
A chatbot answer can be judged from the answer alone. An agent's answer hides its story: the same final sentence can follow one clean tool call or nine flailing ones, a correct plan or a lucky stumble. Output-only metrics would give both runs the same score. Traces capture the *path*, and the metrics of Days 20–22 score the path — tool choices, argument quality, plan adherence, step efficiency. No trace, no path, no score: that is why this day comes first.

### Tracing = observability for non-determinism
Logs freeze what happened once; traces freeze *every* run, structured and queryable. When a production agent misbehaves, the trace is the first artifact you open — and after Day 19, traces are also the input to the eval harness, so the same machinery powers debugging *and* quality gates.
