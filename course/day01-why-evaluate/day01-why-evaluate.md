# Day 01 — Why evaluate, and the eval loop

> **Module 1 · Foundations.** Set up DeepEval, and see with your own eyes why LLM applications cannot be tested with ordinary unit tests.

## Code

### 1.1 · One-time environment setup (terminal, not this notebook)

```bash
cd course
python -m venv .venv
.venv\Scripts\activate        # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Then create `course/.env` with your free Groq key (https://console.groq.com):

```text
GROQ_API_KEY=your-groq-key-here
```

**Always run Jupyter from the `course/` folder** (`cd course` then `jupyter notebook`). That single habit is what lets every day's tiny setup cell find `.env`.

### 1.2 · Check the installation

# %%
import deepeval

print("DeepEval", deepeval.__version__)

### 1.3 · Switch off anonymous telemetry and load the key

# %%
import os

from dotenv import load_dotenv

load_dotenv()  # reads course/.env
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")  # no anonymous usage reports

### 1.4 · See non-determinism live

# %%
from groq import Groq

client = Groq()


def reply(prompt: str, temperature: float = 1.0) -> str:
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


q = "Explain what an LLM token is, in one sentence."
print("Run 1:", reply(q))
print("Run 2:", reply(q))

# %% [markdown]

## Theory

### Unit tests assume determinism
A unit test is a promise: *same input in, same output out*. `assert add(2, 2) == 4` is true forever, so a test can pin it. LLMs do not make that promise. The same prompt returns different words every time — and even at temperature 0, providers, model versions, and caches move underneath you. So `assert reply(q) == "A token is ..."` fails constantly even when the app is working. We need a different kind of test: one that judges *quality*, not equality.

### The eval loop
Every day of this course is one loop with four parts:

1. **Dataset** — golden inputs (plus expected outputs where you have them).
2. **Metric** — a scorer that turns an LLM output into a 0–1 score plus a written reason. Usually the scorer is itself an LLM: an *LLM judge*.
3. **Threshold** — the score below which we call it a failure. 0.5 is a starting point, not a law.
4. **Regression** — re-run the whole pack whenever the prompt, model, or pipeline changes.

You build the loop for the first time on Day 4. Everything after is making the loop stronger: better metrics (Days 5–10), a real RAG pipeline inside it (Days 11–16), agents with traces (Days 17–24), and production guards (Days 25–28).

### Where DeepEval sits
- **RAGAS** — excellent, but RAG-only. DeepEval covers RAG *and* agents, safety, and benchmarks.
- **LangSmith** — tracing-first commercial platform; the open-source parts are thinner.
- **promptfoo** — config-file driven; great for quick tables, less expressive for custom metrics.

DeepEval is open-source, judge-based, and framework-agnostic — your app can be built with anything. Three properties we verify all course long: every metric returns **score + reason + verdict**, metrics can be **customized or written from scratch** (Days 5, 9, 10), and **traces** turn black-box agent runs into testable evidence (Days 17–24).
