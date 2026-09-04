# Day 03 — Judges, the LLM behind the score

> **Module 1 · Foundations.** Build the judge every other day recreates: Llama 3.3 70B on Groq wrapped in `LocalModel`.

## Code

### 3.1 · Load the key and create the judge

# %%
import os

from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.models import LocalModel

# LocalModel = any OpenAI-compatible endpoint. We point it at Groq, so the
# whole course runs on a free key. Four ingredients:
#   model     — the LLM that will do the scoring
#   base_url  — where it lives (Groq speaks the OpenAI protocol)
#   api_key   — read from .env, never pasted into code
#   temp = 0  — judging must be deterministic: same case, same score
judge = LocalModel(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

print("Judge:", judge.get_model_name())

### 3.2 · A second judge: the vision model (Day 26 will use it)

# %%
vision_judge = LocalModel(
    model="llama-3.2-90b-vision-preview",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

print("Vision judge:", vision_judge.get_model_name())

### 3.3 · The same pattern with other providers (reference — do not run)

# %%
# OpenAI:
# from deepeval.models import OpenAIModel
# judge = OpenAIModel(model="gpt-4o", temperature=0)

# Ollama (local):
# from deepeval.models import OllamaModel
# judge = OllamaModel(model="llama3.1", base_url="http://localhost:11434")

# Azure OpenAI:
# from deepeval.models import AzureOpenAIModel
# judge = AzureOpenAIModel(model=..., azure_deployment=..., ...)

# %% [markdown]

## Theory

### What an LLM judge is
The judge is an LLM whose job is scoring other LLM outputs. Every DeepEval metric wraps a *judge prompt* — instructions, the test case's fields, and a required output format — and sends it to this model. The judge returns a score and a reason; the metric compares the score to the threshold and emits the verdict.

### Judge hygiene — three rules
1. **Temperature 0.** Judging must be deterministic. At temperature 1 your scores wander between runs, and so do your pass/fail verdicts.
2. **Independence.** The judge should not be the app's model grading itself — that is a bias shortcut. In this course the app (Days 11–18) runs on `openai/gpt-oss-120b` while the judge is Llama 3.3 — deliberately different models, different providers.
3. **Record the judge.** Log which judge scored each run (Day 27 does this with hyperparameters). Silently swapping the judge changes every score in the pack.

### Why every later day re-creates the judge
This five-line object is the course's only "setup", so each notebook carries its own copy in its first cell. Two reasons: any day works even if you open it first, out of order; and by Day 10 typing it is muscle memory. There is no shared file to import, no hidden state — the judge is just a variable in the notebook.

### Cheap vs capable judges
- An 8B judge (`llama-3.1-8b-instant` on Groq) scores tone and format fine, and costs almost nothing.
- Factual correctness and multi-step reasoning want a 70B+ judge like ours.
- Judging is not free: five RAG metrics on three cases is roughly 40 judge calls (Day 15). Pick the smallest judge that scores reliably for the metric in question, and re-check when you change it.

### LocalModel = any OpenAI-compatible endpoint
`LocalModel` accepts a `base_url`, so Groq, Ollama, vLLM, Together, and any self-hosted OpenAI-compatible server all work as judges. The course uses it so the entire curriculum runs on a free Groq key.
