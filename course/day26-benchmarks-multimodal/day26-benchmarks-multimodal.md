# Day 26 — Benchmarks & multimodal

> **Module 5 · Production.** Public benchmarks measure the *model*; your packs measure the *app*. Plus: image metrics with a vision judge.

## Code

### 26.1 · Run a small HellaSwag slice

# %%
# --- setup (from Day 03): load course/.env and recreate the two judges ---
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

# Image metrics need a judge that can see (Day 03 built this one too)
vision_judge = LocalModel(
    model="llama-3.2-90b-vision-preview",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)


# %%
from deepeval.benchmarks import HellaSwag

# n_shots=0 -> zero-shot; n_problems_per_task keeps the slice cheap.
# First run downloads the dataset from HuggingFace.
benchmark = HellaSwag(n_shots=0, n_problems_per_task=3)
scores = benchmark.evaluate(model=judge)
print("scores:", scores)

### 26.2 · A multimodal case with a vision judge

# %%
from deepeval import evaluate
from deepeval.metrics import ImageCoherenceMetric
from deepeval.test_case import LLMTestCase

# The image is embedded in the answer with a [DEEPEVAL:IMAGE:<url>] marker;
# DeepEval parses it into an MLLMImage and sets multimodal=True automatically.
coherent = LLMTestCase(
    input="Explain the diagram you are showing.",
    actual_output=(
        "The figure below is a bar chart comparing quarterly revenue. "
        "[DEEPEVAL:IMAGE:https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
        "JPEG_example_flower.jpg/320px-JPEG_example_flower.jpg] "
        "It shows a steady increase across the three quarters."
    ),
)

incoherent = LLMTestCase(
    input="Explain the diagram you are showing.",
    actual_output=(
        "The figure is a pie chart of ocean temperatures in 1850. "
        "[DEEPEVAL:IMAGE:https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
        "JPEG_example_flower.jpg/320px-JPEG_example_flower.jpg] "
        "It shows a sharp decline in polar ice."
    ),
)

metric = ImageCoherenceMetric(model=vision_judge, threshold=0.5)
results = evaluate(test_cases=[coherent, incoherent], metrics=[metric])
for tr in results.test_results:
    md = tr.metrics_data[0]
    print(f"{tr.name}: score={md.score} success={md.success}")
    print(f"  reason: {md.reason[:150]}")

# %% [markdown]

## Theory

### Benchmarks measure the model, not your app
HellaSwag (commonsense completion), MMLU (knowledge), GSM8K (math) — these score the *base model's* raw capabilities with zero-shot or few-shot prompts. They tell you nothing about your prompts, your retrieval, your tools, or your users. Both measurements matter, and conflating them is the classic evaluation mistake:

- **Benchmark**: "is this model smarter than the last one?" → run it *once per model swap*.
- **Your packs**: "does my product still behave?" → run on *every change* (Days 8, 27).

### What benchmarks hide
Benchmark numbers assume clean, well-formed prompts and average over datasets that may not resemble your domain. A model that gains 3 points on HellaSwag can still regress on your legal documents. The workflow this course recommends: benchmark for the model shortlist, then decide on your own golden packs — never on the public leaderboard.

### Multimodal evals
Image metrics (Coherence, Helpfulness, Reference, Text-to-Image, Editing) judge text-and-image outputs. The mechanics worth knowing:
- The image rides inside `actual_output` as a `[DEEPEVAL:IMAGE:<url-or-id>]` marker; DeepEval parses it and auto-sets `multimodal=True`.
- The judge must be **vision-capable** — the setup cell above creates `vision_judge` (Llama 3.2 90B Vision on Groq).
- `ImageCoherenceMetric` checks the text actually matches the image shown — the two cases above should split cleanly on it.

The same pattern extends to any app that mixes modalities: describe the output, embed the media, and let a vision judge score the combination.
