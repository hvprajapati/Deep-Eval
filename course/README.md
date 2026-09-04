# AI Evaluation with DeepEval — Full Course

A 28-day, end-to-end course on evaluating LLM applications with [DeepEval](https://deepeval.com/docs) (verified against **DeepEval 4.2.1**). Every day has two parts, in order: **Code** (runnable cells) and **Theory** (the explanation). The course goes from the first `LLMTestCase` to evaluating RAG pipelines, tool-using agents, and multi-agent systems.

Built to be taught alongside `notebook.ipynb` (in the parent folder) — the course reuses and modernizes its G-Eval, DAG, RAG, and agent sections.

## Structure

```
course/
├── .env                  <- your Groq key lives here (never committed)
├── requirements.txt
├── build_notebooks.py    <- regenerates notebooks / syntax-checks the course
├── tests/                <- pytest suite (created by Day 08 when you run it)
└── day01-why-evaluate/
    ├── day01-why-evaluate.md    <- source: theory + code
    └── day01-why-evaluate.ipynb <- the same day as a runnable notebook
    ... (one folder per day, Days 01-28)
```

**Each day is one folder** containing the `.md` (source) and the generated `.ipynb` (what students open). Edit a day's `.md`, run `python build_notebooks.py`, done.

**There are no shared code files.** Everything a day needs — the judge, the RAG pipeline, the agent — lives in that day's own cells. The judge is built on Day 03 and recreated by a small setup cell at the top of every later day, so any notebook runs on its own, in any order.

## Setup (once)

```bash
cd course
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Then put your free [Groq API key](https://console.groq.com) in **`course/.env`**:

```
GROQ_API_KEY=your-groq-key-here
```

**Two rules that make everything else trivial:**

1. **Run Jupyter from the `course/` folder** (`cd course` then `jupyter notebook`). Every notebook's first cell is then just `load_dotenv()` — it finds `course/.env` and the key is ready.
2. **Never paste the key into code.** The setup cells read it from the environment; nothing else.

The course judge is `llama-3.3-70b-versatile` on Groq via `LocalModel`, so the whole curriculum runs on the free tier. Telemetry is opted out in every setup cell via `DEEPEVAL_TELEMETRY_OPT_OUT=YES`.

## The 28 days

| # | Day | What you build / run |
|---|---|---|
| 01 | [Why evaluate, and the eval loop](day01-why-evaluate/day01-why-evaluate.md) | install; see non-determinism live |
| 02 | [LLMTestCase — the unit of evaluation](day02-test-cases/day02-test-cases.md) | test cases by hand; golden JSON dataset |
| 03 | [Judges — the LLM behind the score](day03-judges/day03-judges.md) | the course judge in a cell (Groq `LocalModel`) |
| 04 | [Running your first evaluation](day04-first-evaluation/day04-first-evaluation.md) | `evaluate()`; scores, reasons, thresholds |
| 05 | [G-Eval — metrics you write in English](day05-geval/day05-geval.md) | steps + rubric G-Evals |
| 06 | [Faithfulness, Hallucination & Answer Relevancy](day06-core-metrics/day06-core-metrics.md) | the big-three generator metrics |
| 07 | [Summarization, JSON & score-only mode](day07-summarization-json/day07-summarization-json.md) | reference-free + schema + flaky metrics |
| 08 | [Evals as tests — pytest + deepeval test run](day08-pytest/day08-pytest.md) | `assert_test`, fixtures, CLI |
| 09 | [Custom metrics](day09-custom-metrics/day09-custom-metrics.md) | deterministic + Scorer + judge-based metrics |
| 10 | [DAG metrics — decision-tree evaluation](day10-dag/day10-dag.md) | TaskNode/JudgementNode trees, top-down API |
| 11 | [Anatomy of a RAG you can evaluate](day11-rag-anatomy/day11-rag-anatomy.md) | corpus, retriever, answerer — in cells |
| 12 | [Contextual Precision — is the ranking right?](day12-contextual-precision/day12-contextual-precision.md) | retriever ranking metric + verbose logs |
| 13 | [Contextual Recall & Relevancy](day13-contextual-recall-relevancy/day13-contextual-recall-relevancy.md) | coverage + noise metrics; the diagnosis table |
| 14 | [Generator metrics on real RAG](day14-rag-generator/day14-rag-generator.md) | faithfulness + relevancy on live answers |
| 15 | [The full RAG report](day15-full-rag-report/day15-full-rag-report.md) | all 5 RAG metrics; failure analysis |
| 16 | [Evaluation-driven RAG improvements](day16-rag-improvements/day16-rag-improvements.md) | TF-IDF retriever; re-eval and diff; synthetic cases |
| 17 | [LLM tracing](day17-tracing/day17-tracing.md) | `@observe`, `trace()`, span trees |
| 18 | [Build a tool-using agent](day18-build-agent/day18-build-agent.md) | 3 tools + LangChain agent — in cells |
| 19 | [Traces → agent test cases](day19-agent-test-cases/day19-agent-test-cases.md) | `tools_called`, `expected_tools`, `_trace_dict` |
| 20 | [Action metrics — judging single tool calls](day20-action-metrics/day20-action-metrics.md) | Tool + Argument Correctness |
| 21 | [Trajectory metrics — judging the whole run](day21-trajectory-metrics/day21-trajectory-metrics.md) | Completion, Plan Quality/Adherence, Step Efficiency |
| 22 | [The full agent report](day22-agent-report/day22-agent-report.md) | 6 metrics; fix the loop bug; re-diff |
| 23 | [Multi-turn chatbots](day23-multiturn/day23-multiturn.md) | `ConversationalTestCase`; retention, role, completeness, topic |
| 24 | [Multi-agent systems](day24-multi-agent/day24-multi-agent.md) | supervisor + workers; per-agent and end-to-end evals |
| 25 | [Safety metrics](day25-safety/day25-safety.md) | bias, toxicity, misuse, advice, PII, role |
| 26 | [Benchmarks & multimodal](day26-benchmarks-multimodal/day26-benchmarks-multimodal.md) | HellaSwag slice; image coherence with vision judge |
| 27 | [Evals in production](day27-production/day27-production.md) | JSON datasets, hyperparameters, GitHub Actions gate |
| 28 | [Capstone + Confident AI](day28-capstone/day28-capstone.md) | one suite: RAG + agent + safety; `deepeval view` |

## Tooling

- `python build_notebooks.py` — regenerate every `.ipynb` from its `.md`.
- `python build_notebooks.py --check` — compile-validate every code cell in the course (0 errors expected). No extra files are produced.
- `deepeval test run tests/` — run the Day 08 pytest suite from the course root.

## Teaching notes

- **Reuse the notebook**: `notebook.ipynb`'s G-Eval section → Days 03–05; DAG section → Day 10 (the notebook's two monkey-patches were for older DeepEval bugs — 4.2.1 needs neither); RAG section → Days 11–15; Agent section → Days 17–22.
- **Every day ends on a failing case** — students read the judge's *reason* before touching a fix.
- **Days 16 and 22 are the payoff days**: change one thing, watch scores move, explain why.
- **Stay on the tiny corpus and light throttles** — the whole course runs on a free Groq key.
- **The pack only grows**: Day 15's three questions become Day 16's five, Day 28's eleven-metric capstone.
- **Recap cells are deliberate**: Days 12–16 start with a cell that rebuilds the Day 11 RAG, Days 19–22 rebuild the Day 18 agent. Students retype the core objects daily — by Day 15 the corpus and retriever are theirs, not imports.
