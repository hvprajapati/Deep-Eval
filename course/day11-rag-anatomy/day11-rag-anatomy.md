# Day 11 — Anatomy of a RAG you can evaluate

> **Module 3 · RAG evaluation.** Build the app under test for the rest of the module: a tiny corpus, a naive retriever, and a grounded answerer — all in this notebook.

## Code

### 11.1 · Load the environment

# %%
import os

from dotenv import load_dotenv

load_dotenv()  # reads course/.env — run Jupyter from the course/ folder
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

### 11.2 · The corpus — five documents, one per idea

# %%
CORPUS = [
    {"title": "Tokens", "content": (
        "Large language models split text into tokens, common character sequences roughly 4 "
        "characters or three-quarters of a word long. Both the prompt and the output are counted "
        "in tokens, and pricing and context limits are measured in tokens.")},
    {"title": "Embeddings", "content": (
        "An embedding is a fixed-length vector that represents the meaning of a piece of text. "
        "Texts with similar meaning have vectors that are close together, usually measured by "
        "cosine similarity, which is what lets a system do semantic search.")},
    {"title": "Retrieval-Augmented Generation", "content": (
        "RAG grounds an LLM's answer in external documents. At query time the system retrieves "
        "the most relevant chunks and passes them to the model as context, which reduces "
        "hallucination and lets you update knowledge without retraining.")},
    {"title": "Hallucination", "content": (
        "A hallucination is fluent, confident text that is factually wrong or unsupported by its "
        "sources. It happens because the model predicts likely text rather than looking facts up. "
        "Grounding answers in retrieved context is the main defense.")},
    {"title": "AI agents", "content": (
        "An AI agent is an LLM given a goal, tools, and a loop: plan, call a tool, observe the "
        "result, decide the next step, until the task is done.")},
]

### 11.3 · The retriever — naive keyword overlap (deliberately)

# %%
def retrieve(query: str, k: int = 2) -> list[str]:
    """Naive retriever: rank docs by keyword overlap with the query."""
    q_words = set(query.lower().split())
    scored = []
    for doc in CORPUS:
        doc_words = set((doc["title"] + " " + doc["content"]).lower().split())
        overlap = len(q_words & doc_words)
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc["content"] for _, doc in scored[:k]]

### 11.4 · The generator — answer only from context

# %%
from groq import Groq

RAG_SYSTEM = (
    "Answer ONLY using the provided context. If the context doesn't contain the answer, say you "
    "don't have that information. Be concise (1-2 sentences)."
)


def answer_with_groq(question: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    resp = Groq().chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def ask_rag(question: str, k: int = 2) -> tuple[str, list[str]]:
    """Full pipeline: retrieve, then answer. Returns (answer, passages)."""
    passages = retrieve(question, k)
    return answer_with_groq(question, passages), passages

### 11.5 · Try the pipeline

# %%
for q in ["What is a token in an LLM?", "How does RAG reduce hallucination?"]:
    answer, passages = ask_rag(q)
    print("Q:", q)
    print("A:", answer)
    print("retrieved:", len(passages), "passages")
    print("-" * 70)

# %% [markdown]

## Theory

### Three components, three failure points
1. **Index** — the documents and how they are split. Wrong chunking = right info cut in half.
2. **Retriever** — ranking documents for a query. Wrong ranking = right doc present but buried.
3. **Generator** — turning retrieved chunks into an answer. Wrong grounding = confident fiction.

Each fails differently, so each gets different metrics: retriever metrics judge the *chunks* (Days 12–13), generator metrics judge the *answer* (Day 14). This separation is the single most important idea in RAG evaluation.

### The golden dataset for RAG
A RAG golden case has four parts: `input` (question), `expected_output` (the correct answer), `retrieval_context` (what the retriever actually returned), and `actual_output` (what the generator said). The retriever metrics only need the first three; the generator metrics only need the last three. This course's dataset lives in Day 15, and Day 16 grows it.

### The "answer only from context" prompt — and refusal
`RAG_SYSTEM` tells the model to refuse when the context lacks the answer. That is a *feature to evaluate*, not a limitation: a RAG that refuses is measurably better than one that hallucinates. Day 14 measures exactly this behavior — and the corpus is deliberately tiny so the retriever genuinely misses some questions.

### Why the pipeline lives in cells, not a file
Every later RAG day (12–16) starts with a short recap cell that rebuilds these same pieces. Repetition is the point: by Day 15 the corpus and the retriever are tools you *know*, not black boxes you import.
