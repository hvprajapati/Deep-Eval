# Day 23 — Multi-turn chatbots

> **Module 4 · Agents.** Conversations are evaluated as a whole: memory, persona, completeness, topic discipline — not turn by turn.

## Code

### 23.1 · Two conversations: a healthy bot and a forgetful one

# %%
# --- setup (from Day 03): load course/.env and recreate the course judge ---
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


# %%
from deepeval.test_case import ConversationalTestCase, Turn

healthy = ConversationalTestCase(
    chatbot_role="ShopBot",
    scenario="Customer asks about a refund, then about shipping.",
    turns=[
        Turn(role="user", content="I bought a jacket last week, can I return it?"),
        Turn(role="assistant", content="Yes — returns are accepted within 30 days of purchase."),
        Turn(role="user", content="Great. And how long does shipping take?"),
        Turn(role="assistant", content="Orders ship within 2 business days."),
    ],
)

forgetful = ConversationalTestCase(
    chatbot_role="ShopBot",
    scenario="Customer asks about a refund, then about shipping.",
    turns=[
        Turn(role="user", content="I bought a jacket last week, can I return it?"),
        Turn(role="assistant", content="Yes — returns are accepted within 30 days of purchase."),
        Turn(role="user", content="Great. And how long does shipping take?"),
        Turn(role="assistant", content="What item are you asking about?"),  # forgot the jacket
        Turn(role="user", content="The jacket I just asked about!"),
        Turn(role="assistant", content="Could you repeat your question?"),
    ],
)

### 23.2 · Measure the conversational metrics

# %%
from deepeval import evaluate
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TopicAdherenceMetric,
)

metrics = [
    KnowledgeRetentionMetric(model=judge),
    RoleAdherenceMetric(model=judge),
    ConversationCompletenessMetric(model=judge),
    TopicAdherenceMetric(model=judge, relevant_topics=["refunds", "shipping", "returns"]),
]

results = evaluate(test_cases=[healthy, forgetful], metrics=metrics)
for tr in results.test_results:
    print(f"\n=== {tr.name}: success={tr.success}")
    for md in tr.metrics_data:
        print(f"  {md.name:<26} score={md.score}")
        print(f"    reason: {md.reason[:140]}")

# %% [markdown]

## Theory

### Conversations are not a bag of turns
Single-turn metrics would score each reply in isolation — and the forgetful bot's replies are individually fine ("What item are you asking about?" is a perfectly reasonable sentence). The *failure* exists only across turns: the bot dropped the jacket. Conversational metrics read the whole `ConversationalTestCase` (its `turns`, `scenario`, and `chatbot_role`) and judge the arc, not the atoms.

### The four conversational lenses
- **KnowledgeRetention** — did the bot keep facts from earlier turns (the jacket, the 30-day policy)?
- **RoleAdherence** — did it stay in character (ShopBot, not a poet or a doctor)?
- **ConversationCompleteness** — did every user question get fully resolved by the end?
- **TopicAdherence** — did the conversation stay within the allowed topics (`relevant_topics=["refunds", "shipping"]`)? A bot that starts selling car insurance fails this.

Expect the healthy conversation to pass all four; the forgetful one to fail Retention and Completeness while possibly passing Role and Topic — the split is the diagnosis: the bot's problem is memory, not manners.

### When multi-turn evals matter
Any app with session state: support bots, therapy companions, booking flows, multi-step agents. If your product can be asked "and what about...?" — referencing a previous turn — you need a conversational pack. The golden data is harder to write (scenarios with arcs, not question/answer pairs), but there is no shortcut: memory bugs are invisible to single-turn metrics by definition.
