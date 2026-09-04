"""Evals as pytest tests. Run with: deepeval test run tests/"""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # course/.env — or GROQ_API_KEY arrives from CI secrets (Day 27)

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

# The course judge, defined here so this suite is self-contained.
judge = LocalModel(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

GOLDEN = [
    {"input": "What's your refund policy?",
     "actual_output": "We offer full refunds within 30 days of purchase.",
     "expected_output": "Refunds are available within 30 days of purchase."},
    {"input": "How long does shipping take?",
     "actual_output": "Orders ship within 2 business days.",
     "expected_output": "Orders ship within 2 business days."},
]


@pytest.fixture(scope="module")
def relevancy():
    return AnswerRelevancyMetric(model=judge, threshold=0.5)


@pytest.mark.parametrize(
    "case",
    GOLDEN,
    ids=[c["input"] for c in GOLDEN],
)
def test_relevant_answers(case, relevancy):
    test_case = LLMTestCase(**case)
    assert_test(test_case, metrics=[relevancy])


def test_professional_tone():
    tone = GEval(
        name="Professional Tone",
        criteria="Is the response polite and professional?",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="What's your refund policy?",
        actual_output="No refunds. Read the policy page next time.",
    )
    assert_test(test_case, metrics=[tone])
