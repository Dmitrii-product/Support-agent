"""
RAG Evaluation Suite for the Smart Support Triage Agent.

Runs the agent against every entry in the golden dataset and evaluates
the quality of both retrieval and generation using DeepEval metrics.

Metrics used:
  - Answer Relevancy:     Does the response address the user's question?
  - Faithfulness:         Is the response grounded in the retrieved context?
  - Contextual Relevancy: Is the retrieved context relevant to the query?
  - Hallucination:        Does the response contain hallucinated information?

Usage:
  .venv/bin/pytest tests/evals/test_rag.py -v
"""

import os
import sys
import json
import time
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from langchain_core.messages import HumanMessage

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.graph import build_graph
from tests.evals.custom_llm import CustomGroqLLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
# Pause between test cases to respect Groq free-tier rate limits (12K TPM)
RATE_LIMIT_PAUSE = 65

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    """Build the LangGraph agent once for the entire test module."""
    return build_graph()

@pytest.fixture(scope="module")
def evaluator_llm():
    """Instantiate the Groq-backed evaluator LLM once."""
    return CustomGroqLLM()

def load_golden_dataset():
    """Load the golden dataset from JSON. Returns list of dicts."""
    if not os.path.exists(GOLDEN_DATASET_PATH):
        pytest.skip(f"Golden dataset not found at {GOLDEN_DATASET_PATH}. "
                     "Run generate_goldens.py first.")
    with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data:
        pytest.skip("Golden dataset is empty. Run generate_goldens.py first.")
    return data

# ---------------------------------------------------------------------------
# Parametrized test: one test case per golden entry
# ---------------------------------------------------------------------------

def _get_golden_ids():
    """Generate readable test IDs from golden inputs."""
    if not os.path.exists(GOLDEN_DATASET_PATH):
        return []
    with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [
        f"golden_{i}_{(g.get('input', '') or '')[:40]}"
        for i, g in enumerate(data)
    ]

def _get_golden_data():
    """Load goldens for parametrize, or return empty list if not ready."""
    if not os.path.exists(GOLDEN_DATASET_PATH):
        return []
    with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

_goldens = _get_golden_data()
_golden_ids = _get_golden_ids()


@pytest.mark.skipif(len(_goldens) == 0, reason="Golden dataset not available yet")
@pytest.mark.parametrize("golden", _goldens, ids=_golden_ids)
def test_golden_eval(golden, app, evaluator_llm):
    """
    For each golden entry:
    1. Run the agent with the golden input.
    2. Evaluate Answer Relevancy, Faithfulness, Contextual Relevancy,
       and Hallucination.
    """
    user_input = golden.get("input", "")
    if not user_input:
        pytest.skip("Golden entry has no input.")

    # Invoke the LangGraph agent
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "triage_route": "",
        "retrieved_context": "",
        "final_response": ""
    }

    result = app.invoke(initial_state)
    actual_output = result.get("final_response", "")
    retrieved_text = result.get("retrieved_context", "")
    route = result.get("triage_route", "")

    print(f"\n  Input:    {user_input[:80]}...")
    print(f"  Route:    {route}")
    print(f"  Output:   {len(actual_output)} chars")
    print(f"  Context:  {len(retrieved_text)} chars")

    # Build retrieval context list for DeepEval
    retrieval_context = [retrieved_text] if retrieved_text.strip() else []

    # -- Assemble metrics --
    metrics = []

    # 1. Answer Relevancy — always applicable
    metrics.append(AnswerRelevancyMetric(threshold=0.7, model=evaluator_llm))

    # The following metrics require retrieval context
    if retrieval_context:
        # 2. Faithfulness — is the answer grounded in context?
        metrics.append(FaithfulnessMetric(threshold=0.7, model=evaluator_llm))
        # 3. Contextual Relevancy — is the retrieved context relevant to the query?
        metrics.append(ContextualRelevancyMetric(threshold=0.7, model=evaluator_llm))
        # 4. Hallucination — does the answer fabricate facts beyond context?
        metrics.append(HallucinationMetric(threshold=0.7, model=evaluator_llm))
    else:
        print("  ⚠ No retrieval context — skipping Faithfulness, "
              "Contextual Relevancy, and Hallucination metrics.")

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        # If the golden has an expected output, include it
        expected_output=golden.get("expected_output", None),
        context=golden.get("context", None),
    )

    assert_test(test_case, metrics)

    # Rate limit pause between test cases
    time.sleep(RATE_LIMIT_PAUSE)
