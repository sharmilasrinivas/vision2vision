
import pytest

from backend.app.evaluation import EvaluationResult
from backend.app.llm_evaluator import (
    LLMEvaluationRequest,
    evaluate_text_output,
)


def make_request():
    return LLMEvaluationRequest(
        task_id="llm-test-001",
        task_description="Explain photosynthesis in simple terms.",
        generated_output="Plants use sunlight to make their food.",
        criteria_names=[
            "Mentions sunlight",
            "Explains food production",
        ],
    )


@pytest.mark.asyncio
async def test_llm_evaluator_accepts_valid_response(monkeypatch):
    async def mock_generate_chat_response(prompt):
        assert "photosynthesis" in prompt.lower()
        return """
        {
          "criteria": [
            {
              "name": "Mentions sunlight",
              "passed": true,
              "explanation": "The output mentions sunlight."
            },
            {
              "name": "Explains food production",
              "passed": true,
              "explanation": "The output describes plants making food."
            }
          ]
        }
        """

    monkeypatch.setattr(
        "backend.app.llm_evaluator.generate_chat_response",
        mock_generate_chat_response,
    )

    result = await evaluate_text_output(make_request())

    assert isinstance(result, EvaluationResult)
    assert result.score == 1.0
    assert result.decision == "pass"
    assert result.suggested_action == "accept"


@pytest.mark.asyncio
async def test_llm_evaluator_rejects_invalid_json(monkeypatch):
    async def mock_generate_chat_response(prompt):
        return "This is not valid JSON."

    monkeypatch.setattr(
        "backend.app.llm_evaluator.generate_chat_response",
        mock_generate_chat_response,
    )

    with pytest.raises(ValueError, match="invalid evaluation response"):
        await evaluate_text_output(make_request())


@pytest.mark.asyncio
async def test_llm_evaluator_rejects_mismatched_criteria(monkeypatch):
    async def mock_generate_chat_response(prompt):
        return """
        {
          "criteria": [
            {
              "name": "Mentions sunlight",
              "passed": true,
              "explanation": "Sunlight is mentioned."
            },
            {
              "name": "Uses correct grammar",
              "passed": true,
              "explanation": "The sentence is grammatical."
            }
          ]
        }
        """

    monkeypatch.setattr(
        "backend.app.llm_evaluator.generate_chat_response",
        mock_generate_chat_response,
    )

    with pytest.raises(ValueError, match="did not match"):
        await evaluate_text_output(make_request())
