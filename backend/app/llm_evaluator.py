
"""LLM-assisted evaluation of generated text outputs."""

import json

from pydantic import BaseModel, Field, ValidationError

from backend.app.evaluation import (
    EvaluationCriterion,
    EvaluationResult,
    evaluate_criteria,
)
from backend.app.ollama_client import generate_chat_response


class LLMEvaluationRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=100)
    task_description: str = Field(min_length=1, max_length=2000)
    generated_output: str = Field(min_length=1, max_length=4000)
    criteria_names: list[str] = Field(min_length=1, max_length=10)


class LLMEvaluationResponse(BaseModel):
    criteria: list[EvaluationCriterion] = Field(min_length=1)


async def evaluate_text_output(
    request: LLMEvaluationRequest,
) -> EvaluationResult:
    """Ask the local LLM to assess text against explicit criteria."""

    criteria_list = "\n".join(
        f"- {name}" for name in request.criteria_names
    )

    prompt = f"""
You are a cautious AI output evaluator.

Evaluate the generated text against the task and criteria below.
Judge only the supplied text. Do not claim to inspect images, audio,
or external facts. If evidence is insufficient, mark the criterion
as failed and explain the uncertainty.

Task:
{request.task_description}

Generated output:
{request.generated_output}

Criteria to evaluate:
{criteria_list}

Return only a valid JSON object with this exact structure:
{{
  "criteria": [
    {{
      "name": "criterion name copied from the list",
      "passed": true,
      "explanation": "brief evidence-based explanation"
    }}
  ]
}}

Include exactly one result for every supplied criterion.
Use true or false for passed. Do not include Markdown fences.
""".strip()

    raw_response = await generate_chat_response(prompt)

    try:
        parsed = json.loads(raw_response)
        llm_result = LLMEvaluationResponse.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            "The local LLM returned an invalid evaluation response."
        ) from exc

    expected_names = request.criteria_names
    actual_names = [item.name for item in llm_result.criteria]

    if actual_names != expected_names:
        raise ValueError(
            "The LLM evaluation criteria did not match the requested criteria."
        )

    return evaluate_criteria(
        task_id=request.task_id,
        criteria=llm_result.criteria,
        summary="LLM-assisted evaluation of the supplied text output.",
    )
