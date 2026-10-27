
"""Data models for evaluating Voice2Vision Guardian outputs."""

from typing import Literal

from pydantic import BaseModel, Field

from pydantic import BaseModel, Field, model_validator


class EvaluationCriterion(BaseModel):
    """Result for one evaluation criterion."""

    name: str = Field(min_length=1)
    passed: bool
    explanation: str = Field(min_length=1)



class EvaluationResult(BaseModel):
    """Structured evaluation result for an AI-generated output."""

    task_id: str = Field(min_length=1)
    overall_passed: bool
    score: float = Field(ge=0.0, le=1.0)
    decision: Literal["pass", "fail", "review"]
    summary: str = Field(min_length=1)
    criteria: list[EvaluationCriterion] = Field(min_length=1)
    suggested_action: Literal["accept", "retry", "human_review"]

    @model_validator(mode="after")
    def validate_consistency(self):
        """Ensure the decision agrees with criteria and next action."""
        all_criteria_passed = all(
            criterion.passed for criterion in self.criteria
        )

        if self.overall_passed:
            if not all_criteria_passed:
                raise ValueError(
                    "A passing evaluation requires every criterion to pass."
                )
            if self.decision != "pass":
                raise ValueError(
                    "A passing evaluation must have decision='pass'."
                )
            if self.suggested_action != "accept":
                raise ValueError(
                    "A passing evaluation must suggest action='accept'."
                )
        else:
            if self.decision == "pass":
                raise ValueError(
                    "A failed evaluation cannot have decision='pass'."
                )
            if self.decision == "review" and (
                self.suggested_action != "human_review"
            ):
                raise ValueError(
                    "A review decision must suggest action='human_review'."
                )

        return self
