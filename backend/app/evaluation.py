
"""Data models for evaluating Voice2Vision Guardian outputs."""

from typing import Literal

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


def evaluate_criteria(
    task_id: str,
    criteria: list[EvaluationCriterion],
    summary: str,
) -> EvaluationResult:
    """Calculate an evaluation score and choose a next action."""

    if not criteria:
        raise ValueError("At least one evaluation criterion is required.")

    passed_count = sum(criterion.passed for criterion in criteria)
    score = passed_count / len(criteria)
    overall_passed = passed_count == len(criteria)

    return EvaluationResult(
        task_id=task_id,
        overall_passed=overall_passed,
        score=score,
        decision="pass" if overall_passed else "fail",
        summary=summary,
        criteria=criteria,
        suggested_action="accept" if overall_passed else "retry",
    )
