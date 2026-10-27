
import pytest
from pydantic import ValidationError

from backend.app.evaluation import (
    EvaluationCriterion,
    EvaluationResult,
    evaluate_criteria,
)


def test_evaluation_criterion_valid():
    criterion = EvaluationCriterion(
        name="Required object present",
        passed=True,
        explanation="The requested object is present.",
    )

    assert criterion.passed is True


def test_evaluation_result_valid():
    result = EvaluationResult(
        task_id="task-001",
        overall_passed=True,
        score=1.0,
        decision="pass",
        summary="All criteria passed.",
        criteria=[
            EvaluationCriterion(
                name="Object present",
                passed=True,
                explanation="The object is present.",
            )
        ],
        suggested_action="accept",
    )

    assert result.task_id == "task-001"
    assert result.score == 1.0
    assert result.suggested_action == "accept"


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_evaluation_result_rejects_invalid_score(score):
    with pytest.raises(ValidationError):
        EvaluationResult(
            task_id="task-002",
            overall_passed=False,
            score=score,
            decision="fail",
            summary="Invalid score test.",
            criteria=[
                EvaluationCriterion(
                    name="Object present",
                    passed=False,
                    explanation="The object is missing.",
                )
            ],
            suggested_action="retry",
        )


def test_evaluation_result_rejects_empty_criteria():
    with pytest.raises(ValidationError):
        EvaluationResult(
            task_id="task-003",
            overall_passed=True,
            score=1.0,
            decision="pass",
            summary="No criteria provided.",
            criteria=[],
            suggested_action="accept",
        )


def test_evaluation_rejects_pass_when_criterion_fails():
    with pytest.raises(ValidationError):
        EvaluationResult(
            task_id="task-004",
            overall_passed=True,
            score=0.5,
            decision="pass",
            summary="A criterion failed.",
            criteria=[
                EvaluationCriterion(
                    name="Object present",
                    passed=False,
                    explanation="The requested object is missing.",
                )
            ],
            suggested_action="accept",
        )


def test_review_requires_human_review_action():
    with pytest.raises(ValidationError):
        EvaluationResult(
            task_id="task-005",
            overall_passed=False,
            score=0.5,
            decision="review",
            summary="The result needs review.",
            criteria=[
                EvaluationCriterion(
                    name="Object present",
                    passed=False,
                    explanation="The object is uncertain.",
                )
            ],
            suggested_action="retry",
        )



def test_evaluate_criteria_all_pass():
    criteria = [
        EvaluationCriterion(
            name="Object present",
            passed=True,
            explanation="The requested object is present.",
        ),
        EvaluationCriterion(
            name="Style correct",
            passed=True,
            explanation="The requested style is satisfied.",
        ),
    ]

    result = evaluate_criteria(
        task_id="task-101",
        criteria=criteria,
        summary="All requirements passed.",
    )

    assert result.score == 1.0
    assert result.overall_passed is True
    assert result.decision == "pass"
    assert result.suggested_action == "accept"


def test_evaluate_criteria_partial_failure():
    criteria = [
        EvaluationCriterion(
            name="Object present",
            passed=True,
            explanation="The object is present.",
        ),
        EvaluationCriterion(
            name="Style correct",
            passed=False,
            explanation="The style does not match.",
        ),
    ]

    result = evaluate_criteria(
        task_id="task-102",
        criteria=criteria,
        summary="One requirement failed.",
    )

    assert result.score == 0.5
    assert result.overall_passed is False
    assert result.decision == "fail"
    assert result.suggested_action == "retry"


def test_evaluate_criteria_rejects_empty_list():
    with pytest.raises(ValueError, match="At least one"):
        evaluate_criteria(
            task_id="task-103",
            criteria=[],
            summary="Nothing to evaluate.",
        )
