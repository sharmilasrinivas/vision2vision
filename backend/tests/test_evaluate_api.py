
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_evaluate_endpoint_all_criteria_pass():
    response = client.post(
        "/evaluate",
        json={
            "task_id": "api-test-001",
            "summary": "All requirements passed.",
            "criteria": [
                {
                    "name": "Object present",
                    "passed": True,
                    "explanation": "The requested object is present.",
                },
                {
                    "name": "Style correct",
                    "passed": True,
                    "explanation": "The requested style is satisfied.",
                },
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["score"] == 1.0
    assert result["overall_passed"] is True
    assert result["decision"] == "pass"
    assert result["suggested_action"] == "accept"


def test_evaluate_endpoint_partial_failure():
    response = client.post(
        "/evaluate",
        json={
            "task_id": "api-test-002",
            "summary": "One requirement failed.",
            "criteria": [
                {
                    "name": "Object present",
                    "passed": True,
                    "explanation": "The object is present.",
                },
                {
                    "name": "Style correct",
                    "passed": False,
                    "explanation": "The style does not match.",
                },
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["score"] == 0.5
    assert result["overall_passed"] is False
    assert result["decision"] == "fail"
    assert result["suggested_action"] == "retry"


def test_evaluate_endpoint_rejects_empty_criteria():
    response = client.post(
        "/evaluate",
        json={
            "task_id": "api-test-003",
            "summary": "No criteria provided.",
            "criteria": [],
        },
    )

    assert response.status_code == 422
