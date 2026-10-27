
from fastapi.testclient import TestClient

import backend.app.main as main_module

client = TestClient(main_module.app)


def test_evaluate_endpoint_saves_history(monkeypatch):
    saved = {}

    def mock_save_evaluation(**kwargs):
        saved.update(kwargs)
        return 1

    monkeypatch.setattr(
        main_module,
        "save_evaluation",
        mock_save_evaluation,
    )

    response = client.post(
        "/evaluate",
        json={
            "task_id": "persist-001",
            "summary": "Check a generated output.",
            "criteria": [
                {
                    "name": "Required object present",
                    "passed": True,
                    "explanation": "The object is present.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["score"] == 1.0
    assert saved["task_id"] == "persist-001"
    assert saved["decision"] == "pass"
    assert saved["score"] == 1.0
    assert saved["criteria"][0]["name"] == "Required object present"


def test_history_endpoint_returns_records(monkeypatch):
    expected = [
        {
            "id": 1,
            "task_id": "persist-001",
            "task_description": "Example task",
            "generated_output": "Example output",
            "criteria": [],
            "score": 1.0,
            "decision": "pass",
            "created_at": "2026-10-27 10:00:00",
        }
    ]

    monkeypatch.setattr(
        main_module,
        "get_recent_evaluations",
        lambda limit=10: expected,
    )

    response = client.get("/history?limit=5")

    assert response.status_code == 200
    assert response.json() == expected
