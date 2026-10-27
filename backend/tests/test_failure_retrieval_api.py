
from fastapi.testclient import TestClient

import backend.app.main as main_module

client = TestClient(main_module.app)


def test_retrieve_failures_endpoint_returns_matches(monkeypatch):
    expected = [
        {
            "id": 1,
            "task_id": "failure-001",
            "task_description": "Explain photosynthesis.",
            "generated_output": "Plants make food.",
            "criteria": [],
            "score": 0.0,
            "decision": "fail",
            "created_at": "2026-10-27 10:00:00",
            "failed_criteria": [
                {
                    "name": "Mentions sunlight",
                    "passed": False,
                    "explanation": "Sunlight was omitted.",
                }
            ],
            "relevance_score": 0.5,
        }
    ]

    captured = {}

    def mock_retrieve_similar_failures(query, limit=5):
        captured["query"] = query
        captured["limit"] = limit
        return expected

    monkeypatch.setattr(
        main_module,
        "retrieve_similar_failures",
        mock_retrieve_similar_failures,
    )

    response = client.post(
        "/retrieve-failures",
        json={
            "query": "photosynthesis sunlight",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert captured == {
        "query": "photosynthesis sunlight",
        "limit": 3,
    }


def test_retrieve_failures_endpoint_rejects_invalid_limit():
    response = client.post(
        "/retrieve-failures",
        json={
            "query": "photosynthesis",
            "limit": 0,
        },
    )

    assert response.status_code == 422


def test_retrieve_failures_endpoint_rejects_empty_query():
    response = client.post(
        "/retrieve-failures",
        json={
            "query": "",
            "limit": 5,
        },
    )

    assert response.status_code == 422
