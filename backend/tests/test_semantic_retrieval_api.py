
from fastapi.testclient import TestClient

import backend.app.main as main_module

client = TestClient(main_module.app)


def test_semantic_retrieval_endpoint_returns_matches(monkeypatch):
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
                    "name": "Sunlight",
                    "passed": False,
                    "explanation": "Sunlight was omitted.",
                }
            ],
            "relevance_score": 0.82,
        }
    ]
    captured = {}

    def mock_retrieve(query, limit=5, min_similarity=0.25):
        captured["query"] = query
        captured["limit"] = limit
        captured["min_similarity"] = min_similarity
        return expected

    monkeypatch.setattr(
        main_module,
        "retrieve_semantic_failures",
        mock_retrieve,
    )

    response = client.post(
        "/retrieve-semantic-failures",
        json={
            "query": "Why do plants need solar energy?",
            "limit": 3,
            "min_similarity": 0.4,
        },
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert captured == {
        "query": "Why do plants need solar energy?",
        "limit": 3,
        "min_similarity": 0.4,
    }


def test_semantic_retrieval_endpoint_rejects_invalid_limit():
    response = client.post(
        "/retrieve-semantic-failures",
        json={
            "query": "photosynthesis",
            "limit": 0,
        },
    )

    assert response.status_code == 422


def test_semantic_retrieval_endpoint_rejects_invalid_threshold():
    response = client.post(
        "/retrieve-semantic-failures",
        json={
            "query": "photosynthesis",
            "min_similarity": 1.5,
        },
    )

    assert response.status_code == 422


def test_semantic_retrieval_endpoint_handles_empty_query():
    response = client.post(
        "/retrieve-semantic-failures",
        json={
            "query": "   ",
        },
    )

    # Whitespace is not empty to Pydantic's min_length constraint,
    # so the retrieval function should reject it.
    assert response.status_code == 422
