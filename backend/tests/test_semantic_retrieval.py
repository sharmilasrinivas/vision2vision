
import numpy as np
import pytest

import backend.app.semantic_retrieval as semantic_module


def test_semantic_retrieval_ranks_failures(monkeypatch):
    records = [
        {
            "id": 1,
            "task_id": "failure-001",
            "task_description": "Explain how plants make food.",
            "generated_output": "Plants make food.",
            "criteria": [
                {
                    "name": "Sunlight",
                    "passed": False,
                    "explanation": "The response omitted sunlight.",
                }
            ],
            "score": 0.0,
            "decision": "fail",
            "created_at": "2026-10-27 10:00:00",
        },
        {
            "id": 2,
            "task_id": "failure-002",
            "task_description": "Explain database indexing.",
            "generated_output": "Indexes improve lookup speed.",
            "criteria": [
                {
                    "name": "Index explanation",
                    "passed": False,
                    "explanation": "The explanation is incomplete.",
                }
            ],
            "score": 0.0,
            "decision": "fail",
            "created_at": "2026-10-27 10:01:00",
        },
        {
            "id": 3,
            "task_id": "success-001",
            "task_description": "Explain photosynthesis.",
            "generated_output": "Plants use sunlight.",
            "criteria": [],
            "score": 1.0,
            "decision": "pass",
            "created_at": "2026-10-27 10:02:00",
        },
    ]

    monkeypatch.setattr(
        semantic_module,
        "get_recent_evaluations",
        lambda limit, db_path: records,
    )

    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            # Deterministic embeddings to test ranking without
            # downloading a real model during unit tests.
            return np.array(
                [
                    [1.0, 0.0],  # query
                    [0.95, 0.05],  # relevant failure
                    [0.0, 1.0],  # unrelated failure
                ]
            )

    monkeypatch.setattr(
        semantic_module,
        "_get_model",
        lambda: FakeModel(),
    )

    results = semantic_module.retrieve_semantic_failures(
        query="Why do plants need sunlight?",
        limit=5,
        min_similarity=0.25,
    )

    assert len(results) == 1
    assert results[0]["task_id"] == "failure-001"
    assert results[0]["failed_criteria"][0]["name"] == "Sunlight"
    assert results[0]["relevance_score"] > 0.9


def test_semantic_retrieval_rejects_empty_query():
    with pytest.raises(ValueError, match="Query must not be empty"):
        semantic_module.retrieve_semantic_failures(query="   ")


def test_semantic_retrieval_rejects_invalid_limit():
    with pytest.raises(ValueError, match="Limit must be"):
        semantic_module.retrieve_semantic_failures(
            query="photosynthesis",
            limit=0,
        )


def test_semantic_retrieval_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="Minimum similarity"):
        semantic_module.retrieve_semantic_failures(
            query="photosynthesis",
            min_similarity=1.5,
        )
