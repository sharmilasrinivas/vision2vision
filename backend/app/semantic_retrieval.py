
"""Semantic retrieval for past Voice2Vision Guardian failures."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.history_store import get_recent_evaluations

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once, on first use."""
    return SentenceTransformer(MODEL_NAME)


def retrieve_semantic_failures(
    query: str,
    limit: int = 5,
    db_path: str | Path = "data/voice2vision.db",
    min_similarity: float = 0.25,
) -> list[dict[str, Any]]:
    """Rank past failed evaluations by semantic similarity."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")

    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("Minimum similarity must be between 0 and 1.")

    records = get_recent_evaluations(limit=100, db_path=db_path)
    candidates = []

    for record in records:
        if record["decision"] != "fail":
            continue

        failed_criteria = [
            criterion
            for criterion in record["criteria"]
            if not criterion.get("passed", False)
        ]

        searchable_text = " ".join(
            [
                record["task_description"],
                record["generated_output"],
                " ".join(
                    f'{criterion.get("name", "")} '
                    f'{criterion.get("explanation", "")}'
                    for criterion in failed_criteria
                ),
            ]
        ).strip()

        if searchable_text:
            candidates.append(
                (record, failed_criteria, searchable_text)
            )

    if not candidates:
        return []

    model = _get_model()
    texts = [query] + [
        candidate[2] for candidate in candidates
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = cosine_similarity(
        embeddings[0:1],
        embeddings[1:],
    )[0]

    results = []

    for (record, failed_criteria, _), score in zip(
        candidates,
        scores,
    ):
        similarity = float(score)

        if similarity < min_similarity:
            continue

        result = dict(record)
        result["failed_criteria"] = failed_criteria
        result["relevance_score"] = similarity
        results.append(result)

    results.sort(
        key=lambda record: (
            record["relevance_score"],
            record["id"],
        ),
        reverse=True,
    )

    return results[:limit]
