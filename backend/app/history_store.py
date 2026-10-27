
"""SQLite storage for Voice2Vision Guardian evaluation history."""

import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/voice2vision.db")


def initialize_history_db(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Create the evaluation-history table if it doesn't exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                task_description TEXT NOT NULL,
                generated_output TEXT NOT NULL,
                criteria_json TEXT NOT NULL,
                score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
                decision TEXT NOT NULL
                    CHECK(decision IN ('pass', 'fail', 'review')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_evaluation(
    task_id: str,
    task_description: str,
    generated_output: str,
    criteria: list[dict[str, Any]],
    score: float,
    decision: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Save an evaluation and return its database record ID."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("Score must be between 0 and 1.")

    if decision not in {"pass", "fail", "review"}:
        raise ValueError("Invalid evaluation decision.")

    path = Path(db_path)
    initialize_history_db(path)

    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO evaluation_history (
                task_id,
                task_description,
                generated_output,
                criteria_json,
                score,
                decision
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                task_description,
                generated_output,
                json.dumps(criteria),
                score,
                decision,
            ),
        )
        return int(cursor.lastrowid)


def get_recent_evaluations(
    limit: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return recent evaluation records, newest first."""
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")

    path = Path(db_path)

    if not path.exists():
        return []

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, task_id, task_description, generated_output,
                   criteria_json, score, decision, created_at
            FROM evaluation_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records = []
    for row in rows:
        record = dict(row)
        record["criteria"] = json.loads(record.pop("criteria_json"))
        records.append(record)

    return records



def retrieve_similar_failures(
    query: str,
    limit: int = 5,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Retrieve past failed evaluations relevant to a text query."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100.")

    records = get_recent_evaluations(limit=100, db_path=db_path)
    query_terms = set(query.lower().split())
    matches = []

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
        ).lower()

        record_terms = set(searchable_text.split())
        overlap = query_terms & record_terms

        if not overlap:
            continue

        result = dict(record)
        result["failed_criteria"] = failed_criteria
        result["relevance_score"] = (
            len(overlap) / len(query_terms)
        )
        matches.append(result)

    matches.sort(
        key=lambda record: (
            record["relevance_score"],
            record["id"],
        ),
        reverse=True,
    )

    return matches[:limit]
