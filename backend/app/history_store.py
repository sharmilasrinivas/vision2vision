
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
