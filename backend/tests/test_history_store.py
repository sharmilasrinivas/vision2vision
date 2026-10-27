
import json
import sqlite3

import pytest

from backend.app.history_store import (
    get_recent_evaluations,
    initialize_history_db,
    save_evaluation,
)


def test_initialize_history_db_creates_table(tmp_path):
    db_path = tmp_path / "test_history.db"

    initialize_history_db(db_path)

    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'evaluation_history'
            """
        ).fetchone()

    assert result is not None


def test_save_evaluation_persists_record(tmp_path):
    db_path = tmp_path / "test_history.db"

    record_id = save_evaluation(
        task_id="history-001",
        task_description="Explain photosynthesis.",
        generated_output="Plants use sunlight to make food.",
        criteria=[
            {
                "name": "Mentions sunlight",
                "passed": True,
                "explanation": "Sunlight is mentioned.",
            }
        ],
        score=1.0,
        decision="pass",
        db_path=db_path,
    )

    assert record_id > 0

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT task_id, task_description, generated_output,
                   criteria_json, score, decision
            FROM evaluation_history
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == "history-001"
    assert row[1] == "Explain photosynthesis."
    assert row[2] == "Plants use sunlight to make food."
    assert json.loads(row[3])[0]["name"] == "Mentions sunlight"
    assert row[4] == 1.0
    assert row[5] == "pass"


def test_save_evaluation_rejects_invalid_score(tmp_path):
    with pytest.raises(ValueError, match="Score must be"):
        save_evaluation(
            task_id="history-002",
            task_description="Test task",
            generated_output="Test output",
            criteria=[],
            score=1.5,
            decision="fail",
            db_path=tmp_path / "test_history.db",
        )


def test_save_evaluation_rejects_invalid_decision(tmp_path):
    with pytest.raises(ValueError, match="Invalid evaluation decision"):
        save_evaluation(
            task_id="history-003",
            task_description="Test task",
            generated_output="Test output",
            criteria=[],
            score=0.5,
            decision="unknown",
            db_path=tmp_path / "test_history.db",
        )


def test_get_recent_evaluations_returns_newest_first(tmp_path):
    db_path = tmp_path / "test_history.db"

    for index in range(3):
        save_evaluation(
            task_id=f"history-{index}",
            task_description="Test task",
            generated_output=f"Output {index}",
            criteria=[
                {
                    "name": "Requirement met",
                    "passed": index == 2,
                    "explanation": "Test explanation.",
                }
            ],
            score=index / 2,
            decision="pass" if index == 2 else "fail",
            db_path=db_path,
        )

    records = get_recent_evaluations(limit=2, db_path=db_path)

    assert len(records) == 2
    assert records[0]["task_id"] == "history-2"
    assert records[1]["task_id"] == "history-1"
    assert isinstance(records[0]["criteria"], list)


def test_get_recent_evaluations_returns_empty_for_missing_db(tmp_path):
    records = get_recent_evaluations(
        db_path=tmp_path / "missing.db"
    )

    assert records == []


def test_get_recent_evaluations_rejects_invalid_limit():
    with pytest.raises(ValueError, match="Limit must be"):
        get_recent_evaluations(limit=0)
