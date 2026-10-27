
import httpx
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_chat_endpoint_success(monkeypatch):
    async def mock_generate_chat_response(prompt):
        assert prompt == "Reply with exactly: TEST_OK"
        return "TEST_OK"

    monkeypatch.setattr(
        "backend.app.main.generate_chat_response",
        mock_generate_chat_response,
    )

    response = client.post(
        "/chat",
        json={"prompt": "Reply with exactly: TEST_OK"},
    )

    assert response.status_code == 200
    assert response.json() == {"response": "TEST_OK"}


def test_chat_endpoint_handles_ollama_error(monkeypatch):
    async def mock_generate_chat_response(prompt):
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        raise httpx.ConnectError("Ollama unavailable", request=request)

    monkeypatch.setattr(
        "backend.app.main.generate_chat_response",
        mock_generate_chat_response,
    )

    response = client.post("/chat", json={"prompt": "Hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Unable to communicate with the local Ollama service."
    )


def test_chat_endpoint_rejects_empty_prompt():
    response = client.post("/chat", json={"prompt": ""})

    assert response.status_code == 422
