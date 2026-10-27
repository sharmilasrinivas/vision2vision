
import httpx
import pytest

from backend.app.ollama_client import (
    DEFAULT_MODEL,
    OLLAMA_CHAT_URL,
    generate_chat_response,
)


@pytest.mark.asyncio
async def test_generate_chat_response(monkeypatch):
    captured = {}

    async def mock_post(self, url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]

        return httpx.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "message": {
                    "role": "assistant",
                    "content": "PYTEST_OLLAMA_OK",
                },
                "done": True,
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await generate_chat_response("Test prompt")

    assert result == "PYTEST_OLLAMA_OK"
    assert captured["url"] == OLLAMA_CHAT_URL
    assert captured["json"]["model"] == DEFAULT_MODEL
    assert captured["json"]["messages"][0]["content"] == "Test prompt"
    assert captured["json"]["stream"] is False
