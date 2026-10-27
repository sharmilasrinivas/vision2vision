
"""Client for communicating with a locally running Ollama model."""

import httpx

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:3b"


async def generate_chat_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send a prompt to Ollama and return the generated response."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(OLLAMA_CHAT_URL, json=payload)
        response.raise_for_status()
        result = response.json()

    return result["message"]["content"]
