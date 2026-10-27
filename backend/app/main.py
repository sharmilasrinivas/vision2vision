import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.ollama_client import generate_chat_response

app = FastAPI(
    title="Voice2Vision Guardian API",
    description=(
        "A voice-driven multimodal AI system with "
        "history-aware evaluation and guarded self-correction."
    ),
    version="0.1.0",
)

# Development configuration.
# Restrict allowed origins before public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": "Voice2Vision Guardian",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "backend",
    }


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str]:
    """Generate a response using the local Ollama model."""
    try:
        answer = await generate_chat_response(request.prompt)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Unable to communicate with the local Ollama service.",
        ) from exc

    return {"response": answer}

