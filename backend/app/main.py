import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.evaluation import EvaluationCriterion, evaluate_criteria
from backend.app.history_store import (
    get_recent_evaluations,
    retrieve_similar_failures,
    save_evaluation,
)
from backend.app.llm_evaluator import (
    LLMEvaluationRequest,
    evaluate_text_output,
)
from backend.app.ollama_client import generate_chat_response
from backend.app.semantic_retrieval import retrieve_semantic_failures

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


class EvaluateRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=1000)
    criteria: list[EvaluationCriterion] = Field(min_length=1)


class FailureRetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=100)


class SemanticFailureRetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=100)
    min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    """Evaluate supplied criteria and persist the result."""
    result = evaluate_criteria(
        task_id=request.task_id,
        criteria=request.criteria,
        summary=request.summary,
    )

    save_evaluation(
        task_id=request.task_id,
        task_description=request.summary,
        generated_output=request.summary,
        criteria=[
            criterion.model_dump()
            for criterion in result.criteria
        ],
        score=result.score,
        decision=result.decision,
    )

    return result




@app.post("/evaluate-llm")
async def evaluate_llm(request: LLMEvaluationRequest):
    """Evaluate generated text using the local LLM and save the result."""
    try:
        result = await evaluate_text_output(request)

        save_evaluation(
            task_id=request.task_id,
            task_description=request.task_description,
            generated_output=request.generated_output,
            criteria=[
                criterion.model_dump()
                for criterion in result.criteria
            ],
            score=result.score,
            decision=result.decision,
        )

        return result

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Unable to communicate with the local Ollama service.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@app.post("/retrieve-failures")
def retrieve_failures(request: FailureRetrievalRequest):
    """Retrieve relevant past failed evaluations."""
    try:
        return retrieve_similar_failures(
            query=request.query,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post("/retrieve-semantic-failures")
def retrieve_semantic_failures_endpoint(
    request: SemanticFailureRetrievalRequest,
):
    """Retrieve past failed evaluations using semantic similarity."""
    try:
        return retrieve_semantic_failures(
            query=request.query,
            limit=request.limit,
            min_similarity=request.min_similarity,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.get("/history")
def evaluation_history(limit: int = 10):
    """Return recent saved evaluations."""
    return get_recent_evaluations(limit=limit)


