
# Voice2Vision Guardian

A research-oriented, voice-driven multimodal AI system with two
specialized agents:

- Agent A: Understands user requests, generates images, and responds
  through voice and text.
- Agent B: Evaluates task completion, retrieves historical failures
  and successful corrections, and applies guarded self-correction.

## Research Objective

Investigate whether history-aware retrieval and evidence-based
correction improve task success and reliability compared with
direct execution and one-turn evaluation.

## Technology Stack

- Python
- LangGraph
- FastAPI
- Ollama
- Hugging Face Diffusers
- faster-whisper
- Chroma
- Sentence Transformers
- SQLite
- React and Vite
- MLflow
- pytest

## Project Status

Initial repository setup.

## Research Integrity

All reported performance metrics must come from actual experiments.
The evaluator must use fixed criteria and support correction rollback.
