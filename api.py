"""FastAPI wrapper around the RAG pipeline. Run: uvicorn api:app --reload"""
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag import answer

app = FastAPI(title="scikit-learn Docs RAG API", version="1.0")


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    try:
        return answer(q.question.strip())
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="LLM backend (Ollama) is not reachable. Is it running?")
