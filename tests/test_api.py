import requests
from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_pipeline_result(monkeypatch):
    fake = {"answer": "x [1]", "sources": [], "refused": False, "reason": "ok"}
    monkeypatch.setattr(api, "answer", lambda q: fake)
    r = client.post("/ask", json={"question": "What is C?"})
    assert r.status_code == 200 and r.json() == fake


def test_empty_question_is_rejected():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_missing_ollama_returns_503(monkeypatch):
    def down(q):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(api, "answer", down)
    r = client.post("/ask", json={"question": "What is C?"})
    assert r.status_code == 503
