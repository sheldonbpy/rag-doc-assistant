import pytest
import rag


def make_chunks(n=3, distance=0.2):
    return [{"text": f"text {i}", "distance": distance, "url": f"http://x/{i}", "title": "T", "section": "S"}
            for i in range(n)]


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


def fake_llm(monkeypatch, text):
    monkeypatch.setattr(rag.requests, "post", lambda *a, **k: FakeResponse(text))


def test_weak_retrieval_refuses_without_calling_llm(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q: make_chunks(distance=0.9))

    def boom(*a, **k):
        raise AssertionError("LLM must not be called when retrieval is weak")

    monkeypatch.setattr(rag.requests, "post", boom)
    r = rag.answer("best pizza in Kumasi?")
    assert r["refused"] and r["reason"] == "weak_retrieval" and r["sources"] == []


def test_model_refusal_is_respected(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q: make_chunks())
    fake_llm(monkeypatch, "INSUFFICIENT_CONTEXT")
    r = rag.answer("Does sklearn support GPUs?")
    assert r["refused"] and r["reason"] == "model_refused"


def test_answer_without_citation_is_refused(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q: make_chunks())
    fake_llm(monkeypatch, "C controls regularisation.")
    r = rag.answer("What is C?")
    assert r["refused"] and r["reason"] == "citation_check_failed"


def test_fake_citation_number_is_refused(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q: make_chunks(n=3))
    fake_llm(monkeypatch, "C controls regularisation [9].")
    r = rag.answer("What is C?")
    assert r["refused"] and r["reason"] == "citation_check_failed"


def test_valid_answer_returns_cited_sources(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q: make_chunks(n=3))
    fake_llm(monkeypatch, "C trades off errors against simplicity [1]. A low C is smoother [2].")
    r = rag.answer("What is C?")
    assert not r["refused"] and r["reason"] == "ok"
    assert [s["n"] for s in r["sources"]] == [1, 2]


def test_only_cited_sources_are_returned(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q: make_chunks(n=5))
    fake_llm(monkeypatch, "C trades off errors against simplicity [3].")
    r = rag.answer("What is C?")
    assert [s["n"] for s in r["sources"]] == [3]


def test_clean_answer_removes_repeated_sentence():
    text = ("You should use LinearSVC instead of SVC when you need a faster implementation of Support "
            "Vector Classification for the case of a linear kernel [1]. LinearSVC is another (faster) "
            "implementation of Support Vector Classification for the case of a linear kernel [1]. "
            "It also lacks some of the attributes of SVC, like support_ [1].")
    cleaned = rag.clean_answer(text)
    assert cleaned.count("linear kernel") == 1
    assert "support_" in cleaned


def test_clean_answer_keeps_distinct_sentences():
    text = ("Precision is about avoiding false positives [1]. Recall is about avoiding false negatives [3].")
    assert rag.clean_answer(text) == text
