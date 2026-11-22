# RAG Document Assistant (scikit-learn docs)

Answers questions from the scikit-learn User Guide, cites sources, refuses when evidence is weak. Runs fully local with Ollama.

## Setup
```bash
pip install -r requirements.txt
ollama pull llama3.1:8b                # install Ollama first: https://ollama.com (keep it running)
python ingest.py                       # download + chunk + embed (once)
streamlit run app.py                   # chat UI
python eval.py                         # run test set, then score eval_results.csv by hand
```

## How it works
1. `ingest.py`: scrapes ~40 User Guide pages, splits by heading (+ overlap for long sections), embeds with MiniLM, stores in ChromaDB.
2. `rag.py`: top-5 retrieval -> distance filter -> prompt with numbered sources -> the local LLM answers with [n] citations.
3. Guardrails: weak retrieval = refuse (no LLM call); model says INSUFFICIENT_CONTEXT = refuse; fake/missing citations = refuse; prompt-injection rule in system prompt.
4. `eval.py`: 10 questions (answerable, off-topic, injection). Tune `MAX_DISTANCE` in `rag.py` using the results.

## Ideas to extend
Hybrid search (BM25 + vectors), reranker, chunk-size experiments. Try other models via `OLLAMA_MODEL=qwen2.5:7b`.
