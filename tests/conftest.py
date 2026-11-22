"""Stub the heavy dependencies so tests run without Ollama, ChromaDB data or model downloads."""
import sys
from unittest.mock import MagicMock

for name in ("sentence_transformers", "chromadb"):
    sys.modules[name] = MagicMock()
