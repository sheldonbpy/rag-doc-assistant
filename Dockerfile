FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN --mount=type=cache,id=pipcache2,target=/root/.cache/pip pip install --default-timeout=120 --retries 10 -r requirements-docker.txt
COPY . .
ENV OLLAMA_URL=http://host.docker.internal:11434/api/chat
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
