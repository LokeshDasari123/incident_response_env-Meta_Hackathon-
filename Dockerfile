FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/         ./models/
COPY scenarios/      ./scenarios/
COPY graders/        ./graders/
COPY envs/           ./envs/
COPY server/         ./server/
COPY client/         ./client/
COPY data/processed/ ./data/processed/
COPY openenv.yaml    .

ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=2
ENV LOG_LEVEL=info
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]