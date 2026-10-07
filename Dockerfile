FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY run_benchmark.py ./run_benchmark.py

# results dir is a mounted volume in compose; ensure it exists
RUN mkdir -p results/figures

CMD ["python3", "run_benchmark.py"]
