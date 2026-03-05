FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install heavy packages first (CPU-only torch, avoids GPU bloat)
RUN pip install --no-cache-dir --timeout=120 torch --index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY . .

RUN mkdir -p indexes static

EXPOSE 8000

CMD ["gunicorn", "api:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300"]