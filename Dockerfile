FROM python:3.11-slim

# System deps for psycopg binary + asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir "psycopg[binary]" psycopg-pool apscheduler

# Copy source
COPY src/ ./src/
COPY setup_db.py .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "supply_chain.main:app", "--host", "0.0.0.0", "--port", "8000"]
