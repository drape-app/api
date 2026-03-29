# ---- Build stage ----
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir -e "."

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
