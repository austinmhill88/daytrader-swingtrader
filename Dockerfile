FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config /app/config
COPY src /app/src
COPY backtest /app/backtest

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs

# Set environment variables (override these at runtime)
ENV APCA_API_KEY_ID=""
ENV APCA_API_SECRET_KEY=""
ENV APCA_API_BASE_URL="https://paper-api.alpaca.markets"

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "src.main", "--config", "config/config.yaml", "--paper"]
