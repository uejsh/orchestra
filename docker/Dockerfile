# Production-Ready Dockerfile for Orchestra
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install Orchestra in editable mode
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 orchestra && \
    chown -R orchestra:orchestra /app
USER orchestra

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "from orchestra.monitoring.health import get_health_checker; import sys; sys.exit(0 if get_health_checker().check_all()['status'] == 'healthy' else 1)"

# Default command (can be overridden)
CMD ["python", "-m", "orchestra.cli"]
