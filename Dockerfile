# Use a lightweight python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (creates a virtualenv under /app/.venv)
RUN uv sync --no-dev --no-install-project

# Copy application source code and models
COPY src ./src
COPY models ./models
COPY main.py ./

# Re-run sync to install the project itself
RUN uv sync --no-dev

# Final runtime image
FROM python:3.13-slim-bookworm

# Set metadata
LABEL maintainer="Senior AI Architect"
LABEL description="IT Support Ticket Classifier API"

WORKDIR /app

# Copy the virtual environment and project from the builder
COPY --from=builder /app /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose the default FastAPI port
EXPOSE 8000

# Start the server using uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
