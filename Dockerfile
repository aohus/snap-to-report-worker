FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to not create virtual env (install globally in container)
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml /app/

# Install dependencies
RUN poetry install --no-root --without dev

COPY src /app/src
COPY assets /app/assets

# Create media directory
RUN mkdir -p /app/media

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
