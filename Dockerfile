# Use a lightweight official Python slim image
FROM python:3.11-slim

# Set system environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for compiling certain native C++ wheels if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first to leverage Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source files
COPY src/ /app/src/

# Create folders for data and model persistence inside container
RUN mkdir -p /app/data /app/models

# Expose port 8000 for external traffic
EXPOSE 8000

# Run FastAPI using uvicorn server on container startup
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
