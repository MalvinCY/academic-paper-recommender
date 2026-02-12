# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt

# Copy application code and data
COPY api/ ./api/
COPY data/processed/ ./data/processed/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
