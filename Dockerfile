# Use a lightweight Python image for fast builds and small footprints
FROM python:3.10-slim

# Install system-level dependencies for PDF processing (PyPDF2/LangChain)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application logic
COPY . .

# FastAPI default port
EXPOSE 8000

# Bind to 0.0.0.0 to allow traffic from the Docker network
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]