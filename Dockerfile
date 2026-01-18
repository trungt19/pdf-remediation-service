# PDF Remediation Microservice
FROM python:3.11-slim

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    ocrmypdf \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
