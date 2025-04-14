FROM python:3.10-bullseye
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads translation_logs

# Expose the port for the Flask app
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV HF_API_URL=https://username-universal-translator.hf.space

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "3000", "app:app"]