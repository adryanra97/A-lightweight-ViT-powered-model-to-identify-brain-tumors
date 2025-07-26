# Brain Tumor Detection API with Image Upload
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/

# Create necessary directories
RUN mkdir -p uploads logs models

# Copy model files if they exist (optional)
RUN if [ -d "trained_model" ]; then cp -r trained_model/* ./models/ 2>/dev/null || echo "Model copy failed"; fi

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/brain_tumor_vit_model.pth
ENV UPLOAD_FOLDER=/app/uploads
ENV MAX_CONTENT_LENGTH=16777216

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
