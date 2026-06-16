# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Create a non-root user (HF Spaces runs as uid 1000)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Copy and install dependencies first (Docker layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Create a persistent directory for ChromaDB data
RUN mkdir -p /app/chroma_db && chown user:user /app/chroma_db

# Switch to non-root user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Start the application with uvicorn
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]