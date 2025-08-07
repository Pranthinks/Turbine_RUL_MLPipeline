FROM python:3.10.18-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code and configs into image
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p artifacts logs config data && \
    chmod 755 artifacts logs config data

# Set Python path
ENV PYTHONPATH=/app

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "print('Container healthy')" || exit 1

# Default command runs all stages
CMD ["python", "main.py"]