# Enhanced Dockerfile - Add monitoring capabilities to your existing image

FROM python:3.10.18-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install monitoring dependencies along with your existing requirements
RUN pip install --no-cache-dir -r requirements.txt prometheus_client==0.17.1

# Copy all source code and configs into image
COPY . .

 #Create necessary directories with proper permissions (COMBINED - remove the duplicate)
RUN mkdir -p artifacts logs config data monitoring_config static temp && \
    chmod 755 artifacts logs config data monitoring_config static temp  

# Set Python path
ENV PYTHONPATH=/app

# Add monitoring environment variables
ENV PROMETHEUS_ENABLED=true
ENV PUSHGATEWAY_URL=pushgateway:9091

# Expose FastAPI port
EXPOSE 8000
# Simple health check that works reliably
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import pandas, numpy; print('Container healthy')" || exit 1

# Default command runs all stages
CMD ["python", "main.py"]

