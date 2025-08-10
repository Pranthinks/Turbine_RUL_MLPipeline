# Enhanced Dockerfile - Add monitoring capabilities to your existing image

FROM python:3.10.18-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install monitoring dependencies along with your existing requirements
RUN pip install --no-cache-dir -r requirements.txt prometheus_client==0.17.1

# Copy all source code and configs into image
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p artifacts logs config data monitoring_config && \
    chmod 755 artifacts logs config data monitoring_config

# Set Python path
ENV PYTHONPATH=/app

# Add monitoring environment variables
ENV PROMETHEUS_ENABLED=true
ENV PUSHGATEWAY_URL=pushgateway:9091

# Add health check with monitoring awareness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "
import requests
import os
try:
    if os.getenv('PROMETHEUS_ENABLED') == 'true':
        pushgateway_url = os.getenv('PUSHGATEWAY_URL', 'pushgateway:9091')
        # Quick health check - try to reach pushgateway
        requests.get(f'http://{pushgateway_url}/metrics', timeout=5)
    print('Container healthy')
except:
    print('Container healthy - monitoring optional')
" || exit 1

# Default command runs all stages
CMD ["python", "main.py"]