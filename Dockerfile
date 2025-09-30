# Use Python 3.11 slim image as base for smaller image size
FROM python:3.11-slim

# Set metadata
LABEL maintainer="nitesh.sharma@live.com"
LABEL description="Clinical Insights Assistant - AI-powered clinical trial analysis platform"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Install system dependencies required for the application
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    libpq-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directories
RUN mkdir -p /app/memory /app/data /app/logs /app/src/ui

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . .

# Copy environment file (optional - can be overridden at runtime)
COPY .env .env.example

# Set proper permissions for directories
RUN chmod -R 755 /app && \
    chmod -R 777 /app/memory /app/data /app/logs

# Create a non-root user for security
RUN groupadd -r clinicalai && useradd -r -g clinicalai -d /app -s /sbin/nologin clinicalai
RUN chown -R clinicalai:clinicalai /app

# Switch to non-root user
USER clinicalai

# Expose the Streamlit port
EXPOSE 8501

# Health check to ensure the application is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set the default command to run the Streamlit application
CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]