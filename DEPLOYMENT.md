# Clinical Insights Assistant - Deployment Guide

This guide provides comprehensive instructions for deploying the Clinical Insights Assistant application using Docker.

## 🚀 Quick Start

### Prerequisites
- Docker Engine (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- At least 4GB RAM available for the container
- OpenAI or Azure OpenAI API credentials

### 1. Clone and Setup

```bash
git clone https://github.com/Nits02/clinical-insight-assistance.git
cd clinical-insight-assistance
```

### 2. Configure Environment

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` file with your credentials:

```bash
# For Azure OpenAI (Recommended)
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini-2024-07-18
OPENAI_PROVIDER=azure

# OR for standard OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_PROVIDER=openai
```

### 3. Deploy with Docker Compose (Recommended)

```bash
# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### 4. Deploy with Docker (Alternative)

```bash
# Build the image
docker build -t clinical-insights-assistant .

# Run the container
docker run -d \
  --name clinical-insights-app \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/memory:/app/memory \
  -v $(pwd)/logs:/app/logs \
  clinical-insights-assistant
```

## 🌐 Access the Application

Once deployed, access the application at:
- **Local**: http://localhost:8501
- **Network**: http://your-server-ip:8501

## 📊 Production Deployment

### Cloud Platforms

#### AWS ECS/Fargate
```bash
# Push to ECR
aws ecr create-repository --repository-name clinical-insights-assistant
docker tag clinical-insights-assistant:latest your-account.dkr.ecr.region.amazonaws.com/clinical-insights-assistant:latest
docker push your-account.dkr.ecr.region.amazonaws.com/clinical-insights-assistant:latest
```

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/clinical-insights-assistant
gcloud run deploy --image gcr.io/your-project/clinical-insights-assistant --platform managed
```

#### Azure Container Instances
```bash
# Build and push to ACR
az acr build --registry your-registry --image clinical-insights-assistant .
az container create --resource-group your-rg --name clinical-insights-app --image your-registry.azurecr.io/clinical-insights-assistant
```

### Environment Variables for Production

```bash
# Security
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Performance
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Logging
LOG_LEVEL=INFO
```

## 🔧 Customization

### Resource Requirements

**Minimum:**
- CPU: 1 vCPU
- RAM: 2GB
- Storage: 10GB

**Recommended:**
- CPU: 2 vCPUs
- RAM: 4GB
- Storage: 50GB

### Docker Compose Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'

services:
  clinical-insights-assistant:
    ports:
      - "8502:8501"  # Use different port
    environment:
      - DEBUG=true
    volumes:
      - ./custom-data:/app/data
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in docker-compose.yml or use:
   docker-compose up -d --scale clinical-insights-assistant=0
   docker-compose up -d
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   docker system prune -f
   docker-compose up -d --memory 4g
   ```

3. **Permission Errors**
   ```bash
   # Fix directory permissions
   chmod -R 755 data memory logs
   ```

### Health Checks

```bash
# Check application health
curl http://localhost:8501/_stcore/health

# View container logs
docker logs clinical-insights-app

# Monitor resource usage
docker stats clinical-insights-app
```

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Network**: Use HTTPS in production with a reverse proxy
3. **Updates**: Regularly update the base image and dependencies
4. **Monitoring**: Implement logging and monitoring solutions

## 📝 Maintenance

### Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup

```bash
# Backup persistent data
tar -czf clinical-backup-$(date +%Y%m%d).tar.gz data/ memory/ logs/
```

## 📞 Support

- **Email**: nitesh.sharma@live.com
- **Blog**: https://thedataarch.com/
- **Issues**: https://github.com/Nits02/clinical-insight-assistance/issues

---

For more detailed documentation, visit the main [README.md](README.md) file.