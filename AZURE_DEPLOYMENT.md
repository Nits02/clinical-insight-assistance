# 🚀 Clinical Insights Assistant - Azure Deployment Guide

Welcome to the Clinical Insights Assistant! This guide provides both **quick deployment options** and **detailed step-by-step instructions** for deploying on Microsoft Azure.

---

## 🎯 **QUICK START** (Choose Your Speed!)

### 📋 Prerequisites Checklist
Before starting, ensure you have:
- [ ] **Azure CLI installed** - [Installation Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)  
- [ ] **Docker installed** - [Get Docker](https://docs.docker.com/get-docker/)
- [ ] **Azure subscription** - [Free account](https://azure.microsoft.com/en-us/free/)
- [ ] **Azure OpenAI access** - [Request access](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service/)

### ⚡ **Option 1: Super Quick (5 minutes)**
```bash
# Login to Azure
az login

# Deploy with Container Instances (simplest)
./deploy-azure.sh aci
```

### 🏗️ **Option 2: Production Ready (10 minutes)**  
```bash
# Login to Azure
az login

# Deploy with Container Apps (recommended)
./deploy-azure.sh apps
```

### 🌟 **Option 3: Full Stack (15 minutes)**
```bash
# Login to Azure
az login

# Deploy with all monitoring and features
./deploy-azure.sh full
```

### 🔐 **Required Information**
During deployment, you'll be prompted for:
1. **Azure OpenAI API Key** - From your Azure OpenAI resource
2. **Azure OpenAI Endpoint** - Your service endpoint URL  
3. **Azure OpenAI Deployment Name** - Your model deployment name

### 🧹 **Cleanup**
To remove all Azure resources:
```bash
./deploy-azure.sh cleanup
```

---

## � **REAL-WORLD DEPLOYMENT WALKTHROUGH** (Step-by-Step)

This section documents the actual commands and steps used for a successful deployment, including troubleshooting fixes.

### **Step 1: Prerequisites Verification**
```bash
# Check Azure CLI version
az --version

# Check Docker installation
docker --version

# Verify Azure login status
az account show

# Start Docker if not running (macOS)
open -a Docker

# Wait for Docker to start
sleep 15 && docker info > /dev/null 2>&1 && echo "Docker is now running"
```

### **Step 2: Upgrade Azure CLI (Recommended)**
```bash
# Upgrade to latest version for best compatibility
az upgrade --yes
```

### **Step 3: Create Resource Group and Container Registry**
```bash
# Set variables (replace with your values)
RESOURCE_GROUP="clinical-insights-rg"
LOCATION="eastus"
ACR_NAME="clinicalinsights$(date +%s)"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Get ACR details
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)
```

### **Step 4: Build and Push Container Image**
```bash
# Login to Azure Container Registry
az acr login --name $ACR_NAME

# Build image with correct platform for Azure Container Apps
docker build --platform linux/amd64 -t $ACR_LOGIN_SERVER/clinical-insights-assistant:latest .

# Push image to registry
docker push $ACR_LOGIN_SERVER/clinical-insights-assistant:latest
```

### **Step 5: Setup Key Vault for Secrets**
```bash
# Create Key Vault
KEYVAULT_NAME="clinical-kv-$(date +%s)"
az keyvault create \
    --name $KEYVAULT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# Assign Key Vault Secrets Officer role to current user
az role assignment create \
    --role "Key Vault Secrets Officer" \
    --assignee $(az account show --query "user.name" --output tsv) \
    --scope "/subscriptions/$(az account show --query "id" --output tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEYVAULT_NAME"

# Wait for role assignment to propagate
sleep 30

# Store secrets in Key Vault
az keyvault secret set --vault-name $KEYVAULT_NAME --name "azure-openai-api-key" --value "YOUR_ACTUAL_API_KEY"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "azure-openai-endpoint" --value "https://ai-proxy.lab.epam.com"
az keyvault secret set --vault-name $KEYVAULT_NAME --name "azure-openai-deployment" --value "gpt-4o-mini-2024-07-18"
```

### **Step 6: Setup Container Apps Environment**
```bash
# Install Container Apps extension
az extension add --name containerapp --upgrade

# Register providers
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

# Create Log Analytics workspace
WORKSPACE_NAME="clinical-logs-$(date +%s)"
az monitor log-analytics workspace create \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $WORKSPACE_NAME

# Get workspace details
WORKSPACE_ID=$(az monitor log-analytics workspace show --query customerId --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --output tsv)
WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys --query primarySharedKey --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --output tsv)

# Create Container Apps environment
az containerapp env create \
    --name clinical-env \
    --resource-group $RESOURCE_GROUP \
    --logs-workspace-id $WORKSPACE_ID \
    --logs-workspace-key $WORKSPACE_KEY \
    --location $LOCATION
```

### **Step 7: Deploy Container App**
```bash
# Create Container App
az containerapp create \
    --name clinical-insights-containerapp \
    --resource-group $RESOURCE_GROUP \
    --environment clinical-env \
    --image $ACR_LOGIN_SERVER/clinical-insights-assistant:latest \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --target-port 8501 \
    --ingress external \
    --cpu 2.0 \
    --memory 4.0Gi \
    --min-replicas 1 \
    --max-replicas 3 \
    --env-vars \
        STREAMLIT_SERVER_PORT=8501 \
        STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        PYTHONPATH=/app/src:/app \
        OPENAI_PROVIDER=azure \
        AZURE_OPENAI_API_KEY=YOUR_ACTUAL_API_KEY \
        AZURE_OPENAI_ENDPOINT=https://ai-proxy.lab.epam.com \
        AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini-2024-07-18
```

### **Step 8: Verify Deployment**
```bash
# Get the application URL
APP_URL=$(az containerapp show \
    --name clinical-insights-containerapp \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn \
    --output tsv)

echo "Application URL: https://$APP_URL"

# Test the application
curl -s -o /dev/null -w "%{http_code}" https://$APP_URL/
curl -s -I https://$APP_URL/_stcore/health

# Check container logs
az containerapp logs show --name clinical-insights-containerapp --resource-group $RESOURCE_GROUP --tail 10
```

### **🚨 Common Issues and Fixes**

#### **Issue 1: F-string Syntax Errors**
**Problem**: `SyntaxError: f-string: unmatched '('`
**Solution**: Fix nested quotes in f-strings
```python
# BROKEN:
st.write(f"**Visit:** {issue.visit_number if hasattr(issue, "visit_number") and issue.visit_number else "N/A"}")

# FIXED:
visit_num = issue.visit_number if hasattr(issue, 'visit_number') and issue.visit_number else "N/A"
st.write(f"**Visit:** {visit_num}")
```

#### **Issue 2: Key Vault Permission Denied**
**Problem**: `Caller is not authorized to perform action`
**Solution**: Assign proper RBAC role
```bash
az role assignment create \
    --role "Key Vault Secrets Officer" \
    --assignee $(az account show --query "user.name" --output tsv) \
    --scope "/subscriptions/$(az account show --query "id" --output tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.KeyVault/vaults/$KEYVAULT_NAME"
```

#### **Issue 3: Container Platform Mismatch**
**Problem**: `no child with platform linux/amd64`
**Solution**: Build with correct platform
```bash
docker build --platform linux/amd64 -t $ACR_LOGIN_SERVER/clinical-insights-assistant:latest .
```

#### **Issue 4: Cached Container Image**
**Problem**: Old image still running after updates
**Solution**: Delete and recreate container app
```bash
az containerapp delete --name clinical-insights-containerapp --resource-group $RESOURCE_GROUP --yes
# Then recreate with updated image
```

### **💡 Pro Tips**
1. **Always use `--platform linux/amd64`** when building for Azure Container Apps
2. **Wait 30 seconds** after RBAC role assignments before using Key Vault
3. **Check container logs** if the app doesn't start: `az containerapp logs show`
4. **Use environment variables** instead of hardcoded secrets
5. **Monitor costs** with Azure Cost Management

---

## �📚 **DETAILED MANUAL DEPLOYMENT** (For Advanced Users)

If you prefer manual control or want to understand each step, continue with the detailed instructions below.

## 🌐 Azure Deployment Options Overview

### 1. **Azure Container Instances (ACI)** - Quick & Simple
- **Best for:** Development, testing, small-scale production
- **Cost:** Pay-per-second billing
- **Scaling:** Manual scaling
- **Setup Time:** 5-10 minutes

### 2. **Azure Container Apps** - Modern Serverless
- **Best for:** Production workloads with auto-scaling
- **Cost:** Pay for actual usage
- **Scaling:** Automatic scaling (0 to N)
- **Setup Time:** 15-20 minutes

### 3. **Azure App Service** - Managed Platform
- **Best for:** Enterprise applications with DevOps integration
- **Cost:** Fixed pricing tiers
- **Scaling:** Built-in auto-scaling
- **Setup Time:** 20-30 minutes

### 4. **Azure Kubernetes Service (AKS)** - Full Orchestration
- **Best for:** Large-scale, multi-service applications
- **Cost:** Node-based pricing
- **Scaling:** Advanced orchestration
- **Setup Time:** 45-60 minutes

---

## 🚀 Method 1: Azure Container Instances (Recommended for Quick Start)

### **Prerequisites**
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set your subscription (if you have multiple)
az account set --subscription "your-subscription-id"
```

### **Step 1: Create Resource Group**
```bash
# Create a resource group
az group create \
    --name clinical-insights-rg \
    --location eastus

# Verify creation
az group show --name clinical-insights-rg --output table
```

### **Step 2: Create Azure Container Registry (ACR)**
```bash
# Create ACR (replace 'yourname' with your unique identifier)
ACR_NAME="clinicalinsights$(date +%s)"
az acr create \
    --resource-group clinical-insights-rg \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group clinical-insights-rg --query "loginServer" --output tsv)
echo "ACR Login Server: $ACR_LOGIN_SERVER"

# Login to ACR
az acr login --name $ACR_NAME
```

### **Step 3: Build and Push Docker Image**
```bash
# Navigate to your project directory
cd /path/to/clinical-insight-assistance

# Build and tag the image
docker build -t $ACR_LOGIN_SERVER/clinical-insights-assistant:latest .

# Push the image to ACR
docker push $ACR_LOGIN_SERVER/clinical-insights-assistant:latest

# Verify the image
az acr repository list --name $ACR_NAME --output table
```

### **Step 4: Create Key Vault for Secrets**
```bash
# Create Key Vault
KEYVAULT_NAME="clinical-kv-$(date +%s)"
az keyvault create \
    --name $KEYVAULT_NAME \
    --resource-group clinical-insights-rg \
    --location eastus

# Add your OpenAI secrets
az keyvault secret set \
    --vault-name $KEYVAULT_NAME \
    --name "azure-openai-api-key" \
    --value "your-azure-openai-api-key"

az keyvault secret set \
    --vault-name $KEYVAULT_NAME \
    --name "azure-openai-endpoint" \
    --value "https://your-endpoint.openai.azure.com"

az keyvault secret set \
    --vault-name $KEYVAULT_NAME \
    --name "azure-openai-deployment" \
    --value "gpt-4o-mini-2024-07-18"
```

### **Step 5: Deploy Container Instance**
```bash
# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)

# Deploy to ACI
az container create \
    --resource-group clinical-insights-rg \
    --name clinical-insights-app \
    --image $ACR_LOGIN_SERVER/clinical-insights-assistant:latest \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --dns-name-label clinical-insights-$(date +%s) \
    --ports 8501 \
    --cpu 2 \
    --memory 4 \
    --environment-variables \
        STREAMLIT_SERVER_PORT=8501 \
        STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        PYTHONPATH=/app/src:/app \
        OPENAI_PROVIDER=azure \
    --secure-environment-variables \
        AZURE_OPENAI_API_KEY="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-api-key --query value --output tsv)" \
        AZURE_OPENAI_ENDPOINT="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-endpoint --query value --output tsv)" \
        AZURE_OPENAI_DEPLOYMENT_NAME="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-deployment --query value --output tsv)"

# Get the public URL
FQDN=$(az container show --resource-group clinical-insights-rg --name clinical-insights-app --query "ipAddress.fqdn" --output tsv)
echo "Application URL: http://$FQDN:8501"
```

### **Step 6: Verify Deployment**
```bash
# Check container status
az container show \
    --resource-group clinical-insights-rg \
    --name clinical-insights-app \
    --query "{ProvisioningState:provisioningState,State:containers[0].instanceView.currentState.state}" \
    --output table

# View logs
az container logs \
    --resource-group clinical-insights-rg \
    --name clinical-insights-app

# Test health endpoint
curl http://$FQDN:8501/_stcore/health
```

---

## 🔄 Method 2: Azure Container Apps (Recommended for Production)

### **Step 1: Enable Container Apps Extension**
```bash
# Add the containerapp extension
az extension add --name containerapp --upgrade

# Register providers
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

### **Step 2: Create Container Apps Environment**
```bash
# Create Log Analytics workspace
WORKSPACE_NAME="clinical-logs-$(date +%s)"
az monitor log-analytics workspace create \
    --resource-group clinical-insights-rg \
    --workspace-name $WORKSPACE_NAME

# Get workspace details
WORKSPACE_ID=$(az monitor log-analytics workspace show --query customerId --resource-group clinical-insights-rg --workspace-name $WORKSPACE_NAME --output tsv)
WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys --query primarySharedKey --resource-group clinical-insights-rg --workspace-name $WORKSPACE_NAME --output tsv)

# Create Container Apps environment
ENVIRONMENT_NAME="clinical-env"
az containerapp env create \
    --name $ENVIRONMENT_NAME \
    --resource-group clinical-insights-rg \
    --logs-workspace-id $WORKSPACE_ID \
    --logs-workspace-key $WORKSPACE_KEY \
    --location eastus
```

### **Step 3: Deploy Container App**
```bash
# Create the container app
az containerapp create \
    --name clinical-insights-containerapp \
    --resource-group clinical-insights-rg \
    --environment $ENVIRONMENT_NAME \
    --image $ACR_LOGIN_SERVER/clinical-insights-assistant:latest \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --target-port 8501 \
    --ingress external \
    --cpu 2.0 \
    --memory 4.0Gi \
    --min-replicas 1 \
    --max-replicas 10 \
    --env-vars \
        STREAMLIT_SERVER_PORT=8501 \
        STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        PYTHONPATH=/app/src:/app \
        OPENAI_PROVIDER=azure \
    --secrets \
        azure-openai-key="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-api-key --query value --output tsv)" \
        azure-openai-endpoint="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-endpoint --query value --output tsv)" \
        azure-openai-deployment="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-deployment --query value --output tsv)" \
    --env-vars \
        AZURE_OPENAI_API_KEY=secretref:azure-openai-key \
        AZURE_OPENAI_ENDPOINT=secretref:azure-openai-endpoint \
        AZURE_OPENAI_DEPLOYMENT_NAME=secretref:azure-openai-deployment

# Get the application URL
APP_URL=$(az containerapp show \
    --name clinical-insights-containerapp \
    --resource-group clinical-insights-rg \
    --query properties.configuration.ingress.fqdn \
    --output tsv)

echo "Application URL: https://$APP_URL"
```

---

## 🛠️ Method 3: Azure App Service with Docker

### **Step 1: Create App Service Plan**
```bash
# Create App Service Plan (Linux)
az appservice plan create \
    --name clinical-insights-plan \
    --resource-group clinical-insights-rg \
    --location eastus \
    --is-linux \
    --sku B2

# Verify creation
az appservice plan show \
    --name clinical-insights-plan \
    --resource-group clinical-insights-rg \
    --output table
```

### **Step 2: Create Web App**
```bash
# Create the web app
az webapp create \
    --resource-group clinical-insights-rg \
    --plan clinical-insights-plan \
    --name clinical-insights-webapp-$(date +%s) \
    --deployment-container-image-name $ACR_LOGIN_SERVER/clinical-insights-assistant:latest

# Configure container settings
WEBAPP_NAME=$(az webapp list --resource-group clinical-insights-rg --query "[0].name" --output tsv)

az webapp config container set \
    --name $WEBAPP_NAME \
    --resource-group clinical-insights-rg \
    --docker-custom-image-name $ACR_LOGIN_SERVER/clinical-insights-assistant:latest \
    --docker-registry-server-url https://$ACR_LOGIN_SERVER \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD

# Configure app settings
az webapp config appsettings set \
    --resource-group clinical-insights-rg \
    --name $WEBAPP_NAME \
    --settings \
        WEBSITES_PORT=8501 \
        STREAMLIT_SERVER_PORT=8501 \
        STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
        PYTHONPATH=/app/src:/app \
        OPENAI_PROVIDER=azure \
        AZURE_OPENAI_API_KEY="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-api-key --query value --output tsv)" \
        AZURE_OPENAI_ENDPOINT="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-endpoint --query value --output tsv)" \
        AZURE_OPENAI_DEPLOYMENT_NAME="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-deployment --query value --output tsv)"

# Get the URL
WEBAPP_URL=$(az webapp show --name $WEBAPP_NAME --resource-group clinical-insights-rg --query "hostNames[0]" --output tsv)
echo "Application URL: https://$WEBAPP_URL"
```

---

## ⚙️ Method 4: Azure Kubernetes Service (AKS)

### **Step 1: Create AKS Cluster**
```bash
# Create AKS cluster
az aks create \
    --resource-group clinical-insights-rg \
    --name clinical-insights-aks \
    --node-count 2 \
    --node-vm-size Standard_B2s \
    --generate-ssh-keys \
    --attach-acr $ACR_NAME

# Get credentials
az aks get-credentials \
    --resource-group clinical-insights-rg \
    --name clinical-insights-aks
```

### **Step 2: Create Kubernetes Manifests**
```bash
# Create namespace
kubectl create namespace clinical-insights

# Create secret for environment variables
kubectl create secret generic clinical-secrets \
    --from-literal=AZURE_OPENAI_API_KEY="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-api-key --query value --output tsv)" \
    --from-literal=AZURE_OPENAI_ENDPOINT="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-endpoint --query value --output tsv)" \
    --from-literal=AZURE_OPENAI_DEPLOYMENT_NAME="$(az keyvault secret show --vault-name $KEYVAULT_NAME --name azure-openai-deployment --query value --output tsv)" \
    --namespace clinical-insights
```

Create Kubernetes deployment file:
```yaml
# Save as k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clinical-insights-deployment
  namespace: clinical-insights
spec:
  replicas: 2
  selector:
    matchLabels:
      app: clinical-insights
  template:
    metadata:
      labels:
        app: clinical-insights
    spec:
      containers:
      - name: clinical-insights
        image: $ACR_LOGIN_SERVER/clinical-insights-assistant:latest
        ports:
        - containerPort: 8501
        env:
        - name: STREAMLIT_SERVER_PORT
          value: "8501"
        - name: STREAMLIT_SERVER_ADDRESS
          value: "0.0.0.0"
        - name: PYTHONPATH
          value: "/app/src:/app"
        - name: OPENAI_PROVIDER
          value: "azure"
        - name: AZURE_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: clinical-secrets
              key: AZURE_OPENAI_API_KEY
        - name: AZURE_OPENAI_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: clinical-secrets
              key: AZURE_OPENAI_ENDPOINT
        - name: AZURE_OPENAI_DEPLOYMENT_NAME
          valueFrom:
            secretKeyRef:
              name: clinical-secrets
              key: AZURE_OPENAI_DEPLOYMENT_NAME
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: clinical-insights-service
  namespace: clinical-insights
spec:
  selector:
    app: clinical-insights
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer
```

### **Step 3: Deploy to AKS**
```bash
# Replace placeholder and deploy
sed "s|\$ACR_LOGIN_SERVER|$ACR_LOGIN_SERVER|g" k8s-deployment.yaml | kubectl apply -f -

# Get external IP
kubectl get service clinical-insights-service --namespace clinical-insights

# Wait for external IP and get URL
EXTERNAL_IP=$(kubectl get service clinical-insights-service --namespace clinical-insights --output jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Application URL: http://$EXTERNAL_IP"
```

---

## 🔍 Monitoring and Management

### **1. Application Insights Setup**
```bash
# Create Application Insights
APP_INSIGHTS_NAME="clinical-insights-ai"
az monitor app-insights component create \
    --app $APP_INSIGHTS_NAME \
    --location eastus \
    --resource-group clinical-insights-rg

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
    --app $APP_INSIGHTS_NAME \
    --resource-group clinical-insights-rg \
    --query "instrumentationKey" \
    --output tsv)

echo "Add this to your environment variables:"
echo "APPINSIGHTS_INSTRUMENTATIONKEY=$INSTRUMENTATION_KEY"
```

### **2. Log Analytics Queries**
```kusto
// View application logs
ContainerInstanceLog_CL
| where ContainerGroup_s == "clinical-insights-app"
| order by TimeGenerated desc

// Monitor performance
ContainerInstanceLog_CL
| where Message contains "INFO"
| summarize count() by bin(TimeGenerated, 1h)
```

### **3. Health Monitoring**
```bash
# Create health check script
cat > health-check.sh << 'EOF'
#!/bin/bash
URL=$1
HEALTH_ENDPOINT="$URL/_stcore/health"

if curl -f -s $HEALTH_ENDPOINT > /dev/null; then
    echo "✅ Application is healthy"
    exit 0
else
    echo "❌ Application health check failed"
    exit 1
fi
EOF

chmod +x health-check.sh

# Use it
./health-check.sh "http://your-app-url:8501"
```

---

## 🔐 Security Best Practices

### **1. Network Security**
```bash
# Create Network Security Group
az network nsg create \
    --resource-group clinical-insights-rg \
    --name clinical-insights-nsg

# Allow HTTPS traffic only
az network nsg rule create \
    --resource-group clinical-insights-rg \
    --nsg-name clinical-insights-nsg \
    --name AllowHTTPS \
    --protocol tcp \
    --priority 1000 \
    --destination-port-range 443 \
    --access allow
```

### **2. SSL/TLS Configuration**
```bash
# For App Service - Enable HTTPS only
az webapp update \
    --resource-group clinical-insights-rg \
    --name $WEBAPP_NAME \
    --https-only true

# Create custom domain and SSL (optional)
az webapp config hostname add \
    --resource-group clinical-insights-rg \
    --webapp-name $WEBAPP_NAME \
    --hostname your-custom-domain.com
```

### **3. Managed Identity**
```bash
# Enable system-assigned managed identity
az webapp identity assign \
    --resource-group clinical-insights-rg \
    --name $WEBAPP_NAME

# Grant Key Vault access
PRINCIPAL_ID=$(az webapp identity show --resource-group clinical-insights-rg --name $WEBAPP_NAME --query principalId --output tsv)

az keyvault set-policy \
    --name $KEYVAULT_NAME \
    --object-id $PRINCIPAL_ID \
    --secret-permissions get list
```

---

## 💰 Cost Optimization

### **1. Pricing Comparison**
| Service | Cost (Monthly Est.) | Use Case |
|---------|-------------------|----------|
| Container Instances | $30-50 | Development/Testing |
| Container Apps | $50-100 | Production (Auto-scaling) |
| App Service B2 | $60-80 | Enterprise |
| AKS (2 nodes) | $150-200 | Large Scale |

### **2. Cost-Saving Tips**
```bash
# Use Azure Reserved Instances for predictable workloads
az reservations reservation-order purchase

# Set up auto-shutdown for development
az webapp config appsettings set \
    --resource-group clinical-insights-rg \
    --name $WEBAPP_NAME \
    --settings AUTO_SHUTDOWN_ENABLED=true

# Monitor costs
az consumption usage list \
    --start-date 2024-01-01 \
    --end-date 2024-01-31
```

---

## 🛠️ Troubleshooting

### **Common Issues and Solutions**

1. **Container startup issues**:
```bash
# Check container logs
az container logs --resource-group clinical-insights-rg --name clinical-insights-app

# Debug with exec
az container exec --resource-group clinical-insights-rg --name clinical-insights-app --exec-command "/bin/bash"
```

2. **OpenAI API connection issues**:
```bash
# Test connectivity
az container exec \
    --resource-group clinical-insights-rg \
    --name clinical-insights-app \
    --exec-command "curl -I https://api.openai.com/v1/models"
```

3. **Port configuration issues**:
```bash
# Verify port configuration
az container show \
    --resource-group clinical-insights-rg \
    --name clinical-insights-app \
    --query "containers[0].ports"
```

### **Support Resources**
- **Azure Documentation**: https://docs.microsoft.com/azure/
- **Azure CLI Reference**: https://docs.microsoft.com/cli/azure/
- **Container Instances**: https://docs.microsoft.com/azure/container-instances/
- **Container Apps**: https://docs.microsoft.com/azure/container-apps/

---

## ❓ **NEED HELP?**

- 📧 **Email**: [nitesh.sharma@live.com](mailto:nitesh.sharma@live.com)
- 📖 **Blog**: [https://thedataarch.com/](https://thedataarch.com/)
- 🐛 **Issues**: Check the troubleshooting section above
- 💬 **Questions**: Feel free to reach out for deployment support

---

## 🎯 **NEXT STEPS**

1. **Set up CI/CD Pipeline** with Azure DevOps or GitHub Actions
2. **Configure Custom Domain** and SSL certificates  
3. **Implement Auto-scaling** based on usage patterns
4. **Set up Monitoring** with Application Insights
5. **Configure Backup** strategies for persistent data

---

## � **SUCCESSFUL DEPLOYMENT EXAMPLE**

Here's a real example of a successful deployment completed on October 1, 2025:

### **Final Deployment Results**
```bash
# Successful deployment output
Container app created. Access your app at https://clinical-insights-containerapp.ashyground-84f0e362.eastus.azurecontainerapps.io/

# Resource Summary:
Resource Group: clinical-insights-rg
Location: East US
Container Registry: clinicalinsights1759285118.azurecr.io
Key Vault: clinical-kv-1759285305
Container App: clinical-insights-containerapp
Environment: clinical-env
```

### **Verification Tests**
```bash
# Health check passed ✅
$ curl -s -o /dev/null -w "%{http_code}" https://clinical-insights-containerapp.ashyground-84f0e362.eastus.azurecontainerapps.io/
200

# Streamlit health endpoint passed ✅
$ curl -s -I https://clinical-insights-containerapp.ashyground-84f0e362.eastus.azurecontainerapps.io/_stcore/health
HTTP/2 200 
server: TornadoServer/6.5.2
content-type: text/html; charset=UTF-8

# Container logs show successful startup ✅
{"Log": "F   You can now view your Streamlit app in your browser."}
{"Log": "F   URL: http://0.0.0.0:8501"}
```

### **Total Deployment Time**: ~15 minutes
### **Total Resources Created**: 5 Azure resources
### **Application Status**: ✅ **LIVE AND FUNCTIONAL**

---

## �🎉 **SUMMARY**

**For Quick Deployment:** Use the automated script at the top of this guide
**For Custom Deployment:** Follow the detailed manual instructions  
**For Real-World Experience:** Follow the step-by-step walkthrough above
**For Production:** Consider Container Apps or App Service with monitoring

Your Clinical Insights Assistant is now ready for production deployment on Azure! 🚀

### **🔗 Quick Links**
- **Live Example**: https://clinical-insights-containerapp.ashyground-84f0e362.eastus.azurecontainerapps.io/
- **Azure Portal**: https://portal.azure.com/
- **Container Apps Documentation**: https://docs.microsoft.com/azure/container-apps/
- **Troubleshooting**: See the "Common Issues and Fixes" section above