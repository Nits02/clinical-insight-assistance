#!/bin/bash

# Clinical Insights Assistant - Azure Deployment Script
# Author: Nitesh Sharma (nitesh.sharma@live.com)
# Description: Automated Azure deployment script for Clinical Insights Assistant

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
RESOURCE_GROUP="clinical-insights-rg"
LOCATION="eastus"
APP_NAME="clinical-insights-assistant"
ACR_PREFIX="clinicalinsights"
KEYVAULT_PREFIX="clinical-kv"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${CYAN}=== $1 ===${NC}"
}

check_prerequisites() {
    log_section "Checking Prerequisites"
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install it first:"
        echo "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if logged in to Azure
    if ! az account show &> /dev/null; then
        log_warning "Not logged in to Azure. Please run 'az login' first."
        exit 1
    fi
    
    log_success "All prerequisites are met."
}

setup_resource_group() {
    log_section "Setting up Resource Group"
    
    # Create resource group if it doesn't exist
    if ! az group show --name $RESOURCE_GROUP &> /dev/null; then
        log_info "Creating resource group: $RESOURCE_GROUP"
        az group create --name $RESOURCE_GROUP --location $LOCATION
        log_success "Resource group created successfully."
    else
        log_info "Resource group $RESOURCE_GROUP already exists."
    fi
}

setup_container_registry() {
    log_section "Setting up Azure Container Registry"
    
    # Generate unique ACR name
    ACR_NAME="${ACR_PREFIX}$(date +%s)"
    
    log_info "Creating Azure Container Registry: $ACR_NAME"
    az acr create \
        --resource-group $RESOURCE_GROUP \
        --name $ACR_NAME \
        --sku Basic \
        --admin-enabled true
    
    # Get ACR details
    ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query "loginServer" --output tsv)
    ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" --output tsv)
    ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)
    
    log_success "Container Registry created: $ACR_LOGIN_SERVER"
    
    # Export for use in other functions
    export ACR_NAME ACR_LOGIN_SERVER ACR_USERNAME ACR_PASSWORD
}

build_and_push_image() {
    log_section "Building and Pushing Docker Image"
    
    # Login to ACR
    log_info "Logging into Azure Container Registry..."
    az acr login --name $ACR_NAME
    
    # Build image
    log_info "Building Docker image..."
    docker build -t $ACR_LOGIN_SERVER/$APP_NAME:latest .
    
    # Push image
    log_info "Pushing image to registry..."
    docker push $ACR_LOGIN_SERVER/$APP_NAME:latest
    
    log_success "Image pushed successfully: $ACR_LOGIN_SERVER/$APP_NAME:latest"
}

setup_key_vault() {
    log_section "Setting up Azure Key Vault"
    
    # Generate unique Key Vault name
    KEYVAULT_NAME="${KEYVAULT_PREFIX}-$(date +%s)"
    
    log_info "Creating Key Vault: $KEYVAULT_NAME"
    az keyvault create \
        --name $KEYVAULT_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION
    
    # Prompt for secrets
    echo -e "\n${YELLOW}Please provide your Azure OpenAI credentials:${NC}"
    read -p "Azure OpenAI API Key: " -s AZURE_OPENAI_API_KEY
    echo
    read -p "Azure OpenAI Endpoint: " AZURE_OPENAI_ENDPOINT
    read -p "Azure OpenAI Deployment Name: " AZURE_OPENAI_DEPLOYMENT
    
    # Store secrets
    log_info "Storing secrets in Key Vault..."
    az keyvault secret set --vault-name $KEYVAULT_NAME --name "azure-openai-api-key" --value "$AZURE_OPENAI_API_KEY"
    az keyvault secret set --vault-name $KEYVAULT_NAME --name "azure-openai-endpoint" --value "$AZURE_OPENAI_ENDPOINT"
    az keyvault secret set --vault-name $KEYVAULT_NAME --name "azure-openai-deployment" --value "$AZURE_OPENAI_DEPLOYMENT"
    
    log_success "Secrets stored in Key Vault: $KEYVAULT_NAME"
    
    # Export for use in other functions
    export KEYVAULT_NAME
}

deploy_container_instance() {
    log_section "Deploying to Azure Container Instances"
    
    # Generate unique DNS name
    DNS_NAME="clinical-insights-$(date +%s)"
    
    log_info "Deploying container instance with DNS: $DNS_NAME"
    az container create \
        --resource-group $RESOURCE_GROUP \
        --name clinical-insights-app \
        --image $ACR_LOGIN_SERVER/$APP_NAME:latest \
        --registry-login-server $ACR_LOGIN_SERVER \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --dns-name-label $DNS_NAME \
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
    FQDN=$(az container show --resource-group $RESOURCE_GROUP --name clinical-insights-app --query "ipAddress.fqdn" --output tsv)
    
    log_success "Application deployed successfully!"
    log_success "URL: http://$FQDN:8501"
    
    export APP_URL="http://$FQDN:8501"
}

deploy_container_apps() {
    log_section "Deploying to Azure Container Apps"
    
    # Add the containerapp extension
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
    ENVIRONMENT_NAME="clinical-env"
    az containerapp env create \
        --name $ENVIRONMENT_NAME \
        --resource-group $RESOURCE_GROUP \
        --logs-workspace-id $WORKSPACE_ID \
        --logs-workspace-key $WORKSPACE_KEY \
        --location $LOCATION
    
    # Create the container app
    az containerapp create \
        --name clinical-insights-containerapp \
        --resource-group $RESOURCE_GROUP \
        --environment $ENVIRONMENT_NAME \
        --image $ACR_LOGIN_SERVER/$APP_NAME:latest \
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
        --resource-group $RESOURCE_GROUP \
        --query properties.configuration.ingress.fqdn \
        --output tsv)
    
    log_success "Container App deployed successfully!"
    log_success "URL: https://$APP_URL"
    
    export APP_URL="https://$APP_URL"
}

verify_deployment() {
    log_section "Verifying Deployment"
    
    log_info "Checking application health..."
    sleep 30  # Wait for container to start
    
    if curl -f -s "$APP_URL/_stcore/health" > /dev/null; then
        log_success "✅ Application is healthy and responding!"
    else
        log_warning "⚠️ Application might still be starting up. Please check manually."
    fi
    
    log_info "Deployment verification completed."
}

setup_monitoring() {
    log_section "Setting up Monitoring (Optional)"
    
    read -p "Do you want to set up Application Insights monitoring? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        APP_INSIGHTS_NAME="clinical-insights-ai"
        az monitor app-insights component create \
            --app $APP_INSIGHTS_NAME \
            --location $LOCATION \
            --resource-group $RESOURCE_GROUP
        
        INSTRUMENTATION_KEY=$(az monitor app-insights component show \
            --app $APP_INSIGHTS_NAME \
            --resource-group $RESOURCE_GROUP \
            --query "instrumentationKey" \
            --output tsv)
        
        log_success "Application Insights created."
        log_info "Instrumentation Key: $INSTRUMENTATION_KEY"
        log_info "Add this to your environment variables for enhanced monitoring."
    fi
}

cleanup_resources() {
    log_section "Cleaning up Resources"
    
    log_warning "This will delete ALL resources in the resource group: $RESOURCE_GROUP"
    read -p "Are you sure you want to continue? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deleting resource group and all resources..."
        az group delete --name $RESOURCE_GROUP --yes --no-wait
        log_success "Cleanup initiated. Resources will be deleted in the background."
    else
        log_info "Cleanup cancelled."
    fi
}

show_deployment_info() {
    log_section "Deployment Summary"
    
    echo -e "${CYAN}🎉 Clinical Insights Assistant Deployment Complete!${NC}"
    echo
    echo -e "${GREEN}📋 Deployment Details:${NC}"
    echo -e "   Resource Group: $RESOURCE_GROUP"
    echo -e "   Location: $LOCATION"
    echo -e "   Container Registry: $ACR_LOGIN_SERVER"
    echo -e "   Key Vault: $KEYVAULT_NAME"
    if [ ! -z "$APP_URL" ]; then
        echo -e "   Application URL: $APP_URL"
    fi
    echo
    echo -e "${YELLOW}🔧 Next Steps:${NC}"
    echo -e "   1. Test your application at the provided URL"
    echo -e "   2. Configure custom domain (optional)"
    echo -e "   3. Set up SSL certificate (recommended for production)"
    echo -e "   4. Configure backup strategies"
    echo -e "   5. Set up CI/CD pipeline"
    echo
    echo -e "${BLUE}📞 Support:${NC}"
    echo -e "   Email: nitesh.sharma@live.com"
    echo -e "   Blog: https://thedataarch.com/"
    echo
}

show_help() {
    echo "Clinical Insights Assistant - Azure Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  aci         Deploy using Azure Container Instances (quick)"
    echo "  apps        Deploy using Azure Container Apps (recommended)"
    echo "  full        Full deployment with all components"
    echo "  cleanup     Delete all Azure resources"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 aci       # Quick deployment with Container Instances"
    echo "  $0 apps      # Production deployment with Container Apps"
    echo "  $0 full      # Complete deployment with monitoring"
    echo ""
}

# Main script logic
case "${1:-help}" in
    aci)
        check_prerequisites
        setup_resource_group
        setup_container_registry
        build_and_push_image
        setup_key_vault
        deploy_container_instance
        verify_deployment
        show_deployment_info
        ;;
    apps)
        check_prerequisites
        setup_resource_group
        setup_container_registry
        build_and_push_image
        setup_key_vault
        deploy_container_apps
        verify_deployment
        show_deployment_info
        ;;
    full)
        check_prerequisites
        setup_resource_group
        setup_container_registry
        build_and_push_image
        setup_key_vault
        deploy_container_apps
        verify_deployment
        setup_monitoring
        show_deployment_info
        ;;
    cleanup)
        cleanup_resources
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac