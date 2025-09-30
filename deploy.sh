#!/bin/bash

# Clinical Insights Assistant - Deployment Script
# Author: Nitesh Sharma (nitesh.sharma@live.com)
# Description: Easy deployment and management script for the Clinical Insights Assistant

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="clinical-insights-assistant"
CONTAINER_NAME="clinical-insights-app"
IMAGE_NAME="clinical-insights-assistant:latest"
DOCKER_COMPOSE_FILE="docker-compose.yml"

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

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "All dependencies are available."
}

check_env_file() {
    if [ ! -f ".env" ]; then
        log_warning ".env file not found."
        if [ -f ".env.example" ]; then
            log_info "Copying .env.example to .env"
            cp .env.example .env
            log_warning "Please edit .env file with your API keys before deployment."
            exit 1
        else
            log_error "Neither .env nor .env.example file found."
            exit 1
        fi
    fi
    log_success ".env file found."
}

build_image() {
    log_info "Building Docker image..."
    docker build -t $IMAGE_NAME .
    log_success "Docker image built successfully."
}

start_services() {
    log_info "Starting services with Docker Compose..."
    docker-compose up -d
    log_success "Services started successfully."
    
    log_info "Waiting for application to be ready..."
    sleep 10
    
    # Health check
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        log_success "Application is running and healthy!"
        log_info "Access the application at: http://localhost:8501"
    else
        log_warning "Application might still be starting up. Check logs with: $0 logs"
    fi
}

stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_success "Services stopped successfully."
}

restart_services() {
    log_info "Restarting services..."
    stop_services
    start_services
}

show_logs() {
    log_info "Showing application logs..."
    docker-compose logs -f $APP_NAME
}

show_status() {
    log_info "Service status:"
    docker-compose ps
    
    log_info "Container stats:"
    if docker ps --format "table {{.Names}}" | grep -q $CONTAINER_NAME; then
        docker stats --no-stream $CONTAINER_NAME
    else
        log_warning "Container is not running."
    fi
}

cleanup() {
    log_info "Cleaning up..."
    docker-compose down --volumes --remove-orphans
    docker image rm $IMAGE_NAME 2>/dev/null || true
    log_success "Cleanup completed."
}

update_application() {
    log_info "Updating application..."
    
    # Pull latest changes (if in git repo)
    if [ -d ".git" ]; then
        log_info "Pulling latest changes from git..."
        git pull origin main
    fi
    
    # Rebuild and restart
    log_info "Rebuilding image..."
    docker-compose build --no-cache
    
    log_info "Restarting services..."
    restart_services
    
    log_success "Application updated successfully."
}

backup_data() {
    log_info "Creating backup..."
    BACKUP_NAME="clinical-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    tar -czf $BACKUP_NAME data/ memory/ logs/ 2>/dev/null || true
    log_success "Backup created: $BACKUP_NAME"
}

show_help() {
    echo "Clinical Insights Assistant - Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start the application"
    echo "  stop        Stop the application"
    echo "  restart     Restart the application"
    echo "  build       Build the Docker image"
    echo "  logs        Show application logs"
    echo "  status      Show service status"
    echo "  update      Update and restart the application"
    echo "  backup      Create a backup of data directories"
    echo "  cleanup     Stop services and remove containers/images"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start the application"
    echo "  $0 logs      # View logs"
    echo "  $0 status    # Check status"
    echo ""
}

# Main script logic
case "${1:-help}" in
    start)
        check_dependencies
        check_env_file
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    build)
        check_dependencies
        build_image
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    update)
        check_dependencies
        update_application
        ;;
    backup)
        backup_data
        ;;
    cleanup)
        cleanup
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