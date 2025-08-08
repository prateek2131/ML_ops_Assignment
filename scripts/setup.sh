#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for Docker installation
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    log_error "Please install Docker Desktop for Mac from: https://www.docker.com/products/docker-desktop/"
    log_error "After installation, start Docker Desktop and run this script again"
    exit 1
fi

# Check for Docker Compose installation
if ! command -v docker-compose &> /dev/null; then
    log_error "Docker Compose is not installed"
    log_error "Please install Docker Compose"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    log_error "Please start Docker Desktop and try again"
    exit 1
fi

log_info "Setting up California Housing MLOps project..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_info "Created virtual environment"
fi

source venv/bin/activate

# Install dependencies
log_info "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
log_info "Creating project directories..."
mkdir -p data/{raw,processed} models logs

# Create default .env.production if it doesn't exist
if [ ! -f ".env.production" ]; then
    log_info "Creating default .env.production..."
    echo "API_PORT=8000
MLFLOW_PORT=5001
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOG_LEVEL=INFO" > .env.production
fi

# Start services using Docker Compose
log_info "Starting services with Docker Compose..."
docker-compose -f docker-compose.yml up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 10

log_info "Setup completed successfully!"
log_info "Services are running at:"
log_info "- API: http://localhost:8000"
log_info "- MLflow: http://localhost:5001"
log_info "- Prometheus: http://localhost:9090"
log_info "- Grafana: http://localhost:3000 (admin/admin)"
log_warn "Don't forget to change the Grafana admin password"
