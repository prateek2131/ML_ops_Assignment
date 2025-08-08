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

# Verify required environment variables
if [ -z "$DOCKER_USERNAME" ]; then
    log_error "DOCKER_USERNAME environment variable is not set"
    exit 1
fi

ENVIRONMENT="production"
IMAGE_TAG=${1:-latest}
IMAGE_NAME="${DOCKER_USERNAME}/california-housing-api:${IMAGE_TAG}"

log_info "Deploying to ${ENVIRONMENT} environment..."

# Create deployment directory
DEPLOY_DIR="/tmp/housing-api-${ENVIRONMENT}"
mkdir -p ${DEPLOY_DIR}

# Load environment variables from .env.production
if [ -f ".env.production" ]; then
    log_info "Loading production environment variables..."
    set -a
    source .env.production
    set +a
else
    log_warn ".env.production not found, using default values..."
    export API_PORT=8000
    export MLFLOW_PORT=5001
    export PROMETHEUS_PORT=9090
    export GRAFANA_PORT=3000
    export LOG_LEVEL=INFO
fi

# Copy necessary files
log_info "Copying deployment files..."
cp docker-compose.${ENVIRONMENT}.yml ${DEPLOY_DIR}/docker-compose.yml
cp Dockerfile ${DEPLOY_DIR}/
cp -r monitoring ${DEPLOY_DIR}/
cp -r src ${DEPLOY_DIR}/
cp -r models ${DEPLOY_DIR}/
cp -r logs ${DEPLOY_DIR}/
cp requirements.txt ${DEPLOY_DIR}/
cp .env.production ${DEPLOY_DIR}/ 2>/dev/null || :

cd ${DEPLOY_DIR}

# Build and push image
log_info "Building Docker image: ${IMAGE_NAME}"
docker build -t ${IMAGE_NAME} -f Dockerfile .

# Check if we should push the image
if [[ "${ENVIRONMENT}" == "production" ]]; then
    log_info "Pushing Docker image to registry..."
    docker push ${IMAGE_NAME}
fi

# Stop existing containers
log_info "Stopping existing containers..."
docker-compose down || true

# Start new containers
log_info "Starting new containers..."
export DOCKER_IMAGE=${IMAGE_NAME}
docker-compose up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 30

# Health check
for i in {1..10}; do
    if curl -f http://localhost:${API_PORT}/health; then
        echo "API is healthy!"
        break
    else
        echo "Waiting for API to be ready... (attempt ${i}/10)"
        sleep 10
    fi
done

# Install Python and pip if not already installed
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install python3
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        apt-get update && apt-get install -y python3 python3-pip
    fi
fi

# Install required Python package
pip3 install requests

# Run smoke tests
echo "Running smoke tests..."
python3 -c '
import requests
import sys
import os

API_PORT = os.environ.get("API_PORT")

try:
    response = requests.get(f"http://localhost:{API_PORT}/health")
    if response.status_code == 200:
        print("Smoke tests passed!")
        sys.exit(0)
    else:
        print(f"Smoke tests failed! API returned status code {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"Smoke tests failed! Error: {e}")
    sys.exit(1)
'

echo "Deployment to ${ENVIRONMENT} completed successfully!"
