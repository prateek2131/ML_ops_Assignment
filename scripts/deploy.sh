#!/bin/bash

set -e

ENVIRONMENT=${1:-staging}
IMAGE_TAG=${2:-latest}
DOCKER_USERNAME=${DOCKER_USERNAME}
IMAGE_NAME="${DOCKER_USERNAME}/california-housing-api:${IMAGE_TAG}"

echo "Deploying to ${ENVIRONMENT} environment..."

# Create deployment directory
DEPLOY_DIR="/tmp/housing-api-${ENVIRONMENT}"
mkdir -p ${DEPLOY_DIR}

# Copy necessary files
cp docker/docker-compose.${ENVIRONMENT}.yml ${DEPLOY_DIR}/docker-compose.yml
cp -r docker ${DEPLOY_DIR}/
cp -r monitoring ${DEPLOY_DIR}/
cp -r src ${DEPLOY_DIR}/
cp -r models ${DEPLOY_DIR}/
cp -r logs ${DEPLOY_DIR}/
cp requirements.txt ${DEPLOY_DIR}/

cd ${DEPLOY_DIR}

# Set environment-specific configurations
if [ "${ENVIRONMENT}" = "production" ]; then
    export API_PORT=8000
    export MLFLOW_PORT=5002
    export PROMETHEUS_PORT=9090
    export GRAFANA_PORT=3000
else
    export API_PORT=8001
    export MLFLOW_PORT=5002
    export PROMETHEUS_PORT=9091
    export GRAFANA_PORT=3001
fi

# Build and push image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} -f docker/dockerfile .
echo "Pushing Docker image..."
docker push ${IMAGE_NAME}

# Stop existing containers
docker-compose down || true

# Start new containers
export DOCKER_IMAGE=${IMAGE_NAME}
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
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
