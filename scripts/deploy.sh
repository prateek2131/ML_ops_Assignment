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
cp docker-compose.yml ${DEPLOY_DIR}/
cp -r monitoring ${DEPLOY_DIR}/

cd ${DEPLOY_DIR}

# Set environment-specific configurations
if [ "${ENVIRONMENT}" = "production" ]; then
    export API_PORT=8000
    export MLFLOW_PORT=5000
    export PROMETHEUS_PORT=9090
    export GRAFANA_PORT=3000
else
    export API_PORT=8001
    export MLFLOW_PORT=5001
    export PROMETHEUS_PORT=9091
    export GRAFANA_PORT=3001
fi

# Pull latest image
docker pull ${IMAGE_NAME}

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

# Run smoke tests
echo "Running smoke tests..."
python -c "
import requests
import sys

try:
    # Test health endpoint
    response = requests.get('http://localhost:${API_PORT}/health')
    assert response.status_code == 200
    
    # Test prediction endpoint
    test_data = {
        'features': {
            'MedInc': 8.3252,
            'HouseAge': 41.0,
            'AveRooms': 6.984,
            'AveBedrms': 1.024,
            'Population': 322.0,
            'AveOccup': 2.556,
            'Latitude': 37.88,
            'Longitude': -122.23
        }
    }
    
    response = requests.post('http://localhost:${API_PORT}/predict', json=test_data)
    assert response.status_code == 200
    
    print('Smoke tests passed!')
except Exception as e:
    print(f'Smoke tests failed: {e}')
    sys.exit(1)
"

echo "Deployment to ${ENVIRONMENT} completed successfully!"
