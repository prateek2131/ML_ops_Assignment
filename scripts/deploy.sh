#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

DOCKER_USERNAME=${DOCKER_USERNAME:-"prateek2131"}

log_info() {
echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect docker compose command
detect_docker_compose() {
if command -v docker-compose &> /dev/null; then
echo "docker-compose"
elif docker compose version &> /dev/null; then
echo "docker compose"
else
log_error "Neither docker-compose nor docker compose found"
exit 1
fi
}

# Function to find dockerfile
find_dockerfile() {
if [ -f "Dockerfile" ]; then
echo "Dockerfile"
elif [ -f "dockerfile" ]; then
echo "dockerfile"
else
log_error "No Dockerfile found (checked both 'Dockerfile' and 'dockerfile')"
exit 1
fi
}

# Verify required environment variables
if [ -z "$DOCKER_USERNAME" ]; then
log_error "DOCKER_USERNAME environment variable is not set"
exit 1
fi

ENVIRONMENT="production"
IMAGE_TAG=${1:-latest}
IMAGE_NAME="${DOCKER_USERNAME}/california-housing-api:${IMAGE_TAG}"

# Detect docker compose command
DOCKER_COMPOSE_CMD=$(detect_docker_compose)
log_info "Using Docker Compose command: ${DOCKER_COMPOSE_CMD}"

# Find dockerfile
DOCKERFILE_NAME=$(find_dockerfile)
log_info "Found Dockerfile: ${DOCKERFILE_NAME}"

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

# Verify required files exist before copying
log_info "Verifying required files exist..."
REQUIRED_FILES=(
    "docker-compose.${ENVIRONMENT}.yml"
    "requirements.txt"
    "src/"
    "models/"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    log_error "Missing required files/directories:"
    for file in "${MISSING_FILES[@]}"; do
        log_error "  - $file"
    done
    exit 1
fi

# Check for specific model files that utils.py expects
log_info "Checking for required model files..."
MODEL_FILES=(
    "models/best_model.joblib"
    "models/scaler.joblib"
    "models/model_metadata.json"
)

MISSING_MODELS=()
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_MODELS+=("$file")
    fi
done

if [ ${#MISSING_MODELS[@]} -ne 0 ]; then
    log_warn "Missing model files (will cause API startup failure):"
    for file in "${MISSING_MODELS[@]}"; do
        log_warn "  - $file"
    done
    log_warn "The API will crash without these files!"
fi

# Check if utils.py has required functions
log_info "Validating src/utils.py..."
if [ ! -f "src/utils.py" ]; then
    log_error "src/utils.py not found!"
    exit 1
fi

# Check if utils.py contains required functions
REQUIRED_FUNCTIONS=("load_model_and_scaler" "load_model_metadata" "preprocess_input" "validate_input_features" "create_prediction_log")
for func in "${REQUIRED_FUNCTIONS[@]}"; do
    if ! grep -q "def $func" src/utils.py; then
        log_error "Required function '$func' not found in src/utils.py"
        exit 1
    fi
done

log_info "All required functions found in utils.py"

# Now proceed with copying
cp docker-compose.${ENVIRONMENT}.yml ${DEPLOY_DIR}/docker-compose.yml
cp ${DOCKERFILE_NAME} ${DEPLOY_DIR}/Dockerfile
cp -r monitoring ${DEPLOY_DIR}/
cp -r src ${DEPLOY_DIR}/
cp -r models ${DEPLOY_DIR}/
cp -r logs ${DEPLOY_DIR}/
cp requirements.txt ${DEPLOY_DIR}/
cp .env.production ${DEPLOY_DIR}/ 2>/dev/null || :

cd ${DEPLOY_DIR}

# Build and push image
log_info "Building Docker image: ${IMAGE_NAME}"
docker build -t ${IMAGE_NAME} .

# Check if we should push the image
if [[ "${ENVIRONMENT}" == "production" ]]; then
    log_info "Pushing Docker image to registry..."
    docker push ${IMAGE_NAME}
fi

# Stop existing containers
log_info "Stopping existing containers..."
${DOCKER_COMPOSE_CMD} down || true

# Start new containers
log_info "Starting new containers..."
export DOCKER_IMAGE=${IMAGE_NAME}
${DOCKER_COMPOSE_CMD} up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 30

# Health check
log_info "Performing health checks..."
log_info "Checking if port ${API_PORT} is available..."

# Check if port is in use
if netstat -ln 2>/dev/null | grep ":${API_PORT} " > /dev/null; then
log_info "Port ${API_PORT} is in use"
else
log_warn "Port ${API_PORT} does not appear to be in use"
fi

# Check container status before health checks
log_info "Container status before health checks:"
CONTAINER_STATUS=$(${DOCKER_COMPOSE_CMD} ps)
echo "$CONTAINER_STATUS"

# Check if API container is restarting or has issues
if echo "$CONTAINER_STATUS" | grep -q "Restarting\|Exited\|Exit"; then
log_error "========================================="
log_error "API CONTAINER CRASH DETECTED"
log_error "========================================="
log_error "The API container is crashing on startup."
echo ""

# Get the API service name dynamically
API_SERVICE=$(echo "$CONTAINER_STATUS" | grep "api" | awk '{print $1}' | head -1)
if [ -z "$API_SERVICE" ]; then
API_SERVICE="api"  # fallback
fi

log_error "Full container logs from API (${API_SERVICE}):"
${DOCKER_COMPOSE_CMD} logs ${API_SERVICE} || true
echo ""

log_error "Checking for specific issues in your API setup..."

# Check if model files exist in container
log_error "Checking model files in container:"
docker exec "${API_SERVICE}" ls -la /app/models/ 2>/dev/null || {
    log_error "Cannot access /app/models/ directory in container"
    log_error "This is likely why your API is crashing - model files not found"
}

# Check specific model files your utils.py expects
docker exec "${API_SERVICE}" test -f /app/models/best_model.joblib 2>/dev/null || log_error "❌ /app/models/best_model.joblib NOT FOUND"
docker exec "${API_SERVICE}" test -f /app/models/scaler.joblib 2>/dev/null || log_error "❌ /app/models/scaler.joblib NOT FOUND"
docker exec "${API_SERVICE}" test -f /app/models/model_metadata.json 2>/dev/null || log_error "❌ /app/models/model_metadata.json NOT FOUND"

# Check if utils module can be imported
log_error "Testing Python imports in container:"
docker exec "${API_SERVICE}" python3 -c "
try:
    from src.utils import load_model_and_scaler, load_model_metadata, preprocess_input
    print('✅ src.utils imports successful')
    
    # Test model loading
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        print('❌ Model/scaler loading failed - files not found or corrupted')
        exit(1)
    else:
        print('✅ Model and scaler loaded successfully')
        
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Model loading error: {e}')
    exit(1)
" 2>/dev/null || log_error "❌ Python import/model loading test failed"

# Check FastAPI startup
log_error "Testing FastAPI startup:"
docker exec "${API_SERVICE}" python3 -c "
try:
    from src.api import app
    print('✅ FastAPI app import successful')
except Exception as e:
    print(f'❌ FastAPI app import failed: {e}')
    exit(1)
" 2>/dev/null || log_error "❌ FastAPI startup test failed"

log_error "========================================="
log_error "DIAGNOSIS BASED ON YOUR UTILS.PY:"
log_error "Your utils.py expects these files:"
log_error "  - models/best_model.joblib"
log_error "  - models/scaler.joblib" 
log_error "  - models/model_metadata.json"
log_error ""
log_error "SOLUTIONS:"
log_error "1. Train your models and save them as .joblib files"
log_error "2. OR create dummy models for testing (see error details above)"
log_error "3. Check that models/ directory is properly copied to container"
log_error "4. Verify file permissions on model files"
log_error "========================================="
exit 1
fi

for i in {1..10}; do
log_info "Health check attempt ${i}/10..."

# More detailed curl with connection info
HEALTH_RESPONSE=$(curl -s -w "HTTP_CODE:%{http_code}|TIME_TOTAL:%{time_total}|TIME_CONNECT:%{time_connect}" \
  --connect-timeout 5 \
  --max-time 30 \
  http://localhost:${API_PORT}/health 2>&1)

HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
TIME_TOTAL=$(echo "$HEALTH_RESPONSE" | grep -o "TIME_TOTAL:[0-9.]*" | cut -d: -f2)
TIME_CONNECT=$(echo "$HEALTH_RESPONSE" | grep -o "TIME_CONNECT:[0-9.]*" | cut -d: -f2)
RESPONSE_BODY=$(echo "$HEALTH_RESPONSE" | sed 's/HTTP_CODE:[0-9]*|TIME_TOTAL:[0-9.]*|TIME_CONNECT:[0-9.]*$//')

if [ "$HTTP_CODE" = "200" ]; then
log_info "API is healthy!"
log_info "  Response time: ${TIME_TOTAL}s"
log_info "  Response: ${RESPONSE_BODY}"
break
else
if [ ${i} -eq 10 ]; then
log_error "========================================="
log_error "API HEALTH CHECK FAILED - DEBUGGING INFO"
log_error "========================================="
log_error "Final health check details:"
log_error "  HTTP Status Code: ${HTTP_CODE:-'Connection failed'}"
log_error "  Connection Time: ${TIME_CONNECT:-'N/A'}s"
log_error "  Total Time: ${TIME_TOTAL:-'N/A'}s"
log_error "  Response Body: ${RESPONSE_BODY}"
log_error "  URL: http://localhost:${API_PORT}/health"
echo ""

# Check if curl had connection issues
if echo "$HEALTH_RESPONSE" | grep -q "Connection refused"; then
log_error "DIAGNOSIS: Connection refused - service not listening on port ${API_PORT}"
elif echo "$HEALTH_RESPONSE" | grep -q "timeout"; then
log_error "DIAGNOSIS: Request timeout - service may be overloaded or hung"
elif [ -z "$HTTP_CODE" ]; then
log_error "DIAGNOSIS: No HTTP response - check if service is running"
elif [ "$HTTP_CODE" != "200" ]; then
log_error "DIAGNOSIS: Service responding but returning error status ${HTTP_CODE}"
fi
echo ""

# Container status debugging
log_error "Container status:"
${DOCKER_COMPOSE_CMD} ps || true
echo ""

# Check container logs for all services
log_error "Recent container logs:"
for service in $(${DOCKER_COMPOSE_CMD} config --services 2>/dev/null || echo "api"); do
log_error "--- Logs for service: ${service} ---"
${DOCKER_COMPOSE_CMD} logs --tail=10 ${service} 2>/dev/null || true
echo ""
done

# Network and port debugging
log_error "Network debugging:"
log_error "Active connections on port ${API_PORT}:"
netstat -ln 2>/dev/null | grep ":${API_PORT} " || log_error "No connections found on port ${API_PORT}"
log_error "Docker networks:"
docker network ls || true

# Resource usage
log_error "System resources:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || true

exit 1
else
log_warn "Waiting for API to be ready... (attempt ${i}/10)"
log_warn "  HTTP Status: ${HTTP_CODE:-'Connection failed'}"
log_warn "  Connection Time: ${TIME_CONNECT:-'N/A'}s"
if [ -n "$RESPONSE_BODY" ] && [ ${#RESPONSE_BODY} -lt 200 ]; then
log_warn "  Response: ${RESPONSE_BODY}"
fi
fi
sleep 10
fi
done

# Install Python and pip if not already installed (for CI/CD environments)
if ! command -v python3 &> /dev/null; then
log_info "Installing Python..."
if [[ "$OSTYPE" == "darwin"* ]]; then
brew install python3
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
if command -v apt-get &> /dev/null; then
apt-get update && apt-get install -y python3 python3-pip
elif command -v yum &> /dev/null; then
yum install -y python3 python3-pip
elif command -v apk &> /dev/null; then
apk add --no-cache python3 py3-pip
fi
fi
fi

# Install required Python package
log_info "Installing Python dependencies..."
pip3 install requests 2>/dev/null || python3 -m pip install requests

# Run smoke tests
log_info "Running smoke tests..."

# Enhanced error handling for CI/CD environments
python3 -c "
import requests
import sys
import os
import time

API_PORT = os.environ.get('API_PORT', '8000')
print(f'Testing health endpoint: http://localhost:{API_PORT}/health')

try:
    response = requests.get(f'http://localhost:{API_PORT}/health', timeout=30)
    print(f'HTTP Status Code: {response.status_code}')
    print(f'Response Headers: {dict(response.headers)}')
    print(f'Response Body: {response.text}')
    
    if response.status_code == 200:
        print('Smoke tests passed!')
        sys.exit(0)
    else:
        print(f'Smoke tests failed! API returned status code {response.status_code}')
        sys.exit(1)
        
except requests.exceptions.ConnectionError as e:
    print(f'Smoke tests failed! Connection error: {e}')
    print(f'Could not connect to http://localhost:{API_PORT}/health')
    print('This usually means:')
    print('  1. The API container is not running')
    print('  2. The API is not listening on the expected port')
    print('  3. There are network connectivity issues')
    sys.exit(1)
    
except requests.exceptions.Timeout as e:
    print(f'Smoke tests failed! Timeout error: {e}')
    print(f'Request to http://localhost:{API_PORT}/health timed out after 30 seconds')
    print('This usually means:')
    print('  1. The API is overloaded or hung')
    print('  2. The health endpoint is not responding')
    print('  3. There are performance issues')
    sys.exit(1)
    
except Exception as e:
    print(f'Smoke tests failed! Unexpected error: {e}')
    print(f'Error type: {type(e).__name__}')
    sys.exit(1)
"

# If we reach here, something unexpected happened
log_error "Smoke tests completed but script continued unexpectedly"

log_info "Deployment to ${ENVIRONMENT} completed successfully!"