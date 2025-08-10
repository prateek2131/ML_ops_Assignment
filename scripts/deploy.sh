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

# Set base directory (assuming script is in scripts/ subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log_info "Script directory: ${SCRIPT_DIR}"
log_info "Base directory: ${BASE_DIR}"

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
    if [ -f "${BASE_DIR}/Dockerfile" ]; then
        echo "Dockerfile"
    elif [ -f "${BASE_DIR}/dockerfile" ]; then
        echo "dockerfile"
    else
        log_error "No Dockerfile found in ${BASE_DIR} (checked both 'Dockerfile' and 'dockerfile')"
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
rm -rf ${DEPLOY_DIR}  # Clean up any previous deployment
mkdir -p ${DEPLOY_DIR}

# Load environment variables from .env.production
if [ -f "${BASE_DIR}/.env.production" ]; then
    log_info "Loading production environment variables from ${BASE_DIR}/.env.production"
    set -a
    source "${BASE_DIR}/.env.production"
    set +a
else
    log_warn ".env.production not found in ${BASE_DIR}, using default values..."
    export API_PORT=8000
    export MLFLOW_PORT=5001
    export PROMETHEUS_PORT=9090
    export GRAFANA_PORT=3000
    export LOG_LEVEL=INFO
fi

# Verify required files exist before copying
log_info "Verifying required files exist in ${BASE_DIR}..."

REQUIRED_FILES=(
    "docker-compose.${ENVIRONMENT}.yml"
    "requirements.txt"
    "src/"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "${BASE_DIR}/${file}" ]; then
        MISSING_FILES+=("${file}")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    log_error "Missing required files/directories in ${BASE_DIR}:"
    for file in "${MISSING_FILES[@]}"; do
        log_error " - ${file}"
    done
    log_error "Directory contents of ${BASE_DIR}:"
    ls -la "${BASE_DIR}/"
    exit 1
fi

# Check for models directory and files
if [ ! -d "${BASE_DIR}/models" ]; then
    log_error "models/ directory not found in ${BASE_DIR}"
    log_error "You need to train your models first!"
    log_error "Run your training script to generate the required model files."
    exit 1
fi

log_info "Models directory found. Checking for required model files..."
log_info "Files in ${BASE_DIR}/models/:"
ls -la "${BASE_DIR}/models/"

# Check for specific model files that utils.py expects
MODEL_FILES=(
    "models/best_model.joblib"
    "models/scaler.joblib"
    "models/model_metadata.json"
)

MISSING_MODELS=()
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "${BASE_DIR}/${file}" ]; then
        MISSING_MODELS+=("${file}")
    fi
done

if [ ${#MISSING_MODELS[@]} -ne 0 ]; then
    log_error "========================================="
    log_error "CRITICAL: Missing model files!"
    log_error "========================================="
    log_error "The following required model files are missing:"
    for file in "${MISSING_MODELS[@]}"; do
        log_error " ❌ ${file}"
    done
    log_error ""
    log_error "These files are required by your API (src/utils.py expects them)."
    log_error ""
    log_error "SOLUTION:"
    log_error "1. Run your model training script first:"
    log_error "   python your_training_script.py"
    log_error ""
    log_error "2. Or use the updated training script that generates all required files"
    log_error ""
    log_error "The deployment will fail without these files!"
    exit 1
else
    log_info "✅ All required model files found"
    for file in "${MODEL_FILES[@]}"; do
        SIZE=$(du -h "${BASE_DIR}/${file}" | cut -f1)
        log_info "  ✅ ${file} (${SIZE})"
    done
fi

# Check if utils.py has required functions
log_info "Validating ${BASE_DIR}/src/utils.py..."
if [ ! -f "${BASE_DIR}/src/utils.py" ]; then
    log_error "${BASE_DIR}/src/utils.py not found!"
    exit 1
fi

# Check if utils.py contains required functions
REQUIRED_FUNCTIONS=("load_model_and_scaler" "load_model_metadata" "preprocess_input" "validate_input_features" "create_prediction_log")
MISSING_FUNCTIONS=()

for func in "${REQUIRED_FUNCTIONS[@]}"; do
    if ! grep -q "def $func" "${BASE_DIR}/src/utils.py"; then
        MISSING_FUNCTIONS+=("${func}")
    fi
done

if [ ${#MISSING_FUNCTIONS[@]} -ne 0 ]; then
    log_error "Required functions missing in src/utils.py:"
    for func in "${MISSING_FUNCTIONS[@]}"; do
        log_error " - ${func}"
    done
    exit 1
fi

log_info "✅ All required functions found in utils.py"

# Create optimized Dockerfile for deployment
log_info "Creating optimized Dockerfile for deployment..."
cat > ${DEPLOY_DIR}/Dockerfile << 'EOF'
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with correct structure
COPY src/ ./src/
COPY models/ ./models/

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chmod 755 /app/logs

# Create non-root user
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

log_info "Created optimized Dockerfile"

# Copy files to deployment directory
log_info "Copying deployment files to ${DEPLOY_DIR}..."

# Copy main files
cp "${BASE_DIR}/docker-compose.${ENVIRONMENT}.yml" "${DEPLOY_DIR}/docker-compose.yml"
cp "${BASE_DIR}/requirements.txt" "${DEPLOY_DIR}/"

# Copy directories
cp -r "${BASE_DIR}/src" "${DEPLOY_DIR}/"
cp -r "${BASE_DIR}/models" "${DEPLOY_DIR}/"

# Copy optional directories/files
if [ -d "${BASE_DIR}/monitoring" ]; then
    cp -r "${BASE_DIR}/monitoring" "${DEPLOY_DIR}/"
    log_info "✅ Copied monitoring/ directory"
else
    log_warn "monitoring/ directory not found, skipping..."
fi

if [ -d "${BASE_DIR}/logs" ]; then
    cp -r "${BASE_DIR}/logs" "${DEPLOY_DIR}/"
    log_info "✅ Copied logs/ directory"
else
    mkdir -p "${DEPLOY_DIR}/logs"
    log_info "✅ Created logs/ directory"
fi

if [ -f "${BASE_DIR}/.env.production" ]; then
    cp "${BASE_DIR}/.env.production" "${DEPLOY_DIR}/"
    log_info "✅ Copied .env.production"
fi

# Verify files were copied correctly
log_info "Verifying copied files in deployment directory..."
DEPLOY_VERIFICATION=(
    "src/api.py"
    "src/utils.py"
    "models/best_model.joblib"
    "models/scaler.joblib"
    "models/model_metadata.json"
    "requirements.txt"
    "docker-compose.yml"
    "Dockerfile"
)

ALL_COPIED=true
for file in "${DEPLOY_VERIFICATION[@]}"; do
    if [ -f "${DEPLOY_DIR}/${file}" ]; then
        SIZE=$(du -h "${DEPLOY_DIR}/${file}" | cut -f1)
        log_info "✅ ${file} (${SIZE})"
    else
        log_error "❌ ${file} - NOT COPIED!"
        ALL_COPIED=false
    fi
done

if [ "$ALL_COPIED" = false ]; then
    log_error "File copying failed! Check the errors above."
    exit 1
fi

log_info "✅ All files copied successfully"

# Change to deployment directory
cd "${DEPLOY_DIR}"

# Build and push image
log_info "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

# Check if we should push the image
if [[ "${ENVIRONMENT}" == "production" ]]; then
    log_info "Pushing Docker image to registry..."
    docker push "${IMAGE_NAME}"
fi

# Stop existing containers
log_info "Stopping existing containers..."
${DOCKER_COMPOSE_CMD} down || true

# Start new containers
log_info "Starting new containers..."
export DOCKER_IMAGE="${IMAGE_NAME}"
${DOCKER_COMPOSE_CMD} up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 30

# Health check and remaining script continues as before...
log_info "Performing health checks..."

# Container status check
log_info "Container status:"
CONTAINER_STATUS=$(${DOCKER_COMPOSE_CMD} ps)
echo "$CONTAINER_STATUS"

# Check if API container is restarting or has issues
if echo "$CONTAINER_STATUS" | grep -q "Restarting\|Exited\|Exit"; then
    log_error "========================================="
    log_error "API CONTAINER CRASH DETECTED"
    log_error "========================================="
    
    # Get the API service name dynamically
    API_SERVICE=$(echo "$CONTAINER_STATUS" | grep "api" | awk '{print $1}' | head -1)
    if [ -z "$API_SERVICE" ]; then
        API_SERVICE="api" # fallback
    fi
    
    log_error "Container logs from API (${API_SERVICE}):"
    ${DOCKER_COMPOSE_CMD} logs "${API_SERVICE}" || true
    exit 1
fi

# Health endpoint check
for i in {1..10}; do
    log_info "Health check attempt ${i}/10..."
    
    HEALTH_RESPONSE=$(curl -s -w "HTTP_CODE:%{http_code}" \
        --connect-timeout 5 \
        --max-time 30 \
        "http://localhost:${API_PORT}/health" 2>&1)
    
    HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep -o "HTTP_CODE:[0-9]*" | cut -d: -f2)
    RESPONSE_BODY=$(echo "$HEALTH_RESPONSE" | sed 's/HTTP_CODE:[0-9]*$//')
    
    if [ "$HTTP_CODE" = "200" ]; then
        log_info "✅ API is healthy!"
        log_info "Response: ${RESPONSE_BODY}"
        break
    else
        if [ ${i} -eq 10 ]; then
            log_error "❌ Health check failed after 10 attempts"
            log_error "HTTP Status: ${HTTP_CODE:-'Connection failed'}"
            log_error "Response: ${RESPONSE_BODY}"
            exit 1
        else
            log_warn "⏳ Waiting for API... (attempt ${i}/10)"
            sleep 10
        fi
    fi
done

log_info "🎉 Deployment to ${ENVIRONMENT} completed successfully!"
log_info "API is running at: http://localhost:${API_PORT}"
log_info "Health check: http://localhost:${API_PORT}/health"