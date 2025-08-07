#!/bin/bash

set -e  # Exit on any error

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

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.9 or higher"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose"
        exit 1
    fi
}


setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_info "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install additional test dependencies
    log_info "Installing test dependencies..."
    pip install "fastapi==0.68.2" \
                "starlette==0.14.2" \
                "httpx>=0.23.0" \
                "pytest==7.4.0" \
                "pytest-asyncio==0.23.8" \
                "pytest-cov" \
                "python-multipart==0.0.19"

    # Ensure model directory exists
    mkdir -p models
    
    # Check if model exists, if not create it
    if [ ! -f "models/best_model.joblib" ]; then
        log_info "Model not found, training new model..."
        python scripts/train_models.py
    fi
}

preprocess_data() {
    log_info "Preprocessing data..."
    python src/data_preprocessing.py
}

train_model() {
    log_info "Training model..."
    python scripts/train_models.py
}

run_tests() {
    log_info "Running tests..."
    pytest tests/
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create necessary directories
    mkdir -p monitoring/grafana/provisioning/dashboards
    mkdir -p monitoring/grafana/provisioning/datasources
    
    # Copy dashboard if it exists
    if [ -f "monitoring/grafana_dashboard.json" ]; then
        cp monitoring/grafana_dashboard.json monitoring/grafana/provisioning/dashboards/
    fi
}

deploy_services() {
    local env=$1
    local version=$2
    
    if [ -z "$DOCKER_USERNAME" ]; then
        log_error "DOCKER_USERNAME environment variable is not set"
        exit 1
    fi
    
    log_info "Deploying to $env environment..."
    
    # Load environment variables based on environment
    if [ "$env" == "production" ]; then
        set -a
        source <(grep -v '^#' docker/.env.production)
        set +a
    else
        set -a
        source <(grep -v '^#' docker/.env.staging)
        set +a
    fi
    
    # Run the deployment script
    ./scripts/deploy.sh $env $version
    
    # Wait for services to be ready
    sleep 30
    
    # Check if services are running
    log_info "Checking service health..."
    curl -f http://localhost:8000/health || {
        log_error "API service is not healthy"
        exit 1
    }
}

setup_auto_training() {
    log_info "Setting up automated training..."
    
    # Set up cron job for auto-training if it doesn't exist
    if ! crontab -l | grep -q "auto_train.py"; then
        (crontab -l 2>/dev/null; echo "0 0 * * * cd $(pwd) && ./venv/bin/python scripts/auto_train.py") | crontab -
        log_info "Added auto-training cron job"
    fi
}

main() {
    # Ensure we're in the project root directory
    cd "$(dirname "$0")/.."
    
    log_info "Starting setup process..."
    
    # Check requirements first
    check_requirements
    
    # Setup Python environment
    setup_python_env
    
    # Preprocess data
    preprocess_data
    
    # Train initial model
    train_model
    
    # Ensure model directory exists and has required files
    log_info "Checking model files..."
    if [ ! -f "models/best_model.joblib" ] || [ ! -f "models/scaler.joblib" ] || [ ! -f "models/model_metadata.json" ]; then
        log_info "Model files missing, training new model..."
        python scripts/train_models.py
        sleep 2  # Wait for files to be saved
    fi
    
    # Run tests with increased verbosity and proper environment
    log_info "Running tests..."
    PYTHONPATH=. pytest -v tests/ --log-cli-level=INFO
    
    # Setup monitoring
    setup_monitoring
    
    # Export Docker username for deployment
    read -p "Enter your Docker username: " docker_username
    export DOCKER_USERNAME=$docker_username
    
    # Deploy to staging first
    log_info "Deploying to staging environment..."
    deploy_services staging v1.0
    
    # Ask for production deployment
    read -p "Do you want to deploy to production? (y/n) " deploy_prod
    if [[ $deploy_prod == "y" ]]; then
        log_info "Deploying to production environment..."
        deploy_services production v1.0
    fi
    
    # Setup auto-training
    setup_auto_training
    
    log_info "Setup completed successfully!"
    log_info "Services are running at:"
    if [ "$env" == "production" ]; then
        log_info "- API: http://localhost:9000"
        log_info "- MLflow: http://localhost:5003"
        log_info "- Prometheus: http://localhost:9091"
        log_info "- Grafana: http://localhost:3001 (admin/admin)"
    else
        log_info "- API: http://localhost:8000"
        log_info "- MLflow: http://localhost:5002"
        log_info "- Prometheus: http://localhost:9090"
        log_info "- Grafana: http://localhost:3000 (admin/admin)"
    fi
    
    log_warn "Don't forget to:"
    log_warn "1. Change the Grafana admin password"
    log_warn "2. Set up your backup strategy"
    log_warn "3. Review monitoring alerts in Grafana"
}

# Run main function
main
