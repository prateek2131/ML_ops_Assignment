#!/bin/bash

set -e

# Check for Docker installation
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install Docker Desktop for Mac from: https://www.docker.com/products/docker-desktop/"
    echo "After installation, start Docker Desktop and run this script again"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "Setting up California Housing MLOps project..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed} models logs

# Initialize MLflow
# Try ports 5001-5010 until we find an available one
for port in {5001..5010}; do
    if ! lsof -i ":$port" > /dev/null 2>&1; then
        echo "Starting MLflow server on port $port"
        mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port $port &
        MLFLOW_PID=$!
        MLFLOW_PORT=$port
        echo "MLflow server started with PID: $MLFLOW_PID on port $MLFLOW_PORT"
        break
    fi
done

if [ -z "$MLFLOW_PID" ]; then
    echo "Error: Could not find an available port between 5001-5010"
    exit 1
fi

# Wait for MLflow to start
sleep 5

# Generate data and train models
echo "Generating data and training models..."
python3 src/data_preprocessing.py
python3 scripts/train_models.py

# Build Docker image
echo "Building Docker image..."
docker build -t california-housing-api -f docker/Dockerfile .

echo "Setup completed successfully!"
echo "To start the API server, run: uvicorn src.api:app --reload"
echo "To access MLflow UI, visit: http://localhost:$MLFLOW_PORT"
