#!/bin/bash

set -e

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
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!
echo "MLflow server started with PID: $MLFLOW_PID"

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
echo "To access MLflow UI, visit: http://localhost:5000"
