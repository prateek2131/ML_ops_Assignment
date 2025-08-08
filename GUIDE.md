# California Housing ML Project - Setup and Operation Guide

This guide provides step-by-step instructions for setting up and running the California Housing ML Project.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Services](#running-the-services)
- [Service Verification](#service-verification)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Ensure you have the following installed:
- Docker (20.10.x or higher)
- Docker Compose (2.x or higher)
- Git
- Python 3.9 or higher

## Project Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/prateek2131/ML_ops_Assignment.git
   cd ML_ops_Assignment
   ```

2. **Create Directory Structure**
   ```bash
   mkdir -p models logs data/processed mlartifacts
   mkdir -p monitoring/grafana/provisioning/{dashboards,datasources}
   ```

3. **Data Preparation**
   - Place `california_housing_raw.csv` in `data/raw/` directory
   - Ensure `data/processed` directory exists for processed data

## Environment Configuration

1. **Create Environment File**
   Create a `.env` file in the project root with:
   ```
   API_PORT=8000
   MLFLOW_PORT=5001
   PROMETHEUS_PORT=9090
   GRAFANA_PORT=3000
   LOG_LEVEL=INFO
   MLFLOW_TRACKING_URI=http://mlflow:5001
   ```

## Running the Services

1. **Clean Up Existing Containers**
   ```bash
   # Stop and remove existing containers
   docker-compose down -v

   # Clean up system resources
   docker system prune -f
   docker volume prune -f
   ```

2. **Start Services**
   ```bash
   # Build and start all services
   docker-compose up -d --build

   # Wait for services to initialize
   sleep 30
   ```

## Service Verification

### 1. API Service
- **URL**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs

### 2. MLflow
- **UI URL**: http://localhost:5001
- **Models View**: http://localhost:5001/#/models
- **Verification**: 
  - UI should be accessible at http://localhost:5001
  - Models should be visible at http://localhost:5001/#/models
  - Experiments should be visible in the UI

### 3. Prometheus
- **URL**: http://localhost:9090
- **Targets Check**: http://localhost:9090/targets
- **Expected Targets**:
  - api (UP)
  - mlflow (UP)
  - node-exporter (UP)
  - prometheus (UP)

### 4. Grafana
- **URL**: http://localhost:3000
- **Default Credentials**:
  - Username: admin
  - Password: admin
- **First Login**: You'll be prompted to change the password

## Monitoring and Maintenance

### View Service Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f mlflow
```

### Model Operations
1. **Check Model Status**
   - View MLflow UI: http://localhost:5001
   - Check models directory: `ls -l models/`

2. **Test Predictions**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"features": [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]}'
   ```

### Regular Maintenance
1. **Check Disk Usage**
   ```bash
   docker system df
   ```

2. **Clean Up Resources**
   ```bash
   docker system prune -f
   docker volume prune -f
   ```

3. **Backup Important Data**
   - MLflow data: `mlruns/` directory
   - Model files: `models/` directory
   - Processed data: `data/processed/` directory

## Troubleshooting

### Common Issues and Solutions

1. **Services Won't Start**
   ```bash
   # Check logs
   docker-compose logs -f

   # Restart specific service
   docker-compose restart <service_name>
   ```

2. **Port Conflicts**
   - Check for processes using required ports:
     ```bash
     sudo lsof -i :<port_number>
     ```
   - Edit `.env` file to use different ports

3. **MLflow/Prometheus Target Down**
   ```bash
   # Restart services
   docker-compose restart mlflow prometheus
   ```

4. **Missing Model Files**
   ```bash
   # Check model files
   ls -l models/
   # Should see: best_model.joblib, scaler.joblib, model_metadata.json
   ```

### Shutdown Procedure
```bash
# Normal shutdown
docker-compose down

# Complete cleanup (including volumes)
docker-compose down -v
```

## Support

If you encounter any issues not covered in this guide, please:
1. Check the service logs
2. Verify all prerequisites are met
3. Ensure all required files are in place
4. Create an issue in the repository with:
   - Detailed error description
   - Relevant logs
   - Steps to reproduce

---

Remember to always check the logs if something isn't working as expected. The logs are your best source for troubleshooting information.
