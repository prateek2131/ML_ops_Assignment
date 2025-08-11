# California Housing MLOps Project

This project implements an end-to-end MLOps pipeline for the California Housing dataset, including model training, monitoring, and deployment in a production environment.
## Developers and Contributions

| Enrollment No. | Name                | Contribution (%) |
|----------------|---------------------|------------------|
| 2023ac05351    | Prateek Sanghi       | 100%              |
| 2023ac05525    | Divanshu Verma       | 100%              |
| 2023ac05300    | Jasman Arora         | 100%              |
| 2023ac05306    | Shaik Ahmed Nashith  | 100%              |

## Project Structure

```
ML_ops_Assignment/
├── data/                    # Data directory
│   ├── processed/          # Processed training and test data
│   └── raw/               # Raw data files
├── docker/                 # Docker configuration files
├── logs/                  # Application logs
├── mlartifacts/           # MLflow artifacts
├── mlruns/                # MLflow run data
├── models/                # Saved models
├── monitoring/            # Monitoring configuration
├── scripts/               # Utility scripts
├── src/                   # Source code
└── tests/                 # Test files
```

## Prerequisites

1. Install required software:
   - Python 3.9+
   - Docker and Docker Compose
   - Git

2. System requirements:
   - Memory: 8GB RAM (minimum)
   - Storage: 10GB free space
   - Processor: 2+ cores recommended

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/prateek2131/ML_ops_Assignment.git
   cd ML_ops_Assignment
   ```

2. **Create Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Unix/macOS
   # or
   .\venv\Scripts\activate   # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Data**
   ```bash
   python src/data_preprocessing.py
   ```

## Development Workflow

1. **Data Preprocessing**
   - Raw data is stored in `data/raw/`
   - Run preprocessing:
     ```bash
     python src/data_preprocessing.py
     ```
   - Processed data will be saved in `data/processed/`

2. **Model Training**
   ```bash
   python scripts/train_models.py
   ```
   - Models are tracked using MLflow
   - Best model is saved in `models/best_model.joblib`

3. **Testing**
   ```bash
   pytest tests/
   ```
   - Runs unit tests and integration tests
   - Validates model performance
   - Checks API functionality

## Deployment

### Local Development

1. **Start Services Locally**
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

2. **Access Services**
   - API: http://localhost:8000
   - MLflow: http://localhost:5001
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

### Production Environment

1. **Deploy to Production**
   ```bash
   export DOCKER_USERNAME=your_username
   ./scripts/deploy.sh production v1.0
   ```

   Production Environment Ports:
   - API: 8000
   - MLflow: 5001
   - Prometheus: 9090
   - Grafana: 3000

## Monitoring and Maintenance

1. **Access Monitoring Dashboards**
   - Grafana Dashboard: http://localhost:3000
     - Default credentials: admin/admin
     - Includes model performance metrics
     - System resource monitoring

2. **View MLflow Experiments**
   - MLflow UI: http://localhost:5001
   - Track experiments and model versions
   - Compare model performance

3. **Check Application Logs**
   - API logs: `logs/api.log`
   - Predictions database: `logs/predictions.db`

4. **Monitor Model Drift**
   ```bash
   python scripts/check_model_drift.py
   ```

## API Endpoints

1. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Make Predictions**
   ```bash
   curl -X POST http://localhost:8000/predict      -H "Content-Type: application/json"      -d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984127, "AveBedrms": 1.023810, "Population": 322.0, "AveOccup": 2.555556, "Latitude": 37.88, "Longitude": -122.23}'
   ```

## Troubleshooting

1. **Port Conflicts**
   - Check for existing processes using ports:
     ```bash
     lsof -i :PORT_NUMBER
     ```
   - Modify port numbers in docker-compose files if needed

2. **Container Issues**
   - View container logs:
     ```bash
     docker-compose logs -f [service_name]
     ```
   - Restart services:
     ```bash
     docker-compose restart [service_name]
     ```

3. **Model Performance Issues**
   - Check MLflow logs for model metrics
   - Review monitoring dashboards in Grafana
   - Analyze prediction logs in `logs/predictions.db`

## Contributing

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push changes and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Security Considerations

1. **Environment Variables**
   - Never commit sensitive credentials
   - Use `.env` files for local development
   - Use secure secrets management in production

2. **Access Control**
   - Restrict access to monitoring dashboards
   - Use strong passwords for Grafana
   - Implement API authentication as needed

3. **Data Protection**
   - Validate input data
   - Sanitize API responses
   - Monitor for unusual prediction patterns

## Automated Tasks

1. **Auto-training**
   ```bash
   python scripts/auto_train.py
   ```
   - Automatically retrains model on new data
   - Updates production model if performance improves

2. **Validation**
   ```bash
   python scripts/validate_model.py
   ```
   - Validates model performance
   - Checks for data drift
   - Generates validation reports

## Backup and Recovery

1. **Backup Data**
   - MLflow artifacts: `mlartifacts/`
   - Model files: `models/`
   - Logs: `logs/`
   - Monitoring data: `monitoring/`

2. **Recovery Steps**
   - Restore from backups
   - Rebuild Docker images
   - Redeploy services
   - Validate system functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.
