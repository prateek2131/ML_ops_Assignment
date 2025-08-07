import pytest
from fastapi.testclient import TestClient
import json
import os
import sys
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api import app

# Create dummy model and scaler for testing
def setup_test_model():
    os.makedirs('models', exist_ok=True)
    
    # Create a simple model
    model = RandomForestRegressor(n_estimators=1)
    X = np.random.rand(100, 8)  # 8 features
    y = np.random.rand(100)
    model.fit(X, y)
    
    # Create a scaler
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Save model and scaler
    joblib.dump(model, 'models/best_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save metadata
    with open('models/model_metadata.json', 'w') as f:
        json.dump({
            'best_model': 'random_forest',
            'timestamp': '2025-08-07',
            'metrics': {'mse': 0.1}
        }, f)

# Setup test model before running tests
setup_test_model()

# Initialize the app before creating test client
from src.api import startup_event
import asyncio
asyncio.run(startup_event())

# Create test client
client = TestClient(app)

class TestAPI:
    @classmethod
    def setup_class(cls):
        """Setup for all tests"""
        # Ensure we're in the correct directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_predict_endpoint_valid_input(self):
        """Test prediction endpoint with valid input"""
        test_data = {
            "features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "model_used" in data
        assert "timestamp" in data
        assert isinstance(data["prediction"], float)
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction endpoint with invalid input"""
        test_data = {
            "features": {
                "MedInc": -1,  # Invalid negative value
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_features(self):
        """Test prediction endpoint with missing features"""
        test_data = {
            "features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0
                # Missing other required features
            }
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422
    
    def test_batch_predict_endpoint(self):
        """Test batch prediction endpoint"""
        test_data = {
            "features": [
                {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.984,
                    "AveBedrms": 1.024,
                    "Population": 322.0,
                    "AveOccup": 2.556,
                    "Latitude": 37.88,
                    "Longitude": -122.23
                },
                {
                    "MedInc": 7.2574,
                    "HouseAge": 21.0,
                    "AveRooms": 6.238,
                    "AveBedrms": 0.971,
                    "Population": 2401.0,
                    "AveOccup": 2.109,
                    "Latitude": 37.86,
                    "Longitude": -122.22
                }
            ]
        }
        
        response = client.post("/predict/batch", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_predictions" in data
        assert len(data["predictions"]) == 2
        assert data["total_predictions"] == 2
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
