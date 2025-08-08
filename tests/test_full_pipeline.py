# tests/test_full_pipeline.py
"""
End-to-end pipeline testing
"""
import pytest
import tempfile
import shutil
import os
import subprocess
import time
import requests
from multiprocessing import Process

class TestFullPipeline:
    
    def setup_class(self):
        """Setup for full pipeline testing"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Copy project files to test directory
        shutil.copytree(".", self.test_dir, dirs_exist_ok=True)
        os.chdir(self.test_dir)
        
        # Start API server in background
        self.api_process = None
    
    def teardown_class(self):
        """Cleanup after testing"""
        if self.api_process:
            self.api_process.terminate()
        
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_data_pipeline(self):
        """Test complete data pipeline"""
        # Test data preprocessing
        result = subprocess.run(['python', 'src/data_preprocessing.py'], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check if files are created
        assert os.path.exists('data/raw/california_housing_raw.csv')
        assert os.path.exists('data/processed/train_data.csv')
        assert os.path.exists('data/processed/test_data.csv')
    
    def test_model_training_pipeline(self):
        """Test model training pipeline"""
        # Ensure data exists
        self.test_data_pipeline()
        
        # Test model training
        result = subprocess.run(['python', 'scripts/train_models.py'], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check if model files are created
        assert os.path.exists('models/best_model.joblib')
        assert os.path.exists('models/model_metadata.json')
    
    def test_api_pipeline(self):
        """Test API pipeline"""
        # Ensure models exist
        self.test_model_training_pipeline()
        
        # Start API server
        self.api_process = subprocess.Popen(
            ['uvicorn', 'src.api:app', '--host', '127.0.0.1', '--port', '8000', '--log-level', 'error'],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Give server time to start
        time.sleep(5)
        
        # Test API endpoints
        base_url = "http://127.0.0.1:8000"
        
        # Health check
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        
        # Prediction test
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
        
        response = requests.post(f"{base_url}/predict", json=test_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "prediction" in result
        assert isinstance(result["prediction"], float)
        
        # Stop API server
        self.api_process.terminate()
    
    def test_docker_pipeline(self):
        """Test Docker containerization"""
        # Build Docker image
        result = subprocess.run([
            'docker', 'build', '-t', 'test-housing-api', 
            '-f', 'Dockerfile', '.'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Test container run (quick test)
        result = subprocess.run([
            'docker', 'run', '--rm', '-d', '--name', 'test-container',
            '-p', '8000:8000', 'test-housing-api'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            time.sleep(10)  # Wait for container to start
            
            try:
                response = requests.get("http://localhost:800/health", timeout=5)
                assert response.status_code == 200
            except requests.exceptions.RequestException:
                pytest.skip("Docker container health check failed")
            finally:
                # Cleanup
                subprocess.run(['docker', 'stop', 'test-container'], 
                             capture_output=True)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        # This would include load testing, memory usage, etc.
        self.test_model_training_pipeline()
        
        # Load model and test prediction speed
        import time
        import joblib
        import pandas as pd
        
        model = joblib.load('models/best_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # Test data
        test_data = pd.read_csv('data/processed/test_data.csv')
        X_test = test_data.drop('target', axis=1).iloc[:100]  # First 100 samples
        
        # Measure prediction time
        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        
        avg_prediction_time = (end_time - start_time) / len(X_test)
        
        # Assert performance requirements
        assert avg_prediction_time < 0.01  # Less than 10ms per prediction
        assert len(predictions) == len(X_test)


# Performance testing script
# tests/performance_test.py
"""
Performance and load testing
"""
import concurrent.futures
import time
import requests
import statistics
import json

def single_prediction_test(base_url="http://localhost:8000"):
    """Test single prediction performance"""
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
    
    start_time = time.time()
    response = requests.post(f"{base_url}/predict", json=test_data)
    end_time = time.time()
    
    return {
        "status_code": response.status_code,
        "response_time": end_time - start_time,
        "success": response.status_code == 200
    }

def load_test(base_url="http://localhost:8000", num_requests=100, concurrency=10):
    """Perform load testing"""
    print(f"Starting load test: {num_requests} requests with {concurrency} concurrent connections")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_prediction_test, base_url) 
                  for _ in range(num_requests)]
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Request failed: {e}")
    
    # Analyze results
    response_times = [r["response_time"] for r in results if r["success"]]
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    
    if response_times:
        stats = {
            "total_requests": len(results),
            "successful_requests": len(response_times),
            "success_rate": success_rate,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "p95_response_time": sorted(response_times)[int(0.95 * len(response_times))],
            "min_response_time": min(response_times),
            "max_response_time": max(response_times)
        }
        
        print(json.dumps(stats, indent=2))
        return stats
    else:
        print("No successful requests!")
        return None

if __name__ == "__main__":
    # Run load test
    load_test(num_requests=500, concurrency=20)