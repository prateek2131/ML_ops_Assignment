import requests
import time
import json
import os

API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

def test_api_integration():
    """Test complete API integration"""
    
    # Wait for API to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f'{API_BASE_URL}/health')
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        
        if i == max_retries - 1:
            raise Exception("API failed to start")
        
        time.sleep(2)
    
    print("✅ API is healthy")
    
    # Test prediction endpoint
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
    
    response = requests.post(f'{API_BASE_URL}/predict', json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert 'prediction' in result
    assert isinstance(result['prediction'], float)
    
    print("✅ Single prediction works")
    
    # Test batch prediction
    batch_data = {
        "features": [test_data["features"], test_data["features"]]
    }
    
    response = requests.post(f'{API_BASE_URL}/predict/batch', json=batch_data)
    assert response.status_code == 200
    
    result = response.json()
    assert 'predictions' in result
    assert len(result['predictions']) == 2
    
    print("✅ Batch prediction works")
    
    # Test model info
    response = requests.get(f'{API_BASE_URL}/model/info')
    assert response.status_code == 200
    
    print("✅ Model info endpoint works")
    
    # Test metrics endpoint
    response = requests.get(f'{API_BASE_URL}/metrics')
    assert response.status_code == 200
    
    print("✅ Metrics endpoint works")
    
    print("🎉 All integration tests passed!")

if __name__ == "__main__":
    test_api_integration()