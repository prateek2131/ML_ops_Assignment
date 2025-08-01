import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_model_and_scaler():
    """Load the best model and scaler"""
    try:
        model = joblib.load('models/best_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        return None, None

def load_model_metadata():
    """Load model metadata"""
    try:
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        return None

def preprocess_input(input_data, scaler):
    """Preprocess input data for prediction"""
    try:
        # Convert to DataFrame if it's a dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data)
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        return scaled_data
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        return None

def validate_input_features(input_data):
    """Validate input features"""
    required_features = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    if isinstance(input_data, dict):
        missing_features = set(required_features) - set(input_data.keys())
        if missing_features:
            return False, f"Missing features: {missing_features}"
    elif isinstance(input_data, list):
        for item in input_data:
            missing_features = set(required_features) - set(item.keys())
            if missing_features:
                return False, f"Missing features: {missing_features}"
    
    return True, "All features present"

def create_prediction_log(input_data, prediction, model_name, timestamp=None):
    """Create a log entry for a prediction"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'model_name': model_name,
        'input_data': input_data,
        'prediction': prediction,
        'status': 'success'
    }
    
    return log_entry