from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime
import logging
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

from src.utils import (
    load_model_and_scaler, 
    load_model_metadata, 
    preprocess_input, 
    validate_input_features,
    create_prediction_log
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
ERROR_COUNT = Counter('api_errors_total', 'Total API errors', ['error_type'])

app = FastAPI(
    title="California Housing Price Prediction API",
    description="Machine Learning API for predicting California housing prices",
    version="1.0.0"
)

# Pydantic models for input validation
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group", ge=0, le=15)
    HouseAge: float = Field(..., description="Median house age in block group", ge=1, le=52)
    AveRooms: float = Field(..., description="Average number of rooms per household", ge=1, le=20)
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", ge=0, le=5)
    Population: float = Field(..., description="Block group population", ge=3, le=35682)
    AveOccup: float = Field(..., description="Average house occupancy", ge=0.5, le=50)
    Latitude: float = Field(..., description="Latitude", ge=32.5, le=42)
    Longitude: float = Field(..., description="Longitude", ge=-125, le=-114)
    
    @validator('AveBedrms')
    def validate_bedrooms(cls, v, values):
        if 'AveRooms' in values and v > values['AveRooms']:
            raise ValueError('Average bedrooms cannot exceed average rooms')
        return v

class PredictionRequest(BaseModel):
    features: HousingFeatures
    return_confidence: Optional[bool] = Field(False, description="Return prediction confidence interval")

class BatchPredictionRequest(BaseModel):
    features: List[HousingFeatures]
    return_confidence: Optional[bool] = Field(False, description="Return prediction confidence intervals")

class PredictionResponse(BaseModel):
    prediction: float
    confidence_interval: Optional[Dict[str, float]] = None
    model_used: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_predictions: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str

# Global variables for model and scaler
model = None
scaler = None
model_metadata = None

def init_database():
    """Initialize SQLite database for logging"""
    conn = sqlite3.connect('logs/predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction REAL NOT NULL,
            model_name TEXT NOT NULL,
            response_time REAL,
            status TEXT DEFAULT 'success'
        )
    ''')
    
    conn.commit()
    conn.close()

def log_prediction_to_db(input_data: dict, prediction: float, model_name: str, response_time: float):
    """Log prediction to SQLite database"""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (timestamp, input_data, prediction, model_name, response_time)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(input_data),
            prediction,
            model_name,
            response_time
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error logging to database: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global model, scaler, model_metadata
    
    logger.info("Starting up the API...")
    
    # Initialize database
    init_database()
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    model_metadata = load_model_metadata()
    
    if model is None or scaler is None:
        logger.error("Failed to load model or scaler")
        raise RuntimeError("Model or scaler not found")
    
    logger.info("API startup completed successfully")

@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all requests and measure response time"""
    start_time = time.time()
    
    # Increment request counter
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    
    response = await call_next(request)
    
    # Calculate response time
    process_time = time.time() - start_time
    REQUEST_DURATION.observe(process_time)
    
    # Log the request
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    
    return response

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "California Housing Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    start_time = time.time()
    
    try:
        if model is None or scaler is None:
            ERROR_COUNT.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert request to dict
        input_data = request.features.dict()
        
        # Validate input features
        is_valid, message = validate_input_features(input_data)
        if not is_valid:
            ERROR_COUNT.labels(error_type="invalid_input").inc()
            raise HTTPException(status_code=400, detail=message)
        
        # Preprocess input
        processed_input = preprocess_input(input_data, scaler)
        if processed_input is None:
            ERROR_COUNT.labels(error_type="preprocessing_error").inc()
            raise HTTPException(status_code=400, detail="Error preprocessing input")
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        
        # Calculate confidence interval if requested
        confidence_interval = None
        if request.return_confidence and hasattr(model, 'predict_proba'):
            # For regression, we can estimate confidence using prediction intervals
            # This is a simplified approach
            std_error = 0.1 * prediction  # Simplified error estimation
            confidence_interval = {
                "lower": float(prediction - 1.96 * std_error),
                "upper": float(prediction + 1.96 * std_error)
            }
        
        # Log prediction
        response_time = time.time() - start_time
        log_prediction_to_db(input_data, float(prediction), model_metadata['best_model'], response_time)
        
        # Increment prediction counter
        PREDICTION_COUNT.inc()
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence_interval=confidence_interval,
            model_used=model_metadata['best_model'] if model_metadata else "unknown",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type="internal_error").inc()
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    start_time = time.time()
    
    try:
        if model is None or scaler is None:
            ERROR_COUNT.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        
        for i, features in enumerate(request.features):
            input_data = features.dict()
            
            # Validate input features
            is_valid, message = validate_input_features(input_data)
            if not is_valid:
                ERROR_COUNT.labels(error_type="invalid_input").inc()
                raise HTTPException(status_code=400, detail=f"Invalid input at index {i}: {message}")
            
            # Preprocess input
            processed_input = preprocess_input(input_data, scaler)
            if processed_input is None:
                ERROR_COUNT.labels(error_type="preprocessing_error").inc()
                raise HTTPException(status_code=400, detail=f"Error preprocessing input at index {i}")
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            
            # Calculate confidence interval if requested
            confidence_interval = None
            if request.return_confidence:
                std_error = 0.1 * prediction
                confidence_interval = {
                    "lower": float(prediction - 1.96 * std_error),
                    "upper": float(prediction + 1.96 * std_error)
                }
            
            predictions.append(PredictionResponse(
                prediction=float(prediction),
                confidence_interval=confidence_interval,
                model_used=model_metadata['best_model'] if model_metadata else "unknown",
                timestamp=datetime.now().isoformat()
            ))
            
            # Log each prediction
            response_time = time.time() - start_time
            log_prediction_to_db(input_data, float(prediction), model_metadata['best_model'], response_time)
        
        # Increment prediction counter
        PREDICTION_COUNT.inc(len(predictions))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type="internal_error").inc()
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    
    return {
        "model_name": model_metadata.get('best_model', 'unknown'),
        "training_timestamp": model_metadata.get('timestamp', 'unknown'),
        "metrics": model_metadata.get('metrics', {}),
        "all_model_results": model_metadata.get('all_results', {})
    }

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get prediction history from database"""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, input_data, prediction, model_name, response_time, status
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "input_data": json.loads(row[1]),
                "prediction": row[2],
                "model_name": row[3],
                "response_time": row[4],
                "status": row[5]
            })
        
        return {"history": history, "total_records": len(history)}
        
    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        raise HTTPException(status_code=500, detail="Error fetching prediction history")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)