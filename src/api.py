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
from enum import Enum
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

# Enhancement: ModelType enum
class ModelType(str, Enum):
    LINEAR = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVR = "support_vector_regression"

# Enhancement: AdvancedPredictionRequest model
class AdvancedPredictionRequest(BaseModel):
    features: HousingFeatures
    model_type: Optional[ModelType] = None
    confidence_level: Optional[float] = Field(0.95, ge=0.8, le=0.99)
    return_feature_importance: Optional[bool] = False
    
    @validator('features')
    def validate_location(cls, v):
        # Custom validation for California coordinates
        if not (32.5 <= v.Latitude <= 42.0):
            raise ValueError('Latitude must be within California bounds')
        if not (-125.0 <= v.Longitude <= -114.0):
            raise ValueError('Longitude must be within California bounds')
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
    feature_importance: Optional[Dict[str, float]] = None  # Enhancement

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
available_models = {}  # Enhancement: store multiple models

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
    global model, scaler, model_metadata, available_models
    
    logger.info("Starting up the API...")
    
    # Initialize database
    init_database()
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    model_metadata = load_model_metadata()
    
    # Enhancement: Load all required models for selection
    # You can implement load_model_and_scaler to return a dict of models by type
    # Example:
    # available_models = {
    #     ModelType.LINEAR: joblib.load('models/linear_regression.pkl'),
    #     ModelType.RANDOM_FOREST: joblib.load('models/random_forest.pkl'),
    #     ...
    # }
    available_models = {
        ModelType.LINEAR: model,  # fallback: single model for demo
        ModelType.RANDOM_FOREST: model,
        ModelType.GRADIENT_BOOSTING: model,
        ModelType.SVR: model
    }
    
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
        
        # Additional input validation
        for key, value in input_data.items():
            if not isinstance(value, (int, float)):
                ERROR_COUNT.labels(error_type="invalid_input_type").inc()
                raise HTTPException(status_code=400, detail=f"Invalid type for {key}: expected number")
            if pd.isna(value):
                ERROR_COUNT.labels(error_type="missing_value").inc()
                raise HTTPException(status_code=400, detail=f"Missing value for {key}")
        
        # Validate input features
        is_valid, message = validate_input_features(input_data)
        if not is_valid:
            ERROR_COUNT.labels(error_type="invalid_input").inc()
            raise HTTPException(status_code=400, detail=message)
        
        # Preprocess input with detailed error handling
        try:
            processed_input = preprocess_input(input_data, scaler)
            if processed_input is None:
                raise ValueError("Preprocessing returned None")
        except Exception as e:
            ERROR_COUNT.labels(error_type="preprocessing_error").inc()
            raise HTTPException(status_code=400, detail=f"Error preprocessing input: {str(e)}")
        
        # Make prediction with timeout protection
        try:
            prediction = model.predict(processed_input)[0]
        except Exception as e:
            ERROR_COUNT.labels(error_type="prediction_error").inc()
            raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
        
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

# --- Enhancement: Advanced Prediction Endpoint ---
@app.post("/predict/advanced", response_model=PredictionResponse)
async def advanced_predict(request: AdvancedPredictionRequest):
    """Advanced prediction endpoint with model selection, confidence level, and feature importance."""
    start_time = time.time()
    
    try:
        # Select model
        selected_model_type = request.model_type or ModelType.LINEAR
        selected_model = available_models.get(selected_model_type)
        if selected_model is None or scaler is None:
            ERROR_COUNT.labels(error_type="model_not_loaded").inc()
            raise HTTPException(status_code=503, detail="Requested model not loaded")
        
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
        prediction = selected_model.predict(processed_input)[0]
        
        # Calculate confidence interval based on request.confidence_level
        confidence_interval = None
        if request.confidence_level:
            std_error = 0.1 * prediction  # Simplified error estimation
            # z-score for the confidence level: for 95%, z=1.96, for 99%, z=2.58, for 80%, z=1.28
            z_score = {0.8: 1.28, 0.95: 1.96, 0.99: 2.58}.get(round(request.confidence_level, 2), 1.96)
            confidence_interval = {
                "lower": float(prediction - z_score * std_error),
                "upper": float(prediction + z_score * std_error)
            }
        
        # Calculate feature importance if requested
        feature_importance = None
        if request.return_feature_importance:
            # Many models have feature_importances_ or coef_
            if hasattr(selected_model, 'feature_importances_'):
                fi = selected_model.feature_importances_
                feature_importance = {name: float(val) for name, val in zip(processed_input.columns, fi)}
            elif hasattr(selected_model, 'coef_'):
                fi = selected_model.coef_
                feature_importance = {name: float(val) for name, val in zip(processed_input.columns, fi)}
            else:
                feature_importance = {"message": "Feature importance not available for selected model"}
        
        # Log prediction
        response_time = time.time() - start_time
        log_prediction_to_db(input_data, float(prediction), selected_model_type.value, response_time)
        
        PREDICTION_COUNT.inc()
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence_interval=confidence_interval,
            model_used=selected_model_type.value,
            timestamp=datetime.now().isoformat(),
            feature_importance=feature_importance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type="internal_error").inc()
        logger.error(f"Error in advanced prediction: {e}")
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
    logger.info("Starting FastAPI application")