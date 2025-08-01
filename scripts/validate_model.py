import json
import sys
import logging
from src.model_training import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model_performance():
    """Validate that model meets minimum performance requirements"""
    
    # Load model metadata
    try:
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        logger.error("Model metadata not found. Please train models first.")
        return False
    
    # Define minimum performance thresholds
    MIN_R2_SCORE = 0.6
    MAX_RMSE = 1.0
    
    best_metrics = metadata.get('metrics', {})
    
    # Check R² score
    r2_score = best_metrics.get('r2', 0)
    if r2_score < MIN_R2_SCORE:
        logger.error(f"R² score {r2_score:.4f} is below minimum threshold {MIN_R2_SCORE}")
        return False
    
    # Check RMSE
    rmse = best_metrics.get('rmse', float('inf'))
    if rmse > MAX_RMSE:
        logger.error(f"RMSE {rmse:.4f} is above maximum threshold {MAX_RMSE}")
        return False
    
    logger.info(f"Model validation passed - R²: {r2_score:.4f}, RMSE: {rmse:.4f}")
    return True

if __name__ == "__main__":
    if not validate_model_performance():
        sys.exit(1)
    
    logger.info("Model validation completed successfully!")

