import pandas as pd
import numpy as np
import json
import logging
from scipy import stats
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_drift():
    """Check for data drift in input features"""
    
    try:
        # Load training data
        train_data = pd.read_csv('data/processed/train_data.csv')
        
        # For demonstration, we'll simulate new data
        # In practice, this would be recent production data
        np.random.seed(42)
        new_data = train_data.sample(n=100).copy()
        
        # Simulate some drift by slightly modifying the data
        new_data['MedInc'] *= np.random.normal(1.1, 0.1, len(new_data))
        
        # Calculate statistical tests for drift detection
        drift_detected = False
        drift_features = []
        
        for column in train_data.columns:
            if column == 'target':
                continue
                
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                train_data[column], 
                new_data[column]
            )
            
            # If p-value < 0.05, we reject null hypothesis (distributions are same)
            if p_value < 0.05:
                drift_detected = True
                drift_features.append(column)
                logger.warning(f"Drift detected in feature {column}: p-value = {p_value:.4f}")
        
        if drift_detected:
            logger.warning(f"Data drift detected in features: {drift_features}")
            # In a real scenario, you might want to trigger model retraining
            return False
        else:
            logger.info("No significant data drift detected")
            return True
            
    except Exception as e:
        logger.error(f"Error checking data drift: {e}")
        return False

if __name__ == "__main__":
    if not check_data_drift():
        logger.warning("Data drift detected - consider retraining the model")
        # Don't fail the pipeline for drift detection in this example
        # sys.exit(1)
    
    logger.info("Model drift check completed!")
