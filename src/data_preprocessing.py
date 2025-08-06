"""
Data preprocessing module for California Housing dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load California Housing dataset"""
        try:
            logger.info("Loading California Housing dataset...")
            housing = fetch_california_housing()
            
            # Create DataFrame
            df = pd.DataFrame(housing.data, columns=housing.feature_names)
            df['target'] = housing.target
            
            # Save raw data
            os.makedirs('data/raw', exist_ok=True)
            df.to_csv('data/raw/california_housing_raw.csv', index=False)
            logger.info("Raw data saved to data/raw/california_housing_raw.csv")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocess the dataset"""
        logger.info("Preprocessing data...")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        
        train_data = X_train_scaled.copy()
        train_data['target'] = y_train.values
        train_data.to_csv('data/processed/train_data.csv', index=False)
        
        test_data = X_test_scaled.copy()
        test_data['target'] = y_test.values
        test_data.to_csv('data/processed/test_data.csv', index=False)
        
        # Save scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
        logger.info("Data preprocessing completed")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_data_info(self, df):
        """Get basic information about the dataset"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'target_stats': {
                'mean': df['target'].mean(),
                'std': df['target'].std(),
                'min': df['target'].min(),
                'max': df['target'].max()
            }
        }
        return info

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    raw_data = preprocessor.load_data()
    
    # Get data info
    data_info = preprocessor.get_data_info(raw_data)
    logger.info(f"Dataset info: {data_info}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(raw_data)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
