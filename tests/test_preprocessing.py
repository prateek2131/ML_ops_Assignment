import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    
    def setup_method(self):
        """Setup test environment"""
        self.preprocessor = DataPreprocessor()
    
    def test_load_data(self):
        """Test data loading"""
        df = self.preprocessor.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'target' in df.columns
        assert len(df.columns) == 9  # 8 features + 1 target
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        # Load sample data
        df = self.preprocessor.load_data()
        
        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(df)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert len(X_train) > len(X_test)  # Train should be larger
        assert X_train.shape[1] == X_test.shape[1]  # Same number of features
        
        # Check if data is scaled (mean should be close to 0)
        assert abs(X_train.mean().mean()) < 0.1
    
    def test_get_data_info(self):
        """Test data info extraction"""
        df = self.preprocessor.load_data()
        info = self.preprocessor.get_data_info(df)
        
        assert isinstance(info, dict)
        assert 'shape' in info
        assert 'columns' in info
        assert 'missing_values' in info
        assert 'target_stats' in info
