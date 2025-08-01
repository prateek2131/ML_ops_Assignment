import pytest
import os
import sys
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_training import ModelTrainer
from src.data_preprocessing import DataPreprocessor

class TestModelTrainer:
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create necessary directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Generate test data
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        preprocessor.preprocess_data(df)
        
        # Create params.yaml
        with open('params.yaml', 'w') as f:
            f.write("""
model_params:
  linear_regression:
    fit_intercept: true
  random_forest:
    n_estimators: 10
    max_depth: 5
    random_state: 42
  gradient_boosting:
    n_estimators: 10
    learning_rate: 0.1
    max_depth: 3
    random_state: 42
  svr:
    C: 1.0
    epsilon: 0.1
    kernel: rbf
""")
        
        self.trainer = ModelTrainer()
    
    def teardown_method(self):
        """Cleanup test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_load_data(self):
        """Test data loading"""
        X_train, X_test, y_train, y_test = self.trainer.load_data()
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_load_params(self):
        """Test parameter loading"""
        params = self.trainer.load_params()
        
        assert isinstance(params, dict)
        assert 'linear_regression' in params
        assert 'random_forest' in params
    
    def test_evaluate_model(self):
        """Test model evaluation"""
        from sklearn.linear_model import LinearRegression
        
        X_train, X_test, y_train, y_test = self.trainer.load_data()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        metrics, y_pred = self.trainer.evaluate_model(model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert len(y_pred) == len(y_test)
