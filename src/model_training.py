import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import yaml
import json
from datetime import datetime
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name="california_housing_experiment"):
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
    def load_data(self):
        """Load preprocessed data"""
        logger.info("Loading preprocessed data...")
        
        train_data = pd.read_csv('data/processed/train_data.csv')
        test_data = pd.read_csv('data/processed/test_data.csv')
        
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        return X_train, X_test, y_train, y_test
    
    def load_params(self):
        """Load parameters from params.yaml"""
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)
        return params['model_params']
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test, params):
        """Train Linear Regression model"""
        with mlflow.start_run(run_name="linear_regression"):
            logger.info("Training Linear Regression...")
            
            model = LinearRegression(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="LinearRegression_CaliforniaHousing"
            )
            
            self.models['linear_regression'] = model
            self.results['linear_regression'] = metrics
            
            logger.info(f"Linear Regression - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
            return model, metrics
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, params):
        """Train Random Forest model"""
        with mlflow.start_run(run_name="random_forest"):
            logger.info("Training Random Forest...")
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="RandomForest_CaliforniaHousing"
            )
            
            self.models['random_forest'] = model
            self.results['random_forest'] = metrics
            
            logger.info(f"Random Forest - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
            return model, metrics
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test, params):
        """Train Gradient Boosting model"""
        with mlflow.start_run(run_name="gradient_boosting"):
            logger.info("Training Gradient Boosting...")
            
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log feature importance
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
            mlflow.log_dict(feature_importance, "feature_importance.json")
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="GradientBoosting_CaliforniaHousing"
            )
            
            self.models['gradient_boosting'] = model
            self.results['gradient_boosting'] = metrics
            
            logger.info(f"Gradient Boosting - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
            return model, metrics
    
    def train_svr(self, X_train, y_train, X_test, y_test, params):
        """Train Support Vector Regression model"""
        with mlflow.start_run(run_name="support_vector_regression"):
            logger.info("Training Support Vector Regression...")
            
            model = SVR(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="SVR_CaliforniaHousing"
            )
            
            self.models['svr'] = model
            self.results['svr'] = metrics
            
            logger.info(f"SVR - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            
            return model, metrics
    
    def train_all_models(self):
        """Train all models and compare performance"""
        logger.info("Starting model training pipeline...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Load parameters
        model_params = self.load_params()
        
        # Train all models
        self.train_linear_regression(X_train, y_train, X_test, y_test, model_params['linear_regression'])
        self.train_random_forest(X_train, y_train, X_test, y_test, model_params['random_forest'])
        self.train_gradient_boosting(X_train, y_train, X_test, y_test, model_params['gradient_boosting'])
        self.train_svr(X_train, y_train, X_test, y_test, model_params['svr'])
        
        # Find best model
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        best_model = self.models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with RMSE: {best_metrics['rmse']:.4f}")
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/best_model.joblib')
        
        # Save model metadata
        model_metadata = {
            'best_model': best_model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': best_metrics,
            'all_results': self.results
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Save metrics for DVC
        with open('metrics.json', 'w') as f:
            json.dump({
                'best_model_rmse': best_metrics['rmse'],
                'best_model_mae': best_metrics['mae'],
                'best_model_r2': best_metrics['r2']
            }, f, indent=2)
        
        return best_model, best_model_name, best_metrics
    
    def register_best_model(self, model_name, model):
        """Register best model in MLflow Model Registry"""
        try:
            # Register model
            model_version = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                f"BestModel_CaliforniaHousing"
            )
            
            logger.info(f"Model registered as version {model_version.version}")
            
            # Transition to staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name="BestModel_CaliforniaHousing",
                version=model_version.version,
                stage="Staging"
            )
            
            logger.info("Model transitioned to Staging")
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model, best_model_name, best_metrics = trainer.train_all_models()
    
    logger.info("Model training completed successfully!")
