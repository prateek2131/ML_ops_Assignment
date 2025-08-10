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
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name="california_housing_experiment"):
        self.experiment_name = experiment_name
        self.models = {}
        self.results = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
    @contextmanager
    def mlflow_run(self, run_name):
        """Context manager for MLflow runs to ensure proper cleanup"""
        with mlflow.start_run(run_name=run_name):
            try:
                yield
            except Exception as e:
                logger.error(f"Error in MLflow run {run_name}: {str(e)}")
                raise
        
    def load_data(self):
        """Load preprocessed data with error handling"""
        logger.info("Loading preprocessed data...")
        logger.info(f"Current directory: {os.getcwd()}")
        
        try:
            train_data_path = './data/processed/train_data.csv'
            test_data_path = './data/processed/test_data.csv'
            
            if not os.path.exists(train_data_path):
                raise FileNotFoundError(f"Training data not found at {train_data_path}")
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"Test data not found at {test_data_path}")
                
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            
            if 'target' not in train_data.columns or 'target' not in test_data.columns:
                raise ValueError("'target' column not found in data files")
            
            X_train = train_data.drop('target', axis=1)
            y_train = train_data['target']
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']
            
            logger.info(f"Data loaded successfully - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_params(self):
        """Load parameters from params.yaml with error handling"""
        try:
            params_path = 'params.yaml'
            if not os.path.exists(params_path):
                raise FileNotFoundError(f"Parameters file not found at {params_path}")
                
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
                
            if 'model_params' not in params:
                raise KeyError("'model_params' not found in params.yaml")
                
            logger.info("Parameters loaded successfully")
            return params['model_params']
            
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2': float(r2_score(y_test, y_pred))
            }
            
            return metrics, y_pred
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test, params):
        """Train Linear Regression model"""
        with self.mlflow_run("linear_regression"):
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
        with self.mlflow_run("random_forest"):
            logger.info("Training Random Forest...")
            
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
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
        with self.mlflow_run("gradient_boosting"):
            logger.info("Training Gradient Boosting...")
            
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, y_pred = self.evaluate_model(model, X_test, y_test)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
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
        with self.mlflow_run("support_vector_regression"):
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
    
    def save_results(self, best_model, best_model_name, best_metrics):
        """Save model results and metadata"""
        try:
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Save best model
            model_path = 'models/best_model.joblib'
            joblib.dump(best_model, model_path)
            logger.info(f"Best model saved to {model_path}")
            
            # Save model metadata
            model_metadata = {
                'best_model': best_model_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': best_metrics,
                'all_results': self.results
            }
            
            metadata_path = 'models/model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            logger.info(f"Model metadata saved to {metadata_path}")
            
            # Save metrics for DVC
            dvc_metrics = {
                'best_model_rmse': best_metrics['rmse'],
                'best_model_mae': best_metrics['mae'],
                'best_model_r2': best_metrics['r2']
            }
            
            metrics_path = 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(dvc_metrics, f, indent=2)
            logger.info(f"DVC metrics saved to {metrics_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def train_all_models(self):
        """Train all models and compare performance"""
        logger.info("Starting model training pipeline...")
        
        try:
            # Load data and parameters
            X_train, X_test, y_train, y_test = self.load_data()
            model_params = self.load_params()
            
            # Define model training functions
            model_types = {
                'linear_regression': self.train_linear_regression,
                'random_forest': self.train_random_forest,
                'gradient_boosting': self.train_gradient_boosting,
                'svr': self.train_svr
            }
            
            # Train all models with individual error handling
            successful_models = 0
            for model_name, train_func in model_types.items():
                try:
                    if model_name not in model_params:
                        logger.warning(f"Parameters for {model_name} not found, skipping...")
                        continue
                        
                    train_func(X_train, y_train, X_test, y_test, model_params[model_name])
                    successful_models += 1
                    logger.info(f"Successfully trained {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            
            # Check if any models were successfully trained
            if successful_models == 0 or not self.results:
                raise RuntimeError("No models were successfully trained")
            
            # Find best model based on RMSE
            best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
            best_model = self.models[best_model_name]
            best_metrics = self.results[best_model_name]
            
            logger.info(f"Best model: {best_model_name} with RMSE: {best_metrics['rmse']:.4f}")
            
            # Save results
            self.save_results(best_model, best_model_name, best_metrics)
            
            # Register best model (optional) - can be disabled if causing issues
            try:
                self.register_best_model(best_model_name, best_model)
            except Exception as e:
                logger.warning(f"Could not register best model: {str(e)}")
                logger.info("Continuing without model registration - models are still saved locally")
            
            return best_model, best_model_name, best_metrics
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {str(e)}")
            raise
    
    def register_best_model(self, model_name, model):
        """Register best model in MLflow Model Registry"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            if experiment is None:
                logger.error("Experiment not found for model registration")
                return
            
            # Find the run for the best model - fix the filter string
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.`mlflow.runName` = '{model_name}'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                logger.warning(f"No runs found for model {model_name}, skipping registration")
                return
            
            best_run = runs[0]
            run_id = best_run.info.run_id
            
            logger.info(f"Registering model from run {run_id}")
            
            # Register model with error handling for serialization issues
            try:
                model_uri = f"runs:/{run_id}/model"
                model_version = mlflow.register_model(
                    model_uri,
                    "BestModel_CaliforniaHousing",
                    await_registration_for=300  # Wait up to 5 minutes
                )
                
                logger.info(f"Model registered as version {model_version.version}")
                
                # Transition to staging with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        client.transition_model_version_stage(
                            name="BestModel_CaliforniaHousing",
                            version=model_version.version,
                            stage="Staging"
                        )
                        logger.info("Model transitioned to Staging")
                        break
                    except Exception as stage_error:
                        if attempt == max_retries - 1:
                            raise stage_error
                        logger.warning(f"Stage transition attempt {attempt + 1} failed, retrying...")
                        import time
                        time.sleep(2)
                        
            except Exception as reg_error:
                logger.error(f"Model registration failed: {str(reg_error)}")
                # Try alternative registration approach
                logger.info("Attempting alternative registration method...")
                
                # Log the model again in a new run to avoid serialization issues
                with mlflow.start_run(run_name=f"{model_name}_registration"):
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        registered_model_name="BestModel_CaliforniaHousing"
                    )
                    logger.info("Model registered using alternative method")
                    
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            # Don't raise the exception - registration is optional
    
    def get_model_comparison_summary(self):
        """Return a summary of all trained models"""
        if not self.results:
            return "No models have been trained yet."
        
        summary = "\n" + "="*60 + "\n"
        summary += "MODEL COMPARISON SUMMARY\n"
        summary += "="*60 + "\n"
        
        for model_name, metrics in self.results.items():
            summary += f"\n{model_name.upper().replace('_', ' ')}:\n"
            summary += f"  RMSE: {metrics['rmse']:.4f}\n"
            summary += f"  MAE:  {metrics['mae']:.4f}\n"
            summary += f"  R²:   {metrics['r2']:.4f}\n"
        
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['rmse'])
        summary += f"\nBest Model: {best_model.upper().replace('_', ' ')} (lowest RMSE)\n"
        summary += "="*60
        
        return summary

def main():
    """Main execution function"""
    try:
        trainer = ModelTrainer()
        best_model, best_model_name, best_metrics = trainer.train_all_models()
        
        # Print summary
        print(trainer.get_model_comparison_summary())
        
        logger.info("Model training completed successfully!")
        return trainer
        
    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    trainer = main()