# scripts/auto_retrain.py
"""
Automated model retraining system
"""
import schedule
import time
import os
import logging
from src.model_training import ModelTrainer
from src.data_preprocessing import DataPreprocessor
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AutoRetrainer:
    def __init__(self):
        self.retrain_threshold_days = 7
        self.performance_threshold = 0.05  # 5% performance drop
        
    def check_retrain_conditions(self):
        """Check if model needs retraining"""
        try:
            # Check model age
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            model_timestamp = datetime.fromisoformat(metadata['timestamp'])
            days_old = (datetime.now() - model_timestamp).days
            
            if days_old >= self.retrain_threshold_days:
                logger.info(f"Model is {days_old} days old, triggering retrain")
                return True
            
            # Check performance degradation
            current_performance = self.evaluate_current_model()
            baseline_performance = metadata['metrics']['r2']
            
            performance_drop = baseline_performance - current_performance
            if performance_drop > self.performance_threshold:
                logger.info(f"Performance dropped by {performance_drop:.3f}, triggering retrain")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
            return False
    
    def evaluate_current_model(self):
        """Evaluate current model on recent data"""
        # In a real scenario, this would use fresh data
        # For demo, we'll simulate performance evaluation
        try:
            # Load test data
            import pandas as pd
            from src.utils import load_model_and_scaler
            from sklearn.metrics import r2_score
            
            test_data = pd.read_csv('data/processed/test_data.csv')
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']
            
            model, scaler = load_model_and_scaler()
            y_pred = model.predict(X_test)
            
            return r2_score(y_test, y_pred)
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return 0.0
    
    def retrain_model(self):
        """Retrain the model"""
        logger.info("Starting automated model retraining...")
        
        try:
            # Backup current model
            backup_dir = f"models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)
            
            import shutil
            shutil.copy('models/best_model.joblib', f'{backup_dir}/best_model.joblib')
            shutil.copy('models/model_metadata.json', f'{backup_dir}/model_metadata.json')
            
            # Retrain models
            trainer = ModelTrainer()
            trainer.train_all_models()
            
            logger.info("Model retraining completed successfully")
            
            # Send notification (in production, this could be Slack, email, etc.)
            self.send_retrain_notification("success")
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            self.send_retrain_notification("failed", str(e))
    
    def send_retrain_notification(self, status, error=None):
        """Send notification about retraining status"""
        message = f"Model retraining {status}"
        if error:
            message += f": {error}"
        
        # In production, integrate with notification service
        logger.info(f"NOTIFICATION: {message}")
    
    def run_scheduler(self):
        """Run the retraining scheduler"""
        # Schedule daily checks
        schedule.every().day.at("02:00").do(self.check_and_retrain)
        
        # Schedule weekly forced retrain
        schedule.every().sunday.at("03:00").do(self.retrain_model)
        
        logger.info("Auto-retrainer scheduler started")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def check_and_retrain(self):
        """Check conditions and retrain if needed"""
        if self.check_retrain_conditions():
            self.retrain_model()

if __name__ == "__main__":
    retrainer = AutoRetrainer()
    retrainer.run_scheduler()