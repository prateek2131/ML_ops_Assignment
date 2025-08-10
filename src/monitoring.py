# Enhanced monitoring in src/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, Info
import psutil
import threading
import time
import json
import logging
 

# Custom metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy score')
FEATURE_DRIFT = Gauge('feature_drift_score', 'Feature drift detection score', ['feature'])
SYSTEM_MEMORY = Gauge('system_memory_usage', 'System memory usage percentage')
ACTIVE_MODELS = Info('active_models', 'Information about active models')

class ModelMonitor:
    def __init__(self):
        self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start background system monitoring"""
        def monitor():
            while True:
                # Monitor system resources
                memory_percent = psutil.virtual_memory().percent
                SYSTEM_MEMORY.set(memory_percent)
                
                # Monitor model performance
                self.update_model_metrics()
                
                time.sleep(30)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def update_model_metrics(self):
        """Update model-specific metrics"""
        try:
            # Load current model metadata
            with open('models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            MODEL_ACCURACY.set(metadata['metrics']['r2'])
            
            # Set active model info
            ACTIVE_MODELS.info({
                'model_name': metadata['best_model'],
                'timestamp': metadata['timestamp'],
                'version': '1.0.0'
            })
            
        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")