# src/ab_testing.py
"""
A/B Testing framework for model comparison
"""
import random
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sqlite3

logger = logging.getLogger(__name__)

class ABTestManager:
    def __init__(self, db_path: str = 'logs/ab_tests.db'):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize A/B testing database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                user_id TEXT,
                variant TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                prediction REAL NOT NULL,
                response_time REAL,
                feedback_score REAL,
                conversion BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_configs (
                test_name TEXT PRIMARY KEY,
                variants TEXT NOT NULL,
                traffic_split TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_test(self, test_name: str, variants: Dict[str, float], 
                   start_date: str, end_date: Optional[str] = None):
        """Create a new A/B test"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO test_configs 
            (test_name, variants, traffic_split, start_date, end_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            test_name,
            json.dumps(list(variants.keys())),
            json.dumps(variants),
            start_date,
            end_date
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created A/B test: {test_name} with variants: {variants}")
    
    def get_variant(self, test_name: str, user_id: Optional[str] = None) -> str:
        """Get variant assignment for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get test configuration
        cursor.execute('''
            SELECT traffic_split, status FROM test_configs 
            WHERE test_name = ? AND status = 'active'
        ''', (test_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return "control"  # Default variant
        
        traffic_split = json.loads(result[0])
        
        # Consistent assignment based on user_id if provided
        if user_id:
            random.seed(hash(f"{test_name}_{user_id}"))
        
        rand_val = random.random()
        cumulative = 0
        
        for variant, weight in traffic_split.items():
            cumulative += weight
            if rand_val <= cumulative:
                return variant
        
        return "control"
    
    def log_test_result(self, test_name: str, user_id: Optional[str], 
                       variant: str, input_data: Dict, prediction: float,
                       response_time: float):
        """Log A/B test result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ab_tests 
            (test_name, user_id, variant, timestamp, input_data, 
             prediction, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_name,
            user_id,
            variant,
            datetime.now().isoformat(),
            json.dumps(input_data),
            prediction,
            response_time
        ))
        
        conn.commit()
        conn.close()
    
    def get_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get A/B test results and statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get results by variant
        cursor.execute('''
            SELECT 
                variant,
                COUNT(*) as total_requests,
                AVG(response_time) as avg_response_time,
                AVG(prediction) as avg_prediction,
                AVG(CASE WHEN feedback_score IS NOT NULL THEN feedback_score END) as avg_feedback,
                COUNT(CASE WHEN conversion = 1 THEN 1 END) as conversions
            FROM ab_tests 
            WHERE test_name = ?
            GROUP BY variant
        ''', (test_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        test_results = {}
        for row in results:
            variant = row[0]
            test_results[variant] = {
                'total_requests': row[1],
                'avg_response_time': row[2],
                'avg_prediction': row[3],
                'avg_feedback': row[4],
                'conversions': row[5],
                'conversion_rate': row[5] / row[1] if row[1] > 0 else 0
            }
        
        return test_results

# Integration with FastAPI
def get_model_variant(test_manager: ABTestManager, user_id: Optional[str] = None):
    """Get model variant for A/B testing"""
    variant = test_manager.get_variant("model_comparison", user_id)
    
    variant_models = {
        "control": "random_forest",
        "challenger": "gradient_boosting"
    }
    
    return variant_models.get(variant, "random_forest"), variant