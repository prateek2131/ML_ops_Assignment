import logging
import logging.handlers
import os
from datetime import datetime
import json

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f'logs/mlops_{datetime.now().strftime("%Y%m%d")}.log'
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # API specific logger
    api_logger = logging.getLogger('api')
    api_handler = logging.handlers.RotatingFileHandler(
        'logs/api_requests.log', maxBytes=10*1024*1024, backupCount=5
    )
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(detailed_formatter)
    api_logger.addHandler(api_handler)
    
    return root_logger

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry)

def get_structured_logger(name):
    """Get a logger with JSON formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            f'logs/{name}_structured.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger
