import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_training import ModelTrainer

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()

