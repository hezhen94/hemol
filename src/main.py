# --------------------------
# src/main.py
# --------------------------
import argparse
import os
from src.config import Config
from src.training import TrainingSystem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    system = TrainingSystem(config)
    
    if os.path.exists("checkpoints/latest.zip"):
        system.model = PPO.load("checkpoints/latest.zip", env=system.env)
    else:
        system.initialize_model()
    
    try:
        system.train(total_steps=50000)
    except Exception as e:
        print(f"Training failed: {str(e)}")