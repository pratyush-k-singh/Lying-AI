import logging
import os
from datetime import datetime
from typing import Dict, Any
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

class Logger:
    """Handles all logging for the training process."""
    
    def __init__(self, log_dir: str, exp_name: str):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            exp_name: Name of experiment
        """
        self.exp_name = exp_name
        self.log_dir = os.path.join(log_dir, exp_name)
        self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # Setup file logging
        self.log_file = os.path.join(self.log_dir, 'training.log')
        self._setup_file_logging()
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # Training metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'losses': []
        }
    
    def _setup_file_logging(self) -> None:
        """Setup logging to file."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
    
    def _config_to_dict(self, config: Any) -> Dict:
        """Convert config object to dictionary."""
        if hasattr(config, '__dict__'):
            return {k: self._config_to_dict(v) for k, v in config.__dict__.items()}
        elif hasattr(config, '_asdict'):  # For namedtuples
            return {k: self._config_to_dict(v) for k, v in config._asdict().items()}
        elif isinstance(config, (list, tuple)):
            return [self._config_to_dict(x) for x in config]
        elif isinstance(config, dict):
            return {k: self._config_to_dict(v) for k, v in config.items()}
        else:
            return config
    
    def log_config(self, config: Any) -> None:
        """Log configuration."""
        config_dict = self._config_to_dict(config)
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to tensorboard and update internal metrics."""
        # Update internal metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_episode(self, episode: int, rewards: float, length: int,
                   loss: float, epsilon: float) -> None:
        """Log episode information."""
        metrics = {
            'episode_rewards': rewards,
            'episode_lengths': length,
            'losses': loss,
            'epsilon': epsilon
        }
        self.log_metrics(metrics, episode)
        
        if episode % 100 == 0:
            logging.info(
                f"Episode {episode} - "
                f"Reward: {rewards:.2f}, "
                f"Length: {length}, "
                f"Loss: {loss:.4f}, "
                f"Epsilon: {epsilon:.4f}"
            )
    
    def log_evaluation(self, episode: int, rewards: float, win_rate: float) -> None:
        """Log evaluation results."""
        metrics = {
            'eval_rewards': rewards,
            'eval_win_rate': win_rate
        }
        self.log_metrics(metrics, episode)
        
        logging.info(
            f"Evaluation - "
            f"Episode {episode}, "
            f"Average Reward: {rewards:.2f}, "
            f"Win Rate: {win_rate:.2f}"
        )
    
    def log_model_save(self, path: str) -> None:
        """Log model checkpoint save."""
        logging.info(f"Model checkpoint saved to {path}")
    
    def save_metrics(self) -> None:
        """Save all metrics to file."""
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")
    
    def close(self) -> None:
        """Close tensorboard writer and save final metrics."""
        self.save_metrics()
        self.writer.close()
        