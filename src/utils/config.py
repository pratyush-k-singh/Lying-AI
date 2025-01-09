from dataclasses import dataclass
from typing import Optional
import yaml
import os

@dataclass
class EnvConfig:
    """Environment configuration."""
    num_dice: int = 5
    num_faces: int = 6

@dataclass
class ModelConfig:
    """Neural network configuration."""
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration."""
    num_episodes: int = 10000
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    replay_buffer_size: int = 10000
    eval_frequency: int = 100
    num_eval_episodes: int = 100
    checkpoint_freq: int = 1000

@dataclass
class Config:
    """Main configuration class."""
    env: EnvConfig
    model: ModelConfig
    training: TrainingConfig
    device: str = "cuda"
    exp_name: str = "default"
    save_dir: str = "models"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        env_config = EnvConfig(**config_dict.get('env', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            env=env_config,
            model=model_config,
            training=training_config,
            device=config_dict.get('device', 'cuda'),
            exp_name=config_dict.get('exp_name', 'default'),
            save_dir=config_dict.get('save_dir', 'models'),
            log_dir=config_dict.get('log_dir', 'logs')
        )
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'env': self.env.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'device': self.device,
            'exp_name': self.exp_name,
            'save_dir': self.save_dir,
            'log_dir': self.log_dir
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Default configuration
DEFAULT_CONFIG = {
    'env': EnvConfig().__dict__,
    'model': ModelConfig().__dict__,
    'training': TrainingConfig().__dict__,
    'device': 'cuda',
    'exp_name': 'default',
    'save_dir': 'models',
    'log_dir': 'logs'
}
