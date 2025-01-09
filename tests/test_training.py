import pytest
import os
import torch
from src.training.trainer import Trainer
from src.utils.config import Config, EnvConfig, ModelConfig, TrainingConfig

class TestTrainer:
    @pytest.fixture
    def config(self, tmp_path):
        return Config(
            env=EnvConfig(num_dice=2, num_faces=6),
            model=ModelConfig(hidden_dim=32),
            training=TrainingConfig(
                num_episodes=10,
                batch_size=4,
                eval_frequency=5,
                num_eval_episodes=2,
                checkpoint_freq=5
            ),
            device="cpu",
            exp_name="test",
            save_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs")
        )
    
    @pytest.fixture
    def trainer(self, config):
        return Trainer(config)
    
    def test_initialization(self, trainer, config):
        """Test trainer initialization."""
        assert trainer.config == config
        assert trainer.best_win_rate == 0.0
        assert len(trainer.episode_rewards) == 0
        assert len(trainer.episode_lengths) == 0
    
    def test_train_episode(self, trainer):
        """Test single training episode."""
        metrics = trainer._train_episode()
        
        assert 'episode_reward' in metrics
        assert 'episode_length' in metrics
        assert 'avg_loss' in metrics
        
        assert isinstance(metrics['episode_reward'], (int, float))
        assert isinstance(metrics['episode_length'], int)
        assert isinstance(metrics['avg_loss'], float)
    
    def test_evaluate(self, trainer):
        """Test agent evaluation."""
        metrics = trainer._evaluate(num_episodes=2)
        
        assert 'avg_reward' in metrics
        assert 'win_rate' in metrics
        
        assert isinstance(metrics['avg_reward'], float)
        assert isinstance(metrics['win_rate'], float)
        assert 0 <= metrics['win_rate'] <= 1
    
    def test_save_model(self, trainer, tmp_path):
        """Test model checkpointing."""
        tag = "test_save"
        trainer._save_model(tag)
        
        save_path = os.path.join(
            trainer.config.save_dir,
            f"{trainer.config.exp_name}_{tag}.pt"
        )
        assert os.path.exists(save_path)
        
        # Verify saved model can be loaded
        checkpoint = torch.load(save_path)
        assert 'policy_net' in checkpoint
        assert 'target_net' in checkpoint
        assert 'optimizer' in checkpoint
        assert 'epsilon' in checkpoint
    
    def test_short_training_run(self, trainer):
        """Test complete training run with minimal episodes."""
        trainer.config.training.num_episodes = 2
        trainer.train()
        
        # Verify model checkpoints
        final_path = os.path.join(
            trainer.config.save_dir,
            f"{trainer.config.exp_name}_final.pt"
        )
        assert os.path.exists(final_path)
        
        # Verify logging
        log_path = os.path.join(
            trainer.config.log_dir,
            trainer.config.exp_name,
            'training.log'
        )
        assert os.path.exists(log_path)
    
    def test_resume_training(self, trainer, tmp_path):
        """Test resuming training from checkpoint."""
        # Run initial training
        trainer.config.training.num_episodes = 2
        trainer.train()
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            trainer.config.save_dir,
            f"{trainer.config.exp_name}_final.pt"
        )
        
        # Create new trainer with checkpoint
        resumed_trainer = Trainer(
            trainer.config,
            checkpoint_path=checkpoint_path
        )
        
        # Run additional training
        resumed_trainer.config.training.num_episodes = 2
        resumed_trainer.train()
        
        # Verify continued training was successful
        assert resumed_trainer.best_win_rate >= 0
        