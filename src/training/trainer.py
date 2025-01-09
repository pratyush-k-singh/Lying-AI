import os
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

from src.env.liars_dice_env import LiarsDiceEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.utils.config import Config
from src.utils.logging import Logger

class Trainer:
    """Handles training of Liar's Dice agents."""
    
    def __init__(
        self,
        config: Config,
        agent: Optional[DQNAgent] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.config = config
        
        # Setup environment
        self.env = LiarsDiceEnv(
            num_dice=config.env.num_dice,
            num_faces=config.env.num_faces
        )
        
        # Create or load agent
        if agent is None:
            state_dim = self.env.num_faces + 2  # dice counts + current bid
            action_dim = 1 + 10 * 6  # challenge + (quantity * face_value)
            
            self.agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config.model.hidden_dim,
                learning_rate=config.training.learning_rate,
                gamma=config.training.gamma,
                epsilon_start=config.training.epsilon_start,
                epsilon_end=config.training.epsilon_end,
                epsilon_decay=config.training.epsilon_decay,
                buffer_size=config.training.replay_buffer_size,
                batch_size=config.training.batch_size,
                target_update_freq=config.training.target_update_freq,
                device=config.device
            )
            
            if checkpoint_path:
                self.agent.load(checkpoint_path)
        else:
            self.agent = agent
        
        # Create opponent (random agent for now)
        self.opponent = RandomAgent()
        
        # Setup logging
        self.logger = Logger(config.log_dir, config.exp_name)
        self.logger.log_config(config.__dict__)
        
        # Training stats
        self.best_win_rate = 0.0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self) -> None:
        """Main training loop."""
        progress_bar = tqdm(range(self.config.training.num_episodes))
        
        for episode in progress_bar:
            # Run training episode
            metrics = self._train_episode()
            
            # Update progress bar
            progress_bar.set_description(
                f"Episode {episode} - "
                f"Reward: {metrics['episode_reward']:.2f}, "
                f"Loss: {metrics['avg_loss']:.4f}, "
                f"Epsilon: {self.agent.epsilon:.4f}"
            )
            
            # Log episode results
            self.logger.log_episode(
                episode=episode,
                rewards=metrics['episode_reward'],
                length=metrics['episode_length'],
                loss=metrics['avg_loss'],
                epsilon=self.agent.epsilon
            )
            
            # Periodic evaluation
            if episode % self.config.training.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self.logger.log_evaluation(
                    episode,
                    eval_metrics['avg_reward'],
                    eval_metrics['win_rate']
                )
                
                # Save best model
                if eval_metrics['win_rate'] > self.best_win_rate:
                    self.best_win_rate = eval_metrics['win_rate']
                    self._save_model('best')
            
            # Periodic checkpointing
            if episode % self.config.training.checkpoint_freq == 0:
                self._save_model(f'episode_{episode}')
        
        # Final evaluation and saving
        eval_metrics = self._evaluate()
        self.logger.log_evaluation(
            self.config.training.num_episodes,
            eval_metrics['avg_reward'],
            eval_metrics['win_rate']
        )
        
        self._save_model('final')
        self.logger.close()
    
    def _train_episode(self) -> Dict:
        """Run a single training episode."""
        state, info = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        total_loss = 0
        num_updates = 0
        
        while not done:
            # Agent's turn
            action = self.agent.select_action(state, info['legal_actions'])
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if not done:
                # Opponent's turn
                opp_state = self.env._get_observation()
                opp_action = self.opponent.select_action(opp_state, info['legal_actions'])
                next_state, reward, done, info = self.env.step(opp_action)
                episode_reward -= reward
            
            # Store transition and train
            self.agent.update(state, action, reward, next_state, done)
            loss = self.agent.train()
            if loss is not None:
                total_loss += loss
                num_updates += 1
            
            state = next_state
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'avg_loss': avg_loss
        }
    
    def _evaluate(self, num_episodes: int = None) -> Dict:
        """Evaluate agent against random opponent."""
        if num_episodes is None:
            num_episodes = self.config.training.num_eval_episodes
        
        rewards = []
        wins = 0
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Agent's turn
                action = self.agent.select_action(state, info['legal_actions'])
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if not done:
                    # Opponent's turn
                    opp_state = self.env._get_observation()
                    opp_action = self.opponent.select_action(opp_state, info['legal_actions'])
                    next_state, reward, done, info = self.env.step(opp_action)
                    episode_reward -= reward
                
                state = next_state
            
            rewards.append(episode_reward)
            if episode_reward > 0:
                wins += 1
        
        return {
            'avg_reward': np.mean(rewards),
            'win_rate': wins / num_episodes
        }
    
    def _save_model(self, tag: str) -> None:
        """Save model checkpoint."""
        save_path = os.path.join(
            self.config.save_dir,
            f"{self.config.exp_name}_{tag}.pt"
        )
        self.agent.save(save_path)
        self.logger.log_model_save(save_path)
        