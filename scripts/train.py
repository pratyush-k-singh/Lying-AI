import argparse
import os
from tqdm import tqdm
import numpy as np
import torch

from src.env.liars_dice_env import LiarsDiceEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.utils.config import Config
from src.utils.logging import Logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train Liar\'s Dice Agent')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()

def evaluate_agent(env: LiarsDiceEnv, agent: DQNAgent, 
                  opponent: RandomAgent, num_episodes: int) -> tuple:
    """Evaluate agent against random opponent."""
    rewards = []
    wins = 0
    
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Agent's turn
            action = agent.select_action(state, info['legal_actions'])
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if not done:
                # Opponent's turn
                opp_state = env._get_observation()  # Get state from opponent's perspective
                opp_action = opponent.select_action(opp_state, info['legal_actions'])
                next_state, reward, done, info = env.step(opp_action)
                episode_reward -= reward  # Negate opponent's reward
            
            state = next_state
        
        rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1
    
    return np.mean(rewards), wins / num_episodes

def train(config: Config):
    """Main training loop."""
    # Setup environment and agents
    env = LiarsDiceEnv(
        num_dice=config.env.num_dice,
        num_faces=config.env.num_faces
    )
    
    state_dim = env.num_faces + 2  # dice counts + current bid
    action_dim = 1 + 10 * 6  # challenge + (quantity * face_value)
    
    agent = DQNAgent(
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
        device=config.device
    )
    
    opponent = RandomAgent()
    
    # Setup logging
    logger = Logger(config.log_dir, config.exp_name)
    logger.log_config(config.__dict__)
    
    # Training loop
    best_win_rate = 0
    episode_rewards = []
    
    for episode in tqdm(range(config.training.num_episodes)):
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        total_loss = 0
        num_updates = 0
        
        while not done:
            # Agent's turn
            action = agent.select_action(state, info['legal_actions'])
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if not done:
                # Opponent's turn
                opp_state = env._get_observation()
                opp_action = opponent.select_action(opp_state, info['legal_actions'])
                next_state, reward, done, info = env.step(opp_action)
                episode_reward -= reward
            
            # Store transition and train
            agent.update(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                total_loss += loss
                num_updates += 1
            
            state = next_state
        
        # Log episode results
        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        logger.log_episode(
            episode=episode,
            rewards=episode_reward,
            length=episode_length,
            loss=avg_loss,
            epsilon=agent.epsilon
        )
        
        # Periodic evaluation
        if episode % config.training.eval_frequency == 0:
            eval_reward, win_rate = evaluate_agent(
                env, agent, opponent, config.training.num_eval_episodes
            )
            logger.log_evaluation(episode, eval_reward, win_rate)
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                save_path = os.path.join(
                    config.save_dir,
                    f"{config.exp_name}_best.pt"
                )
                agent.save(save_path)
                logger.log_model_save(save_path)
        
        # Periodic checkpointing
        if episode % config.training.checkpoint_freq == 0:
            save_path = os.path.join(
                config.save_dir,
                f"{config.exp_name}_episode_{episode}.pt"
            )
            agent.save(save_path)
            logger.log_model_save(save_path)
    
    # Final evaluation and saving
    eval_reward, win_rate = evaluate_agent(
        env, agent, opponent, config.training.num_eval_episodes
    )
    logger.log_evaluation(config.training.num_episodes, eval_reward, win_rate)
    
    final_path = os.path.join(config.save_dir, f"{config.exp_name}_final.pt")
    agent.save(final_path)
    logger.log_model_save(final_path)
    
    logger.close()

if __name__ == "__main__":
    args = parse_args()
    config = Config.from_yaml(args.config)
    train(config)
    