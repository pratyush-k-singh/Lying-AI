from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from src.models.networks import DQNetwork
from src.agents.base_agent import BaseAgent
from src.training.replay_buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    """Deep Q-Network agent for Liar's Dice with improved stability."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = "cuda"
    ):
        super().__init__("DQNAgent")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        
        # Create networks
        self.policy_net = DQNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim)
        
        # Training tracking
        self.train_steps = 0
    
    def select_action(self, state: np.ndarray, legal_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return legal_actions[np.random.randint(len(legal_actions))]
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, _ = self.policy_net(state_tensor)
            
            # Create legal action mask
            legal_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
            for action in legal_actions:
                idx = self._action_to_idx(action)
                legal_mask[idx] = True
            
            # Mask illegal actions with large negative values
            q_values = q_values.squeeze()
            q_values[~legal_mask] = float('-inf')
            
            # Select best legal action
            action_idx = q_values.argmax().item()
            return self._idx_to_action(action_idx)
    
    def update(self, state: np.ndarray, action: Tuple[int, int], 
               reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        action_idx = self._action_to_idx(action)
        # Scale reward for better stability
        scaled_reward = reward * 0.1
        self.replay_buffer.push(state, action_idx, scaled_reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """Train the network on a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get current Q values
        current_q_values, current_state_values = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions)
        current_state_values = current_state_values.unsqueeze(1)
        
        # Get target Q values
        with torch.no_grad():
            next_q_values, next_state_values = self.target_net(next_states)
            next_q_values = next_q_values.max(1, keepdim=True)[0]
            next_state_values = next_state_values.unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * (next_state_values + next_q_values)
            
            # Clip target values for stability
            target_q_values = torch.clamp(target_q_values, -10.0, 10.0)
        
        # Compute loss with Huber loss for stability
        q_loss = F.smooth_l1_loss(current_q_values, target_q_values)
        value_loss = F.smooth_l1_loss(current_state_values, target_q_values.detach())
        loss = q_loss + value_loss
        
        # Clip loss for stability
        loss = torch.clamp(loss, -100.0, 100.0)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def _action_to_idx(self, action: Tuple[int, int]) -> int:
        """Convert action tuple to index."""
        if action == (-1, -1):  # Challenge action
            return 0
        quantity, face = action
        return 1 + (quantity - 1) * 6 + (face - 1)
    
    def _idx_to_action(self, idx: int) -> Tuple[int, int]:
        """Convert index to action tuple."""
        if idx == 0:  # Challenge action
            return (-1, -1)
        idx -= 1  # Adjust for challenge action
        quantity = (idx // 6) + 1
        face = (idx % 6) + 1
        return (quantity, face)
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
        