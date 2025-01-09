import numpy as np
from collections import deque
from typing import List, Tuple, Dict
import random

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int, state_dim: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer = deque(maxlen=capacity)
        
        # Statistics
        self.stats = {
            'rewards': [],
            'lengths': [],
            'wins': 0,
            'losses': 0
        }
    
    def push(self, state: np.ndarray, action: Tuple[int, int],
             reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done
        ))
        
        # Update statistics
        if done:
            self.stats['rewards'].append(reward)
            if reward > 0:
                self.stats['wins'] += 1
            else:
                self.stats['losses'] += 1
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)
        
        # Transpose batch of transitions to batch of components
        batch = list(zip(*transitions))
        
        return (
            np.array(batch[0]),  # states
            np.array(batch[1]),  # actions
            np.array(batch[2]),  # rewards
            np.array(batch[3]),  # next_states
            np.array(batch[4])   # dones
        )
    
    def get_stats(self) -> Dict:
        """Get current statistics of the buffer."""
        stats = self.stats.copy()
        if len(stats['rewards']) > 0:
            stats['avg_reward'] = np.mean(stats['rewards'][-100:])
            stats['win_rate'] = stats['wins'] / (stats['wins'] + stats['losses'])
        return stats
    
    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear buffer and reset statistics."""
        self.buffer.clear()
        self.stats = {
            'rewards': [],
            'lengths': [],
            'wins': 0,
            'losses': 0
        }
        