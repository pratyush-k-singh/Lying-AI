import random
import numpy as np
from typing import Tuple, List
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """Agent that selects random legal actions."""
    
    def __init__(self, challenge_prob: float = 0.2):
        super().__init__(name="RandomAgent")
        self.challenge_prob = challenge_prob
    
    def select_action(self, state: np.ndarray, legal_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Randomly select a legal action."""
        if not legal_actions:  # If no legal actions available
            return (-1, -1)  # Default to challenge
            
        # If challenge is available and random check passes, choose challenge
        if (-1, -1) in legal_actions and random.random() < self.challenge_prob:
            return (-1, -1)
            
        # Get all non-challenge actions
        bid_actions = [action for action in legal_actions if action != (-1, -1)]
        
        # If no bid actions available but there are legal actions, must be challenge
        if not bid_actions and legal_actions:
            return (-1, -1)
            
        # Otherwise choose random bid action
        if bid_actions:
            return random.choice(bid_actions)
        else:
            return (-1, -1)  # Fallback to challenge if something goes wrong
    
    def update(self, state: np.ndarray, action: Tuple[int, int], 
               reward: float, next_state: np.ndarray, done: bool) -> None:
        """Random agent doesn't learn."""
        pass
    
    def save(self, path: str) -> None:
        """Nothing to save for random agent."""
        pass
    
    def load(self, path: str) -> None:
        """Nothing to load for random agent."""
        pass
    