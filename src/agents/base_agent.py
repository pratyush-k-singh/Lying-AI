from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np

class BaseAgent(ABC):
    """Abstract base class for Liar's Dice agents."""
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
    
    @abstractmethod
    def select_action(self, state: np.ndarray, legal_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select an action given the current state.
        
        Args:
            state: Current game state observation
            legal_actions: List of legal actions in current state
            
        Returns:
            Tuple of (quantity, face_value) representing the action
        """
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: Tuple[int, int], 
               reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Update the agent's internal state based on experience.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode is done
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent state to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent state from disk."""
        pass
    
    def __str__(self) -> str:
        return self.name
    