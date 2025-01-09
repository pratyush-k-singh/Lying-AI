import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class DQNetwork(nn.Module):
    """Deep Q-Network for Liar's Dice."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_hidden_layers: int = 2):
        super().__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dim),
                 nn.ReLU()]
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        
        # Output layers for policy (actions) and value
        self.shared_layers = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of:
                - Policy logits of shape (batch_size, output_dim)
                - Value predictions of shape (batch_size, 1)
        """
        features = self.shared_layers(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

class ActorCritic(nn.Module):
    """Actor-Critic network architecture for Liar's Dice."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of:
                - Action probabilities
                - State value estimate
        """
        features = self.features(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return F.softmax(action_logits, dim=-1), value
    