import pytest
import torch
import numpy as np
from src.agents.dqn_agent import DQNAgent

class TestDQNAgent:
    @pytest.fixture
    def agent(self):
        return DQNAgent(
            state_dim=8,  # 6 faces + 2 bid values
            action_dim=61,  # challenge + (10 quantities * 6 faces)
            hidden_dim=64,
            device="cpu"  # Use CPU for testing
        )
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert isinstance(agent.policy_net, torch.nn.Module)
        assert isinstance(agent.target_net, torch.nn.Module)
        assert agent.epsilon == 1.0  # Should start with full exploration
        
        # Networks should have same initial weights
        for p1, p2 in zip(agent.policy_net.parameters(), 
                         agent.target_net.parameters()):
            assert torch.equal(p1, p2)
    
    def test_action_conversion(self, agent):
        """Test action conversion between tuples and indices."""
        test_actions = [
            (-1, -1),  # Challenge
            (1, 1),    # One one
            (3, 4),    # Three fours
            (10, 6)    # Ten sixes
        ]
        
        for action in test_actions:
            idx = agent._action_to_idx(action)
            converted = agent._idx_to_action(idx)
            assert converted == action
    
    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.random(8)
        legal_actions = [(1, 1), (1, 2), (2, 2)]
        
        # Test deterministic selection (epsilon = 0)
        agent.epsilon = 0
        action = agent.select_action(state, legal_actions)
        assert action in legal_actions
        
        # Test random selection (epsilon = 1)
        agent.epsilon = 1
        action = agent.select_action(state, legal_actions)
        assert action in legal_actions
    
    def test_update_and_train(self, agent):
        """Test experience storage and training."""
        # Create fake experience
        state = np.random.random(8)
        action = (2, 3)
        reward = 1.0
        next_state = np.random.random(8)
        done = True
        
        # Store experience
        agent.update(state, action, reward, next_state, done)
        assert len(agent.replay_buffer) == 1
        
        # Test training with insufficient data
        loss = agent.train()
        assert loss is None  # Should not train with single sample
        
        # Add more experiences
        for _ in range(agent.batch_size):
            agent.update(state, action, reward, next_state, done)
        
        # Now should be able to train
        loss = agent.train()
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_epsilon_decay(self, agent):
        """Test epsilon-greedy exploration decay."""
        initial_epsilon = agent.epsilon
        
        # Train multiple times
        state = np.random.random(8)
        action = (2, 3)
        reward = 1.0
        next_state = np.random.random(8)
        done = True
        
        for _ in range(agent.batch_size):
            agent.update(state, action, reward, next_state, done)
        
        agent.train()
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_end
    
    def test_save_load(self, agent, tmp_path):
        """Test model saving and loading."""
        # Save model
        save_path = tmp_path / "test_model.pt"
        agent.save(str(save_path))
        assert save_path.exists()
        
        # Modify weights
        original_weights = [p.clone() for p in agent.policy_net.parameters()]
        for p in agent.policy_net.parameters():
            p.data += torch.randn_like(p)
        
        # Load model
        agent.load(str(save_path))
        
        # Verify weights are restored
        for original, loaded in zip(original_weights, 
                                  agent.policy_net.parameters()):
            assert torch.equal(original, loaded)
    
    def test_target_network_update(self, agent):
        """Test target network update mechanism."""
        # Get initial target network weights
        initial_weights = [p.clone() for p in agent.target_net.parameters()]
        
        # Train enough times to trigger target update
        state = np.random.random(8)
        action = (2, 3)
        reward = 1.0
        next_state = np.random.random(8)
        done = True
        
        for _ in range(agent.batch_size):
            agent.update(state, action, reward, next_state, done)
        
        # Train for target_update_freq steps
        for _ in range(agent.target_update_freq):
            agent.train()
        
        # Verify target network was updated
        for initial, current in zip(initial_weights,
                                  agent.target_net.parameters()):
            assert not torch.equal(initial, current)
            