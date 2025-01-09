import pytest
import numpy as np
from src.env.liars_dice_env import LiarsDiceEnv

class TestLiarsDiceEnv:
    @pytest.fixture
    def env(self):
        return LiarsDiceEnv(num_dice=3, num_faces=6)
    
    def test_init(self, env):
        """Test environment initialization."""
        assert env.num_dice == 3
        assert env.num_faces == 6
        assert len(env.dice) == 2  # Two players
    
    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        
        # Check observation shape
        assert obs.shape == (8,)  # 6 faces + 2 bid values
        assert 'legal_actions' in info
        
        # Check dice rolls
        assert all(1 <= die <= 6 for die in env.dice[0])
        assert all(1 <= die <= 6 for die in env.dice[1])
        
        # Check initial state
        assert env.current_bid == (0, 0)
        assert env.current_player == 0
    
    def test_legal_actions(self, env):
        """Test legal action generation."""
        env.reset()
        
        # Initial state should allow all bids
        legal_actions = env._get_legal_actions()
        assert (-1, -1) not in legal_actions  # Can't challenge first move
        assert len(legal_actions) == 6 * env.num_dice * 2  # All possible bids
        
        # Make a bid
        env.step((2, 3))  # Bid "two threes"
        legal_actions = env._get_legal_actions()
        assert (-1, -1) in legal_actions  # Can now challenge
        
        # Verify all actions are higher than current bid
        current_bid_value = 2 * 6 + 3  # Convert bid to comparable value
        for action in legal_actions:
            if action != (-1, -1):
                quantity, face = action
                action_value = quantity * 6 + face
                assert action_value > current_bid_value
    
    def test_challenge_resolution(self, env):
        """Test challenge mechanics."""
        env.reset()
        
        # Set known dice values for testing
        env.dice = [
            np.array([3, 3, 3]),  # Player 1 has three 3s
            np.array([3, 4, 5])   # Player 2 has one 3
        ]
        
        # Make a bid
        _, _, done, _ = env.step((3, 3))  # Bid "three threes"
        assert not done
        
        # Challenge the bid
        obs, reward, done, info = env.step((-1, -1))
        assert done
        
        # Total count of threes is 4, so bid of 3 is valid
        # Challenger should lose
        assert reward < 0
    
    def test_bid_sequence(self, env):
        """Test sequence of bids."""
        env.reset()
        
        # Test sequence of valid bids
        actions = [(1, 1), (1, 2), (2, 2), (2, 3)]
        for action in actions:
            obs, reward, done, info = env.step(action)
            assert not done
            assert reward == 0
            
            # Verify current bid is updated
            assert env.current_bid == action
            # Verify player alternates
            assert env.current_player == (actions.index(action) + 1) % 2
    
    def test_invalid_actions(self, env):
        """Test handling of invalid actions."""
        env.reset()
        
        with pytest.raises(ValueError):
            env.step((0, 1))  # Invalid quantity
        
        with pytest.raises(ValueError):
            env.step((1, 7))  # Invalid face value
        
        # Make valid bid
        env.step((1, 1))
        
        with pytest.raises(ValueError):
            env.step((1, 1))  # Can't make same bid
            
        with pytest.raises(ValueError):
            env.step((1, 0))  # Lower bid not allowed
            