import numpy as np
from typing import Dict, List, Tuple, Optional

class LiarsDiceEnv:
    """
    Environment for the Liar's Dice game.
    Handles game state, rules, and transitions.
    """
    def __init__(self, num_dice: int = 5, num_faces: int = 6):
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.max_quantity = num_dice * 2  # Maximum possible quantity (all dice)
        self.reset()
        
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the game state and return initial observation."""
        # Roll dice for both players
        self.dice = [
            np.random.randint(1, self.num_faces + 1, self.num_dice),
            np.random.randint(1, self.num_faces + 1, self.num_dice)
        ]
        self.current_bid = (0, 0)  # (quantity, face_value)
        self.current_player = 0
        
        # Create observation for current player
        obs = self._get_observation()
        info = {'legal_actions': self._get_legal_actions()}
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Return the current game state as observation."""
        # One-hot encode dice counts for each face value
        player_dice = self.dice[self.current_player]
        dice_counts = np.zeros(self.num_faces)
        for i in range(self.num_faces):
            dice_counts[i] = np.sum(player_dice == i + 1)
        
        # Add current bid to observation
        if self.current_bid == (0, 0):
            current_bid = np.zeros(2)
        else:
            current_bid = np.array([
                self.current_bid[0] / (2 * self.num_dice),  # Normalize quantity
                self.current_bid[1] / self.num_faces  # Normalize face value
            ])
        
        return np.concatenate([dice_counts, current_bid])
    
    def _get_legal_actions(self) -> List[Tuple[int, int]]:
        """Return list of legal actions (quantity, face_value)."""
        actions = []
        
        # If no previous bid, all bids are legal except challenge
        if self.current_bid == (0, 0):
            for q in range(1, self.max_quantity + 1):
                for f in range(1, self.num_faces + 1):
                    actions.append((q, f))
            return actions
        
        # Can always challenge after a bid
        actions.append((-1, -1))
        
        # Add all valid higher bids
        curr_q, curr_f = self.current_bid
        for q in range(1, self.max_quantity + 1):
            for f in range(1, self.num_faces + 1):
                # A bid is higher if:
                # 1. Quantity is higher
                # 2. Same quantity but higher face value
                if (q > curr_q) or (q == curr_q and f > curr_f):
                    actions.append((q, f))
        
        return actions
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a game step with the given action.
        Returns: (observation, reward, done, info)
        """
        if action not in self._get_legal_actions():
            raise ValueError(f"Illegal action: {action}")
        
        if action == (-1, -1):  # Challenge
            return self._handle_challenge()
        
        # Make new bid
        self.current_bid = action
        self.current_player = 1 - self.current_player
        
        obs = self._get_observation()
        info = {'legal_actions': self._get_legal_actions()}
        
        return obs, 0.0, False, info
    
    def _handle_challenge(self) -> Tuple[np.ndarray, float, bool, Dict]:
        """Handle challenge action and determine winner."""
        total_count = 0
        for dice_set in self.dice:
            total_count += np.sum(dice_set == self.current_bid[1])
        
        # Challenge successful if actual count is less than bid
        challenge_success = total_count < self.current_bid[0]
        
        # Reward is 1 for winner, -1 for loser
        # If current player is challenging:
        # - challenge_success means they win (reward = 1)
        # - challenge_failure means they lose (reward = -1)
        reward = 1.0 if challenge_success else -1.0
        
        return self._get_observation(), reward, True, {
            'legal_actions': [],  # Game is over, no legal actions
            'dice': self.dice.copy(),
            'total_count': total_count,
            'bid': self.current_bid,
            'challenge_success': challenge_success
        }
    
    def render(self) -> None:
        """Print current game state."""
        print(f"\nPlayer {self.current_player + 1}'s turn")
        print(f"Your dice: {self.dice[self.current_player]}")
        if self.current_bid != (0, 0):
            print(f"Current bid: {self.current_bid[0]} {self.current_bid[1]}s")