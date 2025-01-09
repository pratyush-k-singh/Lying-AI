import argparse
import os
from typing import Tuple, List
import numpy as np

from src.env.liars_dice_env import LiarsDiceEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Play Liar\'s Dice against trained AI')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    return parser.parse_args()

def format_action(action: Tuple[int, int]) -> str:
    """Format action tuple into human-readable string."""
    if action == (-1, -1):
        return "Challenge (Liar!)"
    quantity, face = action
    return f"{quantity} {face}{'s' if quantity > 1 else ''}"

def get_human_action(legal_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Get action from human player."""
    while True:
        print("\nLegal actions:")
        for i, action in enumerate(legal_actions):
            print(f"{i}: {format_action(action)}")
        
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 0 <= choice < len(legal_actions):
                return legal_actions[choice]
        except ValueError:
            pass
        print("Invalid choice, please try again.")

def play_game(env: LiarsDiceEnv, agent: DQNAgent, human_first: bool = True):
    """Play a single game against the AI."""
    state, info = env.reset()
    done = False
    
    print("\nGame started!")
    print(f"Your dice: {env.dice[1 if human_first else 0]}")
    
    current_player = 0  # 0 for first player, 1 for second
    
    while not done:
        is_human_turn = (current_player == 0) if human_first else (current_player == 1)
        
        if is_human_turn:
            print("\nYour turn!")
            action = get_human_action(info['legal_actions'])
            print(f"\nYou chose: {format_action(action)}")
        else:
            print("\nAI's turn...")
            action = agent.select_action(state, info['legal_actions'])
            print(f"AI chose: {format_action(action)}")
        
        state, reward, done, info = env.step(action)
        
        if done:
            # Game ended with a challenge
            all_dice = env.dice[0] + env.dice[1]  # Combine dice from both players
            last_bid = env.current_bid
            actual_count = sum(1 for die in all_dice if die == last_bid[1])
            
            print("\nGame Over!")
            print(f"All dice: {all_dice}")
            print(f"Last bid: {format_action(last_bid)}")
            print(f"Actual count of {last_bid[1]}s: {actual_count}")
            
            challenger_won = actual_count < last_bid[0]
            if (is_human_turn and challenger_won) or (not is_human_turn and not challenger_won):
                print("You win!")
            else:
                print("AI wins!")
        else:
            current_player = 1 - current_player

def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    
    # Initialize environment
    env = LiarsDiceEnv(
        num_dice=config.env.num_dice,
        num_faces=config.env.num_faces
    )
    
    # Initialize agent
    state_dim = env.num_faces + 2
    action_dim = 1 + 10 * 6
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.model.hidden_dim,
        device=config.device
    )
    
    # Load trained model
    agent.load(args.model)
    print(f"Loaded trained model from {args.model}")
    
    while True:
        # Ask for player preference
        first = input("\nDo you want to go first? (y/n): ").lower() == 'y'
        
        # Play game
        play_game(env, agent, human_first=first)
        
        # Ask to play again
        if input("\nPlay again? (y/n): ").lower() != 'y':
            break
    
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()
    