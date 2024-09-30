import os
import logging
from env import LiarsDiceGame
from agent import Agent

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_log_file_name(log_dir='../logs'):
    """
    Returns the next log file name in sequence (log01, log02, etc.).
    """
    create_directory(log_dir)
    existing_logs = sorted([f for f in os.listdir(log_dir) if f.startswith('log') and f.endswith('.log')])
    next_index = len(existing_logs) + 1
    return os.path.join(log_dir, f"log{next_index:02d}.log")

def get_model_file_name(model_dir='../models', label=None):
    """
    Returns the model file name in the format model_{label}.model, and reprompts if the file exists.
    """
    create_directory(model_dir)
    
    while True:
        if not label:
            label = input("Enter a label for the model: ")

        model_file = os.path.join(model_dir, f"model_{label}.model")
        if os.path.isfile(model_file):
            print(f"Model with label '{label}' already exists. Please choose a different label.")
            label = None  # Reset label to reprompt
        else:
            return model_file

def setup_logging(log_file):
    """
    Sets up logging to a specific file.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

class Trainer:
    def __init__(self, num_players=2, num_dice=5, num_episodes=1000):
        self.num_players = num_players
        self.num_dice = num_dice
        self.num_episodes = num_episodes
        self.agents = [Agent(num_dice) for _ in range(self.num_players)]
        self.game = LiarsDiceGame(num_players, num_dice)
        self.wins = {i: 0 for i in range(num_players)}

    def train(self, load_model_path=None):
        # Load model if needed
        if load_model_path:
            for i in range(self.num_players):
                self.agents[i] = self.agents[i].load_model(load_model_path) or self.agents[i]

        for episode in range(self.num_episodes):
            print(f"Episode {episode + 1}/{self.num_episodes}")
            self.game.reset()
            done = False
            round_result = None

            while not done:
                for player in range(self.num_players):
                    game_state = self.game.get_player_dice(player)
                    bid = self.agents[player].make_bid(game_state)
                    print(f"Player {player} chooses to bid {bid}")
                    self.game.make_bid(player, bid)

                    if self.agents[(player + 1) % self.num_players].decide_challenge(game_state):
                        print(f"Player {(player + 1) % self.num_players} challenges!")
                        success = self.game.challenge(player)
                        done = True
                        if success:
                            round_result = (player + 1) % self.num_players
                            self.wins[round_result] += 1
                            print(f"Player {(player + 1) % self.num_players} wins the challenge!")
                        else:
                            round_result = player
                            self.wins[round_result] += 1
                            print(f"Player {player} wins the challenge!")
                        break

            # Train agent based on outcome (reward mechanism)
            for player in range(self.num_players):
                reward = 1 if player == round_result else -1
                self.agents[player].update(game_state, reward)

        logging.info(f"Final wins: {self.wins}")

    def save_model(self):
        model_file = get_model_file_name()
        for i, agent in enumerate(self.agents):
            agent_save_path = f"{model_file}_player_{i}.model"
            agent.save_model(agent_save_path)
            logging.info(f"Model saved for Player {i} at {agent_save_path}")
