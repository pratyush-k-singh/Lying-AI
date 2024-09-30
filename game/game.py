import random
import torch
import os
import re
from src.env import LiarsDiceGame

def fetch_model_list(directory):
    """Retrieve a list of model files from the specified directory."""
    return [f for f in os.listdir(directory) if f.endswith('.pt')]

def load_selected_model(model_path):
    """Load the model from the specified path."""
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    training_args = checkpoint["args"]

    D_PUB, D_PRI, *_ = calc_args(
        training_args.d1, training_args.d2, training_args.sides, training_args.variant
    )
    model_instance = NetCompBilin(D_PRI, D_PUB)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    return model_instance, training_args

class Player:
    """Represents a human player in the game."""
    def get_action(self, state):
        last_call = game.get_last_call(state)
        while True:
            call_input = input('Your move [e.g. 24 for 2 fours, or "lie" to call a bluff]: ')
            if call_input.lower() == "lie":
                return game.LIE_ACTION
            elif match := re.match(r"(\d)(\d)", call_input):
                n, d = map(int, match.groups())
                action = (n - 1) * game.SIDES + (d - 1)
                if action <= last_call:
                    print(f"Invalid call! You cannot call after {format_action(last_call)}.")
                elif action >= game.LIE_ACTION:
                    print(f"The maximum call you can make is {format_action(game.LIE_ACTION - 1)}.")
                else:
                    return action

    def __repr__(self):
        return "Player"

class AI:
    """Represents the AI player in the game."""
    def __init__(self, private_info):
        self.private_info = private_info

    def get_action(self, state):
        last_call = game.get_last_call(state)
        return game.sample_action(self.private_info, state, last_call, eps=0)

    def __repr__(self):
        return "AI"

def format_action(action):
    """Format the action for display."""
    action = int(action)
    if action == -1:
        return "nothing"
    if action == game.LIE_ACTION:
        return "lie"
    n, d = divmod(action, game.SIDES)
    n, d = n + 1, d + 1
    return f"{n} {d}s"

if __name__ == "__main__":
    model_directory = "../models"  # Update to your model directory
    available_models = fetch_model_list(model_directory)

    if not available_models:
        print("No models found in the specified directory.")
        exit(1)

    print("Available Models:")
    for idx, model_name in enumerate(available_models):
        print(f"{idx + 1}: {model_name}")

    model_choice = int(input("Select a model to challenge (1-{}): ".format(len(available_models)))) - 1
    selected_model_path = os.path.join(model_directory, available_models[model_choice])

    model_instance, training_args = load_selected_model(selected_model_path)
    game = LiarsDiceGame(model_instance, training_args.d1, training_args.d2, training_args.sides, training_args.variant)

    while True:
        while (first_turn := input("Do you want to go first? [y/n/r] ")) not in ["y", "n", "r"]:
            pass

        roll1 = random.choice(list(game.rolls(0)))
        roll2 = random.choice(list(game.rolls(1)))
        privates = [game.make_priv(roll1, 0), game.make_priv(roll2, 1)]
        state = game.make_state()

        if first_turn == "y":
            print(f"> You rolled {roll1}!")
            players = [Player(), AI(privates[1])]
        elif first_turn == "n":
            print(f"> You rolled {roll2}!")
            players = [AI(privates[0]), Player()]
        elif first_turn == "r":
            players = [AI(privates[0]), AI(privates[1])]

        current_player = 0
        while True:
            action = players[current_player].get_action(state)
            print()
            print(f"> The {players[current_player]} called {format_action(action)}!")

            if action == game.LIE_ACTION:
                last_call = game.get_last_call(state)
                result = game.evaluate_call(roll1, roll2, last_call)
                print()
                print(f"> The rolls were {roll1} and {roll2}.")
                if result:
                    print(f"> The call {format_action(last_call)} was valid!")
                    print(f"> The {players[current_player]} loses!")
                else:
                    print(f"> The call {format_action(last_call)} was a bluff!")
                    print(f"> The {players[current_player]} wins!")
                print()
                break

            state = game.apply_action(state, action)
            current_player = 1 - current_player
