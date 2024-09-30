import torch
from torch import nn
import random
import os

# Agent class
class Agent:
    def __init__(self, model_path='model.pt', d1=5, d2=5, sides=6, variant='normal'):
        self.d1 = d1
        self.d2 = d2
        self.sides = sides
        self.variant = variant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print(f"No pretrained model found at {model_path}, training new model")
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)

    def _build_model(self):
        class Net(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(5, 100)
                self.fc2 = nn.Linear(100, 100)
                self.fc3 = nn.Linear(100, output_dim)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)

        input_dim = self.d1
        output_dim = self.sides
        return Net(input_dim, output_dim)

    def make_bid(self, game_state):
        with torch.no_grad():
            game_state_tensor = torch.tensor(game_state, dtype=torch.float).to(self.device)
            bid = self.model(game_state_tensor).argmax().item()
        return bid

    def decide_challenge(self, game_state):
        # Model-based decision logic for challenges
        with torch.no_grad():
            game_state_tensor = torch.tensor(game_state, dtype=torch.float).to(self.device)
            challenge_prob = self.model(game_state_tensor)
        return challenge_prob > 0.5

    def train(self, replay_buffer):
        self.model.train()
        loss_fn = torch.nn.MSELoss()
        for state, reward in replay_buffer:
            self.optimizer.zero_grad()
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            reward_tensor = torch.tensor([reward], dtype=torch.float).to(self.device)
            predicted_reward = self.model(state_tensor)
            loss = loss_fn(predicted_reward, reward_tensor)
            loss.backward()
            self.optimizer.step()

    def load_model(self, path='model.pt'):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save_model(self, path='model.pt'):
        torch.save({'model_state_dict': self.model.state_dict()}, path)

