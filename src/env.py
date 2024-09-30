import random

class LiarsDiceGame:
    def __init__(self, num_players=2, num_dice=5):
        self.num_players = num_players
        self.num_dice = {i: num_dice for i in range(num_players)}  # dice per player
        self.dice = {i: [] for i in range(self.num_players)}  # dice rolls per player
        self.current_bids = []  # Track bids
        self.current_player = 0

    def roll_dice(self):
        for player in range(self.num_players):
            self.dice[player] = [random.randint(1, 6) for _ in range(self.num_dice[player])]

    def get_player_dice(self, player):
        return self.dice[player]

    def make_bid(self, player, quantity, face_value):
        bid = (quantity, face_value)
        self.current_bids.append(bid)
        self.current_player = (self.current_player + 1) % self.num_players

    def challenge(self, player):
        last_bid = self.current_bids[-1]
        total_face_value = sum(die == last_bid[1] for dice in self.dice.values() for die in dice)
        return total_face_value < last_bid[0]

    def update_dice_after_challenge(self, challenger_wins, challenged_player):
        if challenger_wins:
            self.num_dice[challenged_player] -= 1

    def reset(self):
        self.current_bids = []
        self.current_player = 0
        self.roll_dice()
