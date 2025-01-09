# Liar's Dice AI

A deep reinforcement learning agent for the classic game Liar's Dice, achieving an 84% win rate through stable DQN training and reward scaling mechanics.

## Technical Highlights

• Implemented DQN-based AI for Liar's Dice with dual policy/value heads, achieving 84% win rate vs random agents
• Built vectorized game environment with legal action masking and reward scaling for stable RL training
• Developed multi-stage training pipeline with interim model checkpoints and TensorBoard visualization

## Game Rules

Liar's Dice is a strategic dice game of deception and probability. Each player:
- Rolls a hidden set of dice
- Takes turns making increasingly higher bids about the total dice in play
- Can either make a higher bid or challenge the previous bid as a lie

A bid consists of:
- A quantity (e.g., "three")
- A face value (e.g., "fours")

The bid claims how many dice showing that face value are present across all dice in play.

## Project Structure

```
liars_dice_ai/
├── configs/             # Training configuration
│   └── default.yaml    
├── src/
│   ├── env/            # Game environment
│   ├── agents/         # DQN implementation
│   ├── models/         # Neural networks
│   ├── training/       # Training utilities
│   └── utils/          # Helper functions
├── scripts/
│   ├── train.py        # Training script
│   └── play.py         # Human vs AI interface
└── models/             # Saved checkpoints
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pratyushsingh97/liars_dice_ai.git
cd liars_dice_ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

## Usage

### Training New Models
```bash
# Start training
python scripts/train.py --config configs/default.yaml

# Monitor progress
tensorboard --logdir logs/
```

Training checkpoints are saved automatically:
- Best model: `models/default_best.pt`
- Regular checkpoints: `models/default_episode_{N}.pt`

To preserve previous results, modify `exp_name` in config:
```yaml
exp_name: "experiment_v2"  # Creates new log/model directories
```

### Playing Against AI
```bash
# Play against best model
python scripts/play.py --model models/default_best.pt

# Play specific checkpoint
python scripts/play.py --model models/default_episode_1700.pt
```

## Training Performance

Key milestones from our best training run:
- Episode 500: 74% win rate
- Episode 1000: 82% win rate
- Episode 1700: 84% win rate (peak performance)
- Episodes 2000+: Stable performance

## Implementation Details

The DQN implementation features:
- Dual policy and value heads
- Experience replay buffer
- Target network updates
- Reward scaling and gradient clipping
- TensorBoard metrics tracking
- Model checkpointing

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.21.0
- TensorBoard ≥ 2.7.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI's Spinning Up for RL implementation references
- DeepMind's DQN papers for architecture inspiration
