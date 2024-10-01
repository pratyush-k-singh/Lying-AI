# Liars Dice AI Development

This project is a single-player Liars Dice game designed with a focus on AI development. The AI is built using PyTorch, and the game allows you to train the AI through self-play, or optionally, play against the AI yourself. The core goal is to develop, test, and improve the AI's decision-making strategies in this classic dice game.

## Features
- **AI Training**: Train the AI using reinforcement learning by running multiple development cycles (`num_episodes`).
- **Single-Player Mode**: Play against the trained AI to test its behavior in real gameplay.
- Multiple AI Opponents: Play against multiple trained AI models simultaneously.
- **Customizable Game Environment**: Modify the number of dice, players, and sides to suit different game variants.

## Project Structure

```plaintext
liars-dice-ai/
├── game/
│   ├── game.py        # Contains the code to play against a trained model
├── src/
│   ├── agent.py       # Contains the AI agent logic
│   ├── env.py         # Game environment and mechanics
│   ├── trainer.py     # Training logic for AI agents
│   ├── main.py        # Entry point for training the AI
│
├── logs/              # Log files are stored here after each training session
├── models/            # Directory for saving trained models
├── requirements.txt   # Python dependencies
```

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running AI Training

You can train the AI by running the `main.py` script. The AI can be trained from scratch or by loading a previously trained model.

```bash
python src/main.py
```

Upon starting the training, you will be prompted to:
- Specify the number of development cycles (episodes) for training.
- Choose whether to load a previous model or start training a new AI from scratch.

### Playing Against the AI

Although the primary focus of this project is AI development, you can also play against the AI by running the game.py script, although the model itself does need to be run to train the AI first.

### Training Workflow

- **Agent**: The AI agent is implemented using a neural network model built with PyTorch (`agent.py`).
- **Trainer**: The `Trainer` class in `trainer.py` is responsible for running the training loop and managing the game environment.
- **Game Environment**: The core game logic and rules are in `env.py`, handling dice rolls, bids, challenges, and win/loss conditions.

### Saving and Loading Models

The AI models are saved automatically after training in the `models/` directory. The models are named sequentially, and you can load them during training by specifying the model file.

### Logging

Logs of each training session, including episode results and AI decisions, are saved in the `logs/` directory.

## Customization

You can customize the following aspects of the game and AI:
- **Number of Dice/Players**: Change the number of dice or players by modifying the `LiarsDiceGame` initialization in `trainer.py`.
- **AI Model**: Modify the AI architecture in the `Agent` class in `agent.py` to test different strategies and neural network structures.
- **Game Variant**: The `variant` parameter in the `Agent` class allows you to explore different game rules.

## Example Training Session

Here's an example of what running a training session looks like:

```bash
$ python src/main.py
Enter number of development cycles: 1000
Load from a previous model? (Y/N): n
Logging to ../logs/log01.log
Episode 1/1000
Player 0 chooses to bid (3, 2)
Player 1 challenges!
Player 1 wins the challenge!
...
Training completed. Model saved for Player 0 at ../models/model_player_0.model
Model saved for Player 1 at ../models/model_player_1.model
```

## Future Improvements

- **Advanced AI**: Implement more sophisticated learning algorithms or neural network architectures to enhance AI performance.
- **Evaluation Metrics**: Integrate evaluation metrics to better analyze AI performance during training.
- **Expanded Game Modes**: Develop additional game modes or variants for further AI experimentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
