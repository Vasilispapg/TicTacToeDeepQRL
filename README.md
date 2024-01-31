# Tic-Tac-Toe Reinforcement Learning

## Introduction
This project is an implementation of a Tic-Tac-Toe game where two Deep Q-Network (DQN) agents learn to play against each other. Utilizing PyTorch for neural network model construction and Tkinter for GUI visualization, the agents improve their gameplay over time through the reinforcement learning process.

## Features
- **Reinforcement Learning Agents**: Two DQN agents that learn optimal strategies through gameplay.
- **Interactive GUI**: Built with Tkinter, allowing real-time visualization of the agents' gameplay.
- **Customizable Hyperparameters**: Flexibility in tuning learning rates, exploration rates, and neural network architecture.

## Installation
Clone the repository to your local machine: git clone git@github.com:Vasilispapg/TicTacToeDeepQRL.git

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- Required Python libraries (NumPy, PyTorch, Tkinter)

## Usage
To run the game, execute: python main.py

This starts the Tic-Tac-Toe game with the agents learning in real-time. The game progress is displayed in the GUI window.

## Code Structure
- `main.py`: Initializes the game environment, agents, and the GUI loop.
- `env.py`: Contains the TicTacToeEnv class, defining the game environment.
- `agents.py`: Defines the DQNAgent and DQN classes for the reinforcement learning agents.
- `README.md`: This documentation file.

## Usage
1. Run `main.py` to start the game.
2. The RL agents will play against each other, learning and improving over time.
3. The GUI will display the game's progress and outcomes.

## DQNAgent Configuration
You can customize the DQNAgent's behavior by modifying its constructor in `agents.py`. Parameters include:

- `state_size`: Size of the state space.
- `action_size`: Size of the action space.
- `hidden_dim`: Number of neurons in the first hidden layer.
- `hidden_dim2`: Number of neurons in the second hidden layer.
- `epsilon`: Exploration rate.
- `epsilon_min`: Minimum exploration rate.
- `epsilon_decay`: Rate of exploration decay.
- `gamma`: Discount rate.
- `batch_size`: Mini-batch size for training.
- `model`: Neural network model.
- `optimizer`: Optimization method.
- `wins`, `losses`, `ties`: Game statistics.


## Saving and Loading Models
The trained models can be saved and later loaded to resume training or for evaluation:
```python
# Saving the model
save_agent(agent1, 'agent1.pkl')
save_agent(agent2, 'agent2.pkl')

# Loading the model
agent1 = load_agent('agent1.pkl')
agent2 = load_agent('agent2.pkl')
```

## Contributions
Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.
