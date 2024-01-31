import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, hidden_dim2=256):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size,hidden_dim=128, hidden_dim2=256):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = []  # Used for experience replay
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size, hidden_dim, hidden_dim2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.batch_size = 32
        self.wins = 0
        self.losses = 0
        self.ties = 0
        
    def printStats(self):
        print("Wins:", self.wins,end=' ')
        print("Losses:", self.losses,end=' ')
        print("Ties:", self.ties)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Ensure random actions are legal (not on occupied cells)
            return random.choice([i for i in range(self.action_size) if state[0][i] == 0])
        state = torch.FloatTensor(state).to(self.device)  # Convert state to tensor
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()  # Set the model back to train mode
        # Mask the act_values where the cells are occupied (state != 0)
        masked_act_values = np.where(state.cpu().numpy()[0] != 0, -np.inf, act_values.cpu().numpy())
        return np.argmax(masked_act_values[0])


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = (reward + self.gamma * np.amax(self.model(next_state).detach().numpy()))
            state = torch.FloatTensor(state)
            target_f = self.model(state)
            action_index = action[0] * 3 + action[1]  # Convert (row, col) action to single index
            target_f[0][action_index] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def learn(self, state, action, reward, next_state, done):
        # Store the experience in memory
        self.remember(state, action, reward, next_state, done)

        # Experience replay if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)

        # Reduce epsilon to decrease the exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
