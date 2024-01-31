import tkinter as tk
from tkinter import messagebox
import time
from env import TicTacToeEnv, PLAYER_X, PLAYER_O, EMPTY
from agents import DQNAgent
import numpy as np


# Create buttons for the 3x3 grid
buttons = [[None, None, None],
        [None, None, None],
        [None, None, None]]

def update_gui():
    for i in range(3):
        for j in range(3):
            if env.board[i][j] == PLAYER_X:
                buttons[i][j].config(text="X", state=tk.DISABLED)
            elif env.board[i][j] == PLAYER_O:
                buttons[i][j].config(text="O", state=tk.DISABLED)
            else:
                buttons[i][j].config(text="", state=tk.NORMAL)
                
def convert_state(board):
    return [1 if cell == PLAYER_X else -1 if cell == PLAYER_O else 0 for row in board for cell in row]

        
current_episode = 0
current_state = None
game_over = True

def play_game():
    global current_episode, current_state, game_over
    
    if current_episode >= num_episodes:
        print("Reached the end of episodes.")
        return  # Stop if the number of episodes is reached

    if game_over:
        print(f"--- Episode {current_episode} ---")
        game_over = False
        current_state = env.reset()
        current_state = convert_state(current_state)
        current_state = np.array(current_state).reshape(1, -1)
        
        if current_episode % 10 == 0:
            agent1.current_player = PLAYER_X
            agent2.current_player = PLAYER_O
        else:
            agent1.current_player = PLAYER_O
            agent2.current_player = PLAYER_X

        print('Episode:', current_episode, 'Agent1:', agent1.current_player, 'Agent2:', agent2.current_player)
        
        current_episode += 1
    else:
        print('Current:', env.current_player)
        action = agent1.act(current_state) if env.current_player == agent1.current_player else agent2.act(current_state)
        row, col = action // 3, action % 3
        action = (row, col)
        
        next_state, reward, done = env.step(action)
        next_state = convert_state(next_state)
        next_state = np.array(next_state).reshape(1, -1)

        if env.current_player == agent1.current_player:
            agent1.learn(current_state, action, reward, next_state, done)
        else:
            agent2.learn(current_state, action, reward, next_state, done)
        
        print('Episode:', current_episode-1, 'Action:', action, 'Reward:', reward, 'Done:', done)
        current_state = next_state
        game_over = done
    
    update_gui()
    
    if game_over:
        winner = env.check_winner()
        if(agent1.current_player == PLAYER_X):
            if winner == PLAYER_X:
                agent1.wins += 1
                agent2.losses += 1
            elif winner == PLAYER_O:
                agent1.losses += 1
                agent2.wins += 1
            else:
                agent1.ties += 1
                agent2.ties += 1
        else:
            if winner == PLAYER_X:
                agent1.losses += 1
                agent2.wins += 1
            elif winner == PLAYER_O:
                agent1.wins += 1
                agent2.losses += 1
            else:
                agent1.ties += 1
                agent2.ties += 1
        
        if(agent1.current_player == PLAYER_X):
            if(winner == PLAYER_X):
                print('Agent1 wins')
            elif(winner == PLAYER_O):
                print('Agent2 wins')
            else:
                print('Tie')
        else:
            if(winner == PLAYER_X):
                print('Agent2 wins')
            elif(winner == PLAYER_O):
                print('Agent1 wins')
            else:
                print('Tie')
                    
    window.after(500, play_game)  # Schedule next call


import pickle
def save_agent(agent, filename):
    # Move model to CPU for compatibility
    agent.model.to('cpu')
    with open(filename, 'wb') as file:
        pickle.dump(agent, file, protocol=pickle.HIGHEST_PROTOCOL)
    # If you want, move model back to the appropriate device after saving
    agent.model.to(agent.device)

def load_agent(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
    
def initWindow():
    for i in range(3):
        for j in range(3):
            buttons[i][j] = tk.Button(window, text="", width=10, height=3)
            buttons[i][j].grid(row=i, column=j)
            

if __name__ == "__main__":
    
    num_episodes = 1000
    state_size = 9
    action_size = 9
    current_episode = 0
    current_state = None
    game_over = True
    
    window = tk.Tk()
    window.title("Tic Tac Toe")
    initWindow()
    
    env = TicTacToeEnv()
    agent1 = DQNAgent(state_size, action_size, hidden_dim=16, hidden_dim2=32)
    agent2 = DQNAgent(state_size, action_size, hidden_dim=256, hidden_dim2=512)
    play_game()
    print('Agent1:',agent1.printStats())
    print('Agent2:',agent2.printStats())
    
    window.after(500, play_game)  # Start the game loop
    window.mainloop()
    
    save_agent(agent1, 'agent1.pkl')
    save_agent(agent2, 'agent2.pkl')
    print('Agents saved successfully.')


