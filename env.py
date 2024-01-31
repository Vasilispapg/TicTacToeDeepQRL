# Constants for the game
PLAYER_X = "X"
PLAYER_O = "O"
EMPTY = None

class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [[EMPTY, EMPTY, EMPTY],
                      [EMPTY, EMPTY, EMPTY],
                      [EMPTY, EMPTY, EMPTY]]
        self.current_player = PLAYER_X
        self.game_over = False
        return self.board

    def step(self, action):
        row, col = action
        if self.board[row][col] == EMPTY and not self.game_over:
            self.board[row][col] = self.current_player
            winner = self.check_winner()
            if winner:
                self.game_over = True
                reward = 2 if winner == self.current_player else -2
                return self.board, reward, self.game_over
            elif not any(EMPTY in row for row in self.board):  # Draw
                self.game_over = True
                return self.board, 0, self.game_over
            else:
                self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
                return self.board, 0, self.game_over
        else:
            return self.board, -10, self.game_over  # Penalize for invalid move

    def _get_state(self):
        return [1 if cell == PLAYER_X else -1 if cell == PLAYER_O else 0 for row in self.board for cell in row]

    def check_winner(self):
        board = self.board
        # Check rows
        for row in board:
            if row[0] == row[1] == row[2] != EMPTY:
                return row[0]
        
        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != EMPTY:
                return board[0][col]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != EMPTY:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != EMPTY:
            return board[2][0]
        
        return None
