import pdb
import random
import neuralnetwork as neural

class Board:
    # Board cells initialized
    cells=[3, 3, 3, 3, 3, 3, 3, 3, 3]

    # Method to print the board.
    def BoardPrint(self, plA, plB):
        boardCopy=[3, 3, 3, 3, 3, 3, 3, 3, 3]  # Copy to avoid to change the original
        # This part changes the value of player number (1 or 2)
        # with the marks chosen e.g. "X", "O"
        for i in range(0,9):
            if self.cells[i] == plA.number:
                boardCopy[i] = plA.mark
            elif self.cells[i] == plB.number:
                boardCopy[i] = plB.mark
            else:
                boardCopy[i] = " "
        # Print the board with TICTACTOE format
        print("|{}|{}|{}|".format(boardCopy[0],boardCopy[1],boardCopy[2]))
        print("-------")
        print("|{}|{}|{}|".format(boardCopy[3],boardCopy[4],boardCopy[5]))
        print("-------")
        print("|{}|{}|{}|".format(boardCopy[6],boardCopy[7],boardCopy[8]))

    def returnWinner(self):
        winner = 0
        for i in range(3):
            if self.cells[3*i] == self.cells[3*i+1] and self.cells[3*i+1] == self.cells[3*i+2] and self.cells[3*i] != 3:
                winner = self.cells[3*i]
            if self.cells[i] == self.cells[3+i] and self.cells[3+i] == self.cells[6+i] and self.cells[i] != 3:
                winner = self.cells[i]

        if self.cells[0] == self.cells[4] and self.cells[4] == self.cells[8] and self.cells[0] != 3:
                winner = self.cells[4]

        if self.cells[2] == self.cells[4] and self.cells[4] == self.cells[6] and self.cells[2] != 3:
                winner = self.cells[4]
        return winner

    def invertBoard(board):
        for i, hist in enumerate(board):
            for j, move in enumerate(hist[0]):
                if move == 1:
                    board[i][0][j] = 2
                elif move == 2:
                    board[i][0][j] = 1
        return board

    def formatBoard(board, turn):
        if turn == 1:
            board = Board.invertBoard(board)
        for i, hist in enumerate(board):
            for j, move in enumerate(hist[0]):
                if move == 1:
                    board[i][0][j] = 1
                elif move == 2:
                    board[i][0][j] = 2
        return board

    def isSpaceFree(board, move):
        # Return true if the passed move is free on the passed board.
        return board[move] == 3

    def RandomMoveFromList(board, movesList):
        # Returns a valid move from the passed list on the passed board.
        # Returns None if there is no valid move.
        possibleMoves = []
        for i in movesList:
            if Board.isSpaceFree(board, i):
                possibleMoves.append(i)

        if len(possibleMoves) != 0:
            return random.choice(possibleMoves)
        else:
            return None

class Player:
    mark = 'X' # Mark to be printed in the BoardPrint method
    number = 1 # Player number 1 or 2

    def __init__(self, m, n):
        self.mark = m
        self.number = n

    def move(self,boardA, player_move):
        if player_move < 1 or player_move > 9 or boardA[player_move-1] != 3:
            #print("Wrong move try again")
            return False #It isn't valid move out of range or occupied cell
        else:
            boardA[player_move-1] = self.number
            return True

class MachinePlayer(Player):

    def __init__(self ,m, n):
        super().__init__(m,n)

    def get_move(board, neuralnet, turn):
        boardArray = Board.formatBoard([[board[:], 0]], turn)[0][0]
        player_move = neuralnet(boardArray)
        player_move = player_move.detach().numpy().tolist()
        #print(player_move)
        return player_move.index(max(player_move))+1

    # Training when machine does a forbiden movement.
    # Trained to move in a random free cell
    def new_training(board, neuralnet, turn):
        move = Board.RandomMoveFromList(board,[0,1,2,3,4,5,6,7,8])
        boardArray = Board.formatBoard([[board[:], 0]], turn)[0][0]
        player_move = neuralnet(boardArray)
        player_move = player_move.detach().numpy().tolist()
        player_move[player_move.index(max(player_move))] = 0.1
        player_move[move] = 0.5
        neural.trainingNet([(boardArray,player_move)], neuralnet)
        #print(move)
        return move + 1

    # Training when human player win
    # Trained to replicate human movements
    def winner_training(history, net, winner):
        history = Board.formatBoard(history[:], winner)
        if winner != 0:
            neural.trainingNet(history, net)
