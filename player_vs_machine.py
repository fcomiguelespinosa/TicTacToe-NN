import pdb
import random
import neuralnetwork as neural
import tictactoe as ttt

# Instances Player 1, Player 2 and Board
playerA = ttt.Player("X",1)
playerB = ttt.MachinePlayer("O",2)
board = ttt.Board()
moves = 0
wrong = 0
isWinner = 0
history = []

# Random value to determinate who will start to move
player_turn = random.randint(1,2)
net = neural.returnNet()
#neural.firstTraining(net)

while moves < 9 and isWinner == 0:
    isValid = False
    print("(1-9) Where will you move? ")
    board.BoardPrint(playerA, playerB)
    boardcopy = board.cells[:]
    moveFormat = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # Bucle to retry move when player o machine does a wrong movement
    while not isValid:
        try:
            if player_turn == playerA.number:
                moveP = int(input())
                isValid = playerA.move(board.cells, moveP)
            else:
                moveP = ttt.MachinePlayer.get_move(board.cells, net, player_turn)
                isValid = playerB.move(board.cells, moveP)
                if not isValid:
                    #Machine did a forbiden movement training to correct.
                    moveP = ttt.MachinePlayer.new_training(board.cells,net, player_turn)
                    isValid = playerB.move(board.cells, moveP)
                    wrong += 1
        except:
            isValid = False

    #Switch player turn
    player_turn = player_turn % 2 +1
    isWinner = board.returnWinner()
    moveFormat[moveP - 1] = 3
    history.append([boardcopy[:] , moveFormat])
    moves += 1

print("Winner is {}".format(isWinner))
if (isWinner == 0):
    isWinner = player_turn
history = history[(moves+1)%2 ::2]
ttt.MachinePlayer.winner_training(history, net, isWinner)
board.BoardPrint(playerA, playerB)
print("Number wrong moves {}".format(wrong))
neural.saveNet(net)
