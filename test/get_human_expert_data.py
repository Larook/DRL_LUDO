import time
import unittest
import sys
import numpy as np

from ludopy import make_img_of_board
from ludopy.visualizer import draw_basic_board

sys.path.append("../")

from matplotlib import pyplot as plt


def prRed(skk): print("\033[91m {}\033[00m".format(skk))
def prGreen(skk): print("\033[92m {}\033[00m".format(skk))
def prYellow(skk): print("\033[93m {}\033[00m".format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m".format(skk))
def prPurple(skk): print("\033[95m {}\033[00m".format(skk))
def prCyan(skk): print("\033[96m {}\033[00m".format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m".format(skk))
def prBlack(skk): print("\033[98m {}\033[00m".format(skk))

def show_start_board():
    board_img = draw_basic_board()
    plt.imshow(board_img, interpolation='nearest')
    plt.draw()
    plt.pause(0.005)
    time.sleep(0.01)

def show_board(g):
    board_img = make_img_of_board(*g.hist[-1])
    plt.imshow(board_img, interpolation='nearest')
    plt.draw()
    plt.pause(0.005)


def choose_the_action(move_pieces):
    chosen_correct = False
    while not chosen_correct:
        piece_to_move = input("Some input please: ")
        if piece_to_move.isdigit():
            chosen_correct = int(piece_to_move) in move_pieces
            piece_to_move = int(piece_to_move)
    return piece_to_move

def get_expert_data():
    import ludopy
    import numpy as np

    player_0_won = False
    turns_passed = 0
    ai_agents = [0]  # which id of player should be played by ai?

    while not player_0_won:
        g = ludopy.Game()
        there_is_a_winner = False
        show_start_board()
        prCyan("Start of the new game!")

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = g.get_observation()

            # show state of the map
            show_board(g)

            """ let computer players do their actions """
            if player_i not in ai_agents:
                if len(move_pieces) > 0:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
                else:
                    piece_to_move = -1

            else:
                # show_board(g)

                """ select an action of AI player """
                if len(move_pieces):
                    if len(move_pieces) == 1:
                        piece_to_move = move_pieces[0]
                    else:
                        # ask for action
                        prRed("<DICE=%d>please choose an action to take\tavailable_actions(pieces): %s" % (dice, move_pieces))
                        piece_to_move = choose_the_action(move_pieces)
                        print("piece_to_move = ", piece_to_move)

                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                else:
                    piece_to_move = -1

            """ perform action and end round """
            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
            # g.__add_to_hist()
            show_board(g)

            # exit("end of round!")
            if there_is_a_winner and player_i == 0:
                player_0_won = True
            # turns_passed += 1
            # print("turns_passed = ", turns_passed)
    print("Saving history to numpy file")
    g.save_hist("game_history.npy")
    print("Saving game video")
    g.save_hist_video("randomwalk_game_video.mp4")

    return True


if __name__ == '__main__':
    get_expert_data()

