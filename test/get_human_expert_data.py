import time
import unittest
import sys
import numpy as np
import pandas as pd

from ludopy import make_img_of_board
from ludopy.visualizer import draw_basic_board
from DQN_plays import get_game_state, get_state_after_action_g, get_reshaped_ann_input
from rewards import get_reward
import config

sys.path.append("../")

from matplotlib import pyplot as plt
import datetime

from pydub import AudioSegment
from pydub.playback import play
# Input an existing mp3 filename
mp3File = 'human_data/Bruh-Sound-Effect.mp3'
# load the file into pydub
music = AudioSegment.from_mp3(mp3File)

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
    plt.tight_layout()
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

    expert_data_l = []

    g = ludopy.Game()
    there_is_a_winner = False
    show_start_board()
    prCyan("Start of the new game!")

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        """ let computer players do their actions """
        if player_i not in ai_agents:
            if len(move_pieces) > 0:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
            else:
                piece_to_move = -1

        else:
            """ select an action of AI player """
            if len(move_pieces):
                if len(move_pieces) == 1:
                    piece_to_move = move_pieces[0]
                else:
                    begin_state = get_game_state(g.get_pieces()[player_i])
                    show_board(g)  # show state of the map
                    # ask for action
                    play(music)
                    prGreen("<DICE=%d>please choose an action to take\tavailable_actions(pieces): %s" % (dice, move_pieces))
                    piece_to_move = choose_the_action(move_pieces)
                    new_state = get_state_after_action_g(g, piece_to_move)
                    pieces_player_begin = g.get_pieces()[player_i][player_i]

                    round_info = {'round': g.round,
                                  'dice': dice,
                                  'pieces_player_begin': pieces_player_begin,
                                  'available_actions': move_pieces,
                                  'state_begin': begin_state,
                                  'action': piece_to_move,
                                  'state_new': new_state,
                                  'ann_input': get_reshaped_ann_input(begin_state, new_state, piece_to_move),
                                  }
                    expert_data_l.append(round_info)
                    """ have to run this function:  
                    loss_avg = optimize_model(dice=dice, pieces_player_begin=pieces_player_begin, batch=batch,
                                              target_net=target_net, available_actions=move_pieces)
                    """

                    reward, _ = get_reward(begin_state, piece_to_move, new_state, g.get_pieces()[player_i][player_i],
                                                         actual_action=True)  # immediate reward
                    print("piece_to_move = %d | reward = %f " % (piece_to_move, reward))

            else:
                piece_to_move = -1

        """ perform action and end round """
        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        show_board(g)
        # there_is_a_winner = True  # FOR CHECKING THE SAVING

        # exit("end of round!")
        if there_is_a_winner:
            play(music)
            play(music)
            play(music)
            prCyan("GAME IS DONE")
            if player_i == 0:
                player_0_won = True
                prCyan("YOU WON!")

    now = datetime.datetime.now()
    print(now.year, now.month, now.day, now.hour, now.minute, now.second)
    print("Saving moves and states to csv")
    df_game = pd.DataFrame(expert_data_l)
    df_game.to_csv('human_data/game_' + str(now.day) +'_'+ str(now.hour) +'_'+ str(now.minute) +'_'+ '.csv')

    print("Saving history to numpy file")
    g.save_hist("game_history.npy")
    print("Saving game video")
    g.save_hist_video("randomwalk_game_video.mp4")

    return True


if __name__ == '__main__':
    get_expert_data()

