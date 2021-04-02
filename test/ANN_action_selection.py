import copy
import math
import random
import unittest
import sys
from collections import namedtuple

import numpy as np


sys.path.append("../")

from Feedforward import *
from rewards import *


def get_game_state(players):
    """
    state represented by  240 variables - for each player 60
    Each state (id of tile) can have values 0 - 1, where 0 means 0 pawns on the tile, and 1 means 4 pawns on tile

    :param pieces:
    :return:
    """
    # players = pieces[0]
    POSITIONS_PER_PLAYER = 60

    # for every player
    # loop through the pawns
    # save the positions
    # update state of the player
    state_all = np.empty([4, POSITIONS_PER_PLAYER], float)
    for index, player in enumerate(players):
        pawn_positions = player
        state = np.zeros(POSITIONS_PER_PLAYER)
        for pawn_id in pawn_positions:
            state[pawn_id] += 0.25
        state_all[index] = state

    # print("state_all", state_all)
    return state_all


def get_pawn_id_from_tile(tile_id, player_pieces):
    """
    get the id of a random pawn that is located on the tile_id
    Careful! this id of pawn might not be the movable pawn!!
    """
    # best_pawn_id = 100
    for i, pawn in enumerate(player_pieces):
        if pawn == tile_id:
            best_pawn_id = i
            return best_pawn_id


def choose_action_furthest_pawn(state, pieces, move_pieces):
    """
    Always choose the furthest pawn to move!
    :param state: from papers - length 240
    :param pieces: all of the pieces
    :param move_pieces: only movable ones
    :return: id of further pawn - if possible. If its not movable just return random movable one
    """
    player_pieces = pieces[0][0]
    player0_state = state[0]
    furthest_dist = 0
    best_pawn_id_to_move = 100

    # if all are zeros then choose random
    if player0_state[0] == 1:
        return move_pieces[np.random.randint(0, len(move_pieces))]

    # if a pawn is outside home - get his distance (id of tile that he is standing on)
    for tile_id, occupation_of_tile in enumerate(player0_state):
        # if is outside of home and the furthest so far
        if tile_id >= furthest_dist and occupation_of_tile != 0:
            furthest_tile_id = tile_id
            # get the id of the pawn
            move_pawn_id = get_pawn_id_from_tile(furthest_tile_id, player_pieces)
            # check if it is in movable pieces
            if move_pawn_id in move_pieces:
                best_pawn_id_to_move = move_pawn_id

    # if managed to find a movable piece outside home
    if best_pawn_id_to_move != 100:
        return best_pawn_id_to_move
    else:
        # if not just pick random movable piece
        return move_pieces[np.random.randint(0, len(move_pieces))]


# TODO
def get_policy_action_ann(model, state_now, action_tuple):
    """
    DQN network to select the action of optimal policy
    :param state_papers_start:
    :return:
    """
    # TODO: flatten everything to 1 tensor
    input_ann = np.array(state_now)
    input_ann = input_ann.reshape((240, 1))
    # print(input_ann.shape)
    input_ann = np.append(input_ann, action_tuple)
    # print(input_ann.shape)

    input_ann = torch.FloatTensor(input_ann)
    # input_ann = torch.FloatTensor(action_tuple)
    Q_pred = model(input_ann)
    print("Q_pred = ", Q_pred)
    # t.max(1) will return largest column value of each row.
    # second column on max result is index of where max element was
    # found, so we pick action with the larger expected reward.
    # Q_pred_max = Q_pred.max(1)[1].view(1, 1)
    # print("Q_pred_max = ", Q_pred_max)
    return Q_pred


# TODO - works but probably too long execution
def get_state_after_action(game, pawn):
    """
    to feed the ANN we need state before and after action
    :param pawn:
    :return: state after action
    """

    game = copy.deepcopy(game)

    # Check if there is an observation pending
    if not game.observation_pending:
        raise RuntimeError("There is no pending observation. "
                           "There has to be a pending observation has to be answered first")
    # Check if the given piece_to_move is among the current_move_pieces
    if len(game.current_move_pieces) and pawn not in game.current_move_pieces:
        raise RuntimeError("The piece given has to be among the given move_pieces")
    # If it is then move the piece
    elif len(game.current_move_pieces):
        new_enemys = game.players[game.current_player].move_piece(pawn, game.current_dice, game.current_enemys)

        # game.__set_enemy_pieces(game.current_player, new_enemys)  # does the same as lines below
        # Go through the enemies and set the changes in their pieces
        for e_i, e in enumerate(game.enemys_order[game.current_player]):
            game.players[e].set_pieces(new_enemys[e_i])

    # If there was no pieces that could be moved then nothing can be done
    else:
        pass  # This line is present for readability

    # Check if the player now is the winner
    player_is_a_winner = game.players[game.current_player].player_winner()
    if player_is_a_winner:
        # Check if player is the first winner
        if game.first_winner_was == -1:
            game.first_winner_was = game.current_player
        # Check if player has been added to game_winners
        if game.current_player not in game.game_winners:
            game.game_winners.append(game.current_player)

    next_player = True
    # In the first round the players has 3 attempts to get a piece out of home
    if game.round == 1 and \
            all(p_piece == 0 for p_piece in game.players[game.current_player].get_pieces()) and \
            game.current_start_attempts < 3:
        game.current_start_attempts += 1
        next_player = False
    else:
        game.current_start_attempts = 0
    # If it is not in the first round a dice on 6 will give an extra move
    if game.round != 1 and game.current_dice == 6:
        next_player = False

    # Set the observation pending to false as the last given observation was handled
    game.observation_pending = False

    # Get the environment after the move
    # after_obs = game.__gen_observation(game.current_player, roll_dice=False)
    roll_dice = False
    if roll_dice:
        # Roll the dice
        game.__dice_generator()
    dice = game.current_dice

    player = game.players[game.current_player]
    # Get the pieces that can be moved with the current dice
    move_pieces = player.get_pieces_that_can_move(dice)
    game.current_move_pieces = move_pieces

    # Get where the player's pieces are and the enemy's pieces are
    player_pieces, enemy_pieces = game.get_pieces(game.current_player)
    game.current_enemys = enemy_pieces

    # # Check if the player is a winner
    # player_is_a_winner = player.player_winner()
    # # Check if there is a winner
    # there_is_a_winner = any([p.player_winner() for p in game.players])

    # after_obs = dice, np.copy(move_pieces), np.copy(player_pieces), np.copy(
    #     enemy_pieces), player_is_a_winner, there_is_a_winner

    return get_game_state(game.get_pieces()[0])


# TODO
def optimize_model():
    pass


def action_selection(game, move_pieces, ann_model, begin_state, steps_done):
    """
    Use MLP to get the Q_value of chosen action and state
    choose the best action and return the new_state
    :param move_pieces:
    :param ann_model:
    :param begin_state:
    :param steps_done:
    :return:
    """
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200

    if len(move_pieces):
        # piece_to_move = move_pieces[0]  # move only 1 piece
        # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
        # piece_to_move = choose_action_furthest_pawn(begin_state, pieces, move_pieces)  # select furthest pawn

        # piece_to_move is chosen action!
        action_q_l = []
        best_Q_val = -99999999999999

        for movable_pawn in move_pieces:
            # get pawn with best Q

            current_player = 0
            # action = (x0, xf)
            new_state = get_state_after_action(game, movable_pawn)
            # print("new_state = ", new_state)

            # look for the position of the given pawn before and after a move
            action = (begin_state[current_player][movable_pawn] / 60, new_state[current_player][movable_pawn] / 60)

            with torch.no_grad():
                # TODO: every pawn has the same Q ???
                Q_val = get_policy_action_ann(ann_model, begin_state, action)
            action_q_l.append((movable_pawn, Q_val))

            # epsilon greedy
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

        if sample > eps_threshold:
            # choose best pawn
            for action, Q_v in action_q_l:
                if Q_v > best_Q_val:
                    piece_to_move = action
        else:
            # get random pawn
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
            new_state = get_state_after_action(game, piece_to_move)

    else:
        piece_to_move = -1
        new_state = get_state_after_action(game, piece_to_move)

    return piece_to_move, new_state




def dqn_approach():
    import ludopy
    import numpy as np

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    g = ludopy.Game()
    there_is_a_winner = False

    # create model of ANN
    ann_model = Feedforward(input_size=242, hidden_size=21)  # model of ANN
    epochs = 3


    steps_done = 0
    optimizer = optim.RMSprop(ann_model.parameters())  # TODO!

    for epoch in range(epochs):

        while not there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = g.get_observation()
            pieces = g.get_pieces()

            # state_own_start = get_game_state_own(pieces)
            # print("<state_own_start> round = %d \tstate = %s" % (g.round, state_own_start))
            begin_state = get_game_state(pieces[0])
            # print("<begin_state> round = %d \tstate = %s" % (g.round, begin_state[0]))
            print("<begin_state> round = %d " % (g.round))

            """ select and perform an action """
            piece_to_move, new_state = action_selection(g, move_pieces, ann_model, begin_state, steps_done)

            # perform action
            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
            reward = get_reward(begin_state, piece_to_move, new_state)
            # Transition(begin_state, piece_to_move, new_state, reward)  # TODO: Need this?

            """ perform one step of optimization """
            optimize_model()


    print("Saving history to numpy file")
    g.save_hist("game_history.npy")
    print("Saving game video")
    g.save_hist_video("game_ANN_test.mp4")

    return True


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, dqn_approach())


if __name__ == '__main__':
    unittest.main()
