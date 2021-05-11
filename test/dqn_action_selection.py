import copy
import math
import random
import time

from Feedforward import *
import config


def take_best_action(game, move_pieces, begin_state, q_net):
    action_q_l = []
    best_Q_val = -9999999999

    for possible_action in move_pieces:
        # get piece with best Q
        before_new_state_t = time.time()
        new_state = get_state_after_action(game, possible_action)
        # print("<timing> get_state_after_action =", time.time() - before_new_state_t)

        input_ann = get_reshaped_ann_input(begin_state, new_state, possible_action)
        input_ann = torch.FloatTensor(input_ann)

        # TODO: check if no_grad() is needed
        with torch.no_grad():
            Q_est = q_net(input_ann)

        action_q_l.append((possible_action, Q_est))

    for action, Q_v in action_q_l:
        if Q_v > best_Q_val:
            piece_to_move = action
    return piece_to_move


def action_selection(game, move_pieces, q_net, begin_state, steps_done, is_random=False, show=False,
                     exploit_model=False):
    """
    Use MLP to get the Q_value of chosen action and state
    choose the best action and return the new_state
    :param move_pieces:
    :param q_net:
    :param begin_state:
    :param steps_done:
    :return:
    """
    """ !!! check help/tweak_epsilon.py """
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 10000  # after 10 games eps_threshold=0.053

    if len(move_pieces):
        # epsilon greedy
        sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.7 * steps_done / EPS_DECAY)

        config.epsilon_now = eps_threshold
        # print("eps_threshold = ", eps_threshold)
        steps_done += 1

        if sample > eps_threshold or exploit_model:
            # choose best pawn
            if not is_random:
                piece_to_move = take_best_action(game=game, move_pieces=move_pieces, begin_state=begin_state, q_net=q_net)
                if show:
                    print("\tbest action selected")
            else:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn

        else:
            # get random pawn
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
            if show:
                print("\trandom action selected")
        new_state = get_state_after_action(game, piece_to_move)

    else:
        piece_to_move = -1
        new_state = get_state_after_action(game, piece_to_move)
        if show:
            print("\tthe only possible action selected")

    return piece_to_move, new_state


def get_state_after_action(game, pawn):
    """
    based on game.answer_observation()
    to feed the ANN we need state before and after action
    :param pawn:
    :return: state after action
    """

    # game = copy.copy(game)
    game = copy.deepcopy(game)

    # # Check if there is an observation pending
    # if not game.observation_pending:
    #     raise RuntimeError("There is no pending observation. "
    #                        "There has to be a pending observation has to be answered first")
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
    # else:
    #     pass  # This line is present for readability

    # # Check if the player now is the winner
    # player_is_a_winner = game.players[game.current_player].player_winner()
    # if player_is_a_winner:
    #     # Check if player is the first winner
    #     if game.first_winner_was == -1:
    #         game.first_winner_was = game.current_player
    #     # Check if player has been added to game_winners
    #     if game.current_player not in game.game_winners:
    #         game.game_winners.append(game.current_player)

    # next_player = True
    # # In the first round the players has 3 attempts to get a piece out of home
    # if game.round == 1 and \
    #         all(p_piece == 0 for p_piece in game.players[game.current_player].get_pieces()) and \
    #         game.current_start_attempts < 3:
    #     game.current_start_attempts += 1
    #     next_player = False
    # else:
    #     game.current_start_attempts = 0
    # # If it is not in the first round a dice on 6 will give an extra move
    # if game.round != 1 and game.current_dice == 6:
    #     next_player = False

    # # Set the observation pending to false as the last given observation was handled
    # game.observation_pending = False

    # Get the environment after the move
    # after_obs = game.__gen_observation(game.current_player, roll_dice=False)

    player = game.players[game.current_player]
    # Get the pieces that can be moved with the current dice
    move_pieces = player.get_pieces_that_can_move(game.current_dice)
    game.current_move_pieces = move_pieces

    # # Get where the player's pieces are and the enemy's pieces are
    # player_pieces, enemy_pieces = game.get_pieces(game.current_player)
    # game.current_enemys = enemy_pieces

    # # Check if the player is a winner
    # player_is_a_winner = player.player_winner()
    # # Check if there is a winner
    # there_is_a_winner = any([p.player_winner() for p in game.players])

    # after_obs = dice, np.copy(move_pieces), np.copy(player_pieces), np.copy(
    #     enemy_pieces), player_is_a_winner, there_is_a_winner

    return get_game_state(game.get_pieces()[0])


def get_game_state(pieces_seen_from_players):
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
    for index, pawn_positions in enumerate(pieces_seen_from_players):
        state = np.zeros(POSITIONS_PER_PLAYER)
        for pawn_id in pawn_positions:
            state[pawn_id] += 0.25
        state_all[index] = state

    # print("state_all", state_all)
    return state_all
