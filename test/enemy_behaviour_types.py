import numpy as np

from dqn_action_selection import get_game_state
from rewards import map_enemy_tile_id_to_player_0


def perform_random_action(move_pieces):
    action = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
    return action


def perform_semismart_aggressive(player_i, pieces_on_board, dice, move_pieces, player_pieces, enemy_pieces):
    # state_begin = get_game_state(pieces_on_board)

    # choose an action that knocks down opponents
    # check current state of pieces and their possible location after moving with the dice
    # have the list of the future locations

    future_locations = []
    for loc_current in player_pieces:
        if loc_current == 0:
            if dice == 6:
                loc_next = 1
            else:
                loc_next = loc_current
        else:
            loc_next = loc_current + dice
        future_locations.append(loc_next)

    # for every future location check if there are enemy pieces to knock
    # use mapping from enemy_to_player map_enemy_tile_id_to_player_0 or enemy_pieces
    for piece_i, loc_next in enumerate(future_locations):

        for enemy_i in range(len(enemy_pieces)):
            curr_enemy_pieces = enemy_pieces[enemy_i]
            for enemy_tile in curr_enemy_pieces:
                # for enemy_i, enemy_tile in enumerate(enemy_pieces):
                # TODO: wrongly maps in this case! works for player0 though
                player_tile = map_enemy_tile_id_to_player_0(enemy_i+1, enemy_tile)
                if loc_next == player_tile:
                    return piece_i  # this should knock an opponent!

    # if couldnt knock down anyone do random
    return perform_random_action(move_pieces)

