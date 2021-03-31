import unittest
import sys
import numpy as np
sys.path.append("../")


def get_game_state(pieces):
    """
    state represented by  240 variables - for each player 60
    Each state (id of tile) can have values 0 - 1, where 0 means 0 pawns on the tile, and 1 means 4 pawns on tile

    :param pieces:
    :return:
    """
    players = pieces[0]
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


def get_policy_action_ann(state_papers_start):
    """
    DQN network to select the action of optimal policy
    :param state_papers_start:
    :return:
    """
    pass


def dqn_approach():
    import ludopy
    import numpy as np

    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        pieces = g.get_pieces()

        # state_own_start = get_game_state_own(pieces)
        # print("<state_own_start> round = %d \tstate = %s" % (g.round, state_own_start))
        state_start = get_game_state(pieces)
        print("<state_start> round = %d \tstate = %s" % (g.round, state_start[0]))

        if len(move_pieces):
            # piece_to_move = move_pieces[0]  # move only 1 piece
            # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
            # piece_to_move = choose_action_furthest_pawn(state_start, pieces, move_pieces)  # select furthest pawn

            # TODO: piece_to_move is action!
            piece_to_move = get_policy_action_ann(state_start)
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

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
