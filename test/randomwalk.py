import unittest
import sys
import numpy as np
sys.path.append("../")

def randwalk():
    import ludopy
    import numpy as np

    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

    print("Saving history to numpy file")
    g.save_hist("game_history.npy")
    print("Saving game video")
    g.save_hist_video("game_video.mp4")

    return True


def get_game_state_own(pieces):
    """
    :returns state in form of 0-|10-13,15,|41-38,45,|10-14,|
    """
    tiles_horizon = 6   # get the state of 4 pawns with +-6 fields
    players = pieces[0]
    player_0 = players[0]

    # check all the players and their all pieces and tell when they are in range of +-6 pieces from "my_pawn"
    """
    somehow save the information about the state in a string maybe?
    In general it works but cant print when somebody was struck out!
    """
    distance_between_players = 13

    state = ""
    for my_pid in range(len(player_0)):
        my_pawn = player_0[my_pid]
        if my_pawn == 0:
            state += str(my_pawn) + "-|"
        else:
            state += str(my_pawn) + "-"

            for i_player in range(1, len(players)):  # dont check own pawns
                for i_pawn in range(len(players[i_player])):

                    # we have to check if they are nearby enemy pawns any our pawn
                    enemy_pawn_now = players[i_player][i_pawn]
                    if enemy_pawn_now != 0:

                        # map enemy position to player_0 position - WORKS
                        enemy_pawn_now += i_player * distance_between_players
                        if enemy_pawn_now > 53:
                            enemy_pawn_now -= 53

                        # check if enemy pawns in radius of horizon
                        if enemy_pawn_now in range(my_pawn - tiles_horizon, my_pawn + tiles_horizon):
                            state += str(enemy_pawn_now) + ","
                            print("found enemy[%d] pawn = %d near player_0[%d] = %d" % (i_player, enemy_pawn_now, my_pid, my_pawn))
                        if my_pawn == enemy_pawn_now:
                            print("Someone should die") # 37, 40, 19

            state += "|"
    return state


def choose_action_furthest_pawn_own(state, move_pieces):
    """
    RuntimeError: The piece given has to be among the given move_pieces - cant return piece that is unmovable

    :param state:
    :param move_pieces:
    :return:
    """
    out_of_home_found = False
    furthest_dist = 0
    furthest_id_pawn = 100
    # if all are zeros then choose random
    pawns = state.split("|")[0:-1]
    for i in range(len(pawns)):
        my_pawn_state = pawns[i]
        this_dist = int(my_pawn_state.split("-")[0])
        if this_dist != 0:
            out_of_home_found = True and i in move_pieces
            if this_dist >= furthest_dist:
                if i in move_pieces:
                    furthest_id_pawn = i

    if not out_of_home_found: # or furthest_id_pawn == 100:  # this situation when out of home, but unmovable
        # choose random pawn
        return move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        # print("move_pieces  ", move_pieces)
        # print("furthest_id_pawn ", furthest_id_pawn)
        return furthest_id_pawn


def get_game_state_papers(pieces):
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


def choose_action_furthest_pawn_paper(state, pieces, move_pieces):
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


def playground_1():
    import ludopy
    import numpy as np

    # g = ludopy.Game(ghost_players=[0, 1, 3])  # This will prevent players 1 and 3 from moving out of the start and thereby they are not in the game
    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        pieces = g.get_pieces()

        # state_own_start = get_game_state_own(pieces)
        # print("<state_own_start> round = %d \tstate = %s" % (g.round, state_own_start))
        state_papers_start = get_game_state_papers(pieces)
        print("<state_papers_start> round = %d \tstate = %s" % (g.round, state_papers_start[0]))

        if len(move_pieces):
            # piece_to_move = move_pieces[0]  # move only 1 piece
            # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
            # piece_to_move = choose_action_furthest_pawn_own(state_own_start, move_pieces)  # go one piece always with old state
            piece_to_move = choose_action_furthest_pawn_paper(state_papers_start, pieces, move_pieces)  # select furthest pawn

            # TODO: piece_to_move is action!
            piece_to_move = get_policy_action_ann(state_papers_start)
        else:
            piece_to_move = -1

        # Ok so we have to choose a piece to move and see the value we are getting
        # see how the states are saved
        # want to save position of every piece, and if we can then the surroundings of it -> g.get_pieces

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

        # pieces, enemy_pieces = g.get_pieces(seen_from=0)
        # state_own_end = get_game_state_own(g.get_pieces())
        # print("<end> round = %d \tstate = %s" % (g.round, state_own_end))

    print("Saving history to numpy file")
    g.save_hist("game_history.npy")
    print("Saving game video")
    g.save_hist_video("game_playground_1.mp4")

    return True



class MyTestCase(unittest.TestCase):
    def test_something(self):
        # self.assertEqual(True, randwalk())
        self.assertEqual(True, playground_1())


if __name__ == '__main__':
    unittest.main()
