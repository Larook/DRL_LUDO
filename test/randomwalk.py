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


def get_game_state(pieces):
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


def choose_action_furthest_pawn(state, move_pieces):
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


def playground_1():
    import ludopy
    import numpy as np

    # g = ludopy.Game(ghost_players=[0, 1, 3])  # This will prevent players 1 and 3 from moving out of the start and thereby they are not in the game
    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        state_start = get_game_state(g.get_pieces())
        print("<START> round = %d \tstate = %s" % (g.round, state_start))

        if len(move_pieces):
            # piece_to_move = move_pieces[0]  # move only 1 piece
            # piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            piece_to_move = choose_action_furthest_pawn(state_start, move_pieces)
        else:
            piece_to_move = -1

        # TODO: Ok so we have to choose a piece to move and see the value we are getting
        # see how the states are saved
        # want to save position of every piece, and if we can then the surroundings of it -> g.get_pieces

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

        # pieces, enemy_pieces = g.get_pieces(seen_from=0)
        state_end = get_game_state(g.get_pieces())
        print("<end> round = %d \tstate = %s" % (g.round, state_end))

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
