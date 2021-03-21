import unittest
import sys
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

    # print("Hello from get_game_state")
    fields_horizon = 6   # get the state of 4 pawns with +-6 fields
    players = pieces[0]
    player_0 = players[0]
    # player_1_state = pieces[1]
    # player_2_state = pieces[2]
    # player_3_state = pieces[3]

    # check all the players and their all pieces and tell when they are in range of +-6 pieces from "my_pawn"
    """
    somehow save the information about the state in a string maybe?
    54-xxxExxPxxxxxx|12-xxxxxxPxxxxxx|0|0

    <> Notes on the way of saving state:
    
    sometimes one enemy pawn is seen by our 2 pawns:
            found enemy[2] pawn = 1 near player_0[0] = 4
            found enemy[2] pawn = 1 near player_0[1] = 1
            state now =  4-1,|1-1,|0|0|
    
    I cant see them dying - ok field 1 is safe state
    but why
        found enemy[3] pawn = 6 near player_0[0] = 4
        found enemy[2] pawn = 18 near player_0[1] = 18
        state now =  4-6,|18-18,|0|0|
        
    Do other players start from the same global coordinates of saved tiles? Or is it player specific?
    I think it is player-specific because max index of each pawn is 59
    we have to somehow unify this, so the states of other players are mapped to the player0 - there is 13 tiles difference between each base
    
    enemy positions cant exceed 52 -> if >52 then next one is 1 WORKS
            
            round = 6 	state = 0|0|30-|0|
            round = 7 	state = 0|0|30-|0|
            found enemy[1] pawn = 41 near player_0[2] = 38
            found enemy[2] pawn = 40 near player_0[2] = 38
            round = 7 	state = 0|0|38-41,40,|0|
            found enemy[1] pawn = 42 near player_0[2] = 38
            found enemy[2] pawn = 40 near player_0[2] = 38
            round = 7 	state = 0|0|38-42,40,|0|
            found enemy[1] pawn = 42 near player_0[2] = 38
            found enemy[2] pawn = 41 near player_0[2] = 38
            round = 7 	state = 0|0|38-42,41,|0|
            found enemy[1] pawn = 42 near player_0[2] = 38
            found enemy[2] pawn = 41 near player_0[2] = 38
            round = 8 	state = 0|0|38-42,41,|0|                <- here enemy at 41
            found enemy[1] pawn = 42 near player_0[2] = 41
            round = 8 	state = 0|0|41-42,|0|
            found enemy[1] pawn = 42 near player_0[2] = 41
            round = 8 	state = 0|0|41-42,|0|                   <- here enemy dead
            round = 8 	state = 0|0|41-|0|
            round = 8 	state = 0|0|41-|0|
            round = 9 	state = 0|0|41-|0|
            
            In general it works but cant print when somebody was struck out!
        
    """
    distance_between_players = 13

    state = ""
    for my_pid in range(len(player_0)):
        my_pawn = player_0[my_pid]
        if my_pawn == 0:
            state += str(my_pawn) + "|"
        else:
            state += str(my_pawn) + "-"

            for i_player in range(1, len(players)):  # dont check own pawns
                for i_pawn in range(len(players[i_player])):

                    # now we have all enemy pawns
                    #TODO we have to check if they are nearby any our pawn - we have to compare them with all my pawns
                    enemy_pawn_now = players[i_player][i_pawn]
                    if enemy_pawn_now != 0:

                        # map enemy position to player_0 position - WORKS
                        enemy_pawn_now += i_player * distance_between_players
                        if enemy_pawn_now > 53:
                            enemy_pawn_now -= 53

                        if enemy_pawn_now in range(my_pawn - fields_horizon, my_pawn + fields_horizon):
                            state += str(enemy_pawn_now) + ","
                            print("found enemy[%d] pawn = %d near player_0[%d] = %d" % (i_player, enemy_pawn_now, my_pid, my_pawn))
                            if my_pawn == enemy_pawn_now:
                                print("Someone should die") # 37, 40, 19
                            # print("pieces[i_player][i_pawn] = ", players[i_player][i_pawn])
                        # else:
                            # state += str(my_pawn) + "?"
                            # pass

            state += "|"
    # print("state now = ", state)
    return state


def playground_1():
    import ludopy
    import numpy as np

    # g = ludopy.Game(ghost_players=[0, 1, 3])  # This will prevent players 1 and 3 from moving out of the start and thereby they are not in the game
    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):
            # piece_to_move = move_pieces[0]  # move only 1 piece
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            piece_to_move = -1

        # TODO: Ok so we have to choose a piece to move and see the value we are getting
        # see how the states are saved
        # want to save position of every piece, and if we can then the surroundings of it -> g.get_pieces

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

        # pieces, enemy_pieces = g.get_pieces(seen_from=0)
        # save state
        state = get_game_state(g.get_pieces())
        # pieces = g.get_pieces()
        # pieces_0 = g.get_pieces(seen_from=0)
        print("round = %d \tstate = %s" % (g.round, state))

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
