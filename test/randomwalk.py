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


def playground_1():
    import ludopy
    import numpy as np

    g = ludopy.Game(ghost_players=[1, 3])  # This will prevent players 1 and 3 from moving out of the start and thereby they are not in the game
    # g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()

        if len(move_pieces):

            # piece_to_move = move_pieces[0]  # move only 1 piece
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        else:
            piece_to_move = -1

        # Ok so we have to choose a piece to move and see the value we are getting
        # see how the states are saved
        # want to save position of every piece, and if we can then the surroundings of it

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)

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
