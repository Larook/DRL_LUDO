import unittest
import sys
import numpy as np
sys.path.append("../")

def randwalk():
    import ludopy
    import numpy as np

    player_0_won = False
    games_played = 0
    while not player_0_won:
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
            if there_is_a_winner and player_i == 0:
                player_0_won = True
        games_played += 1
        print("games_played = ", games_played)
    print("Saving history to numpy file")
    g.save_hist("game_history.npy")
    print("Saving game video")
    g.save_hist_video("randomwalk_game_video.mp4")

    return True


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, randwalk())

if __name__ == '__main__':
    unittest.main()
