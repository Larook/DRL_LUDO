
def count_pieces_on_tile(player_no, state, tile_no):
    value = state[player_no][tile_no]
    return value * 4


def get_reward(begin_state, piece_to_move, new_state):
    # TODO: maybe also add reward for entering a piece into safe_zone and blockade?
    """
    • 1.0 for winning a game.
        • 0.25 for releasing a piece from HOME.
    • 0.2 for defending a vulnerable piece.
        • 0.15 for knocking an opponent’s piece.
    • 0.1 for moving the piece that is closest to home.
    • 0.05 for forming a blockade.
    • -0.25 for getting a piece knocked in the next turn.
        • -1.0 for losing a game.
    """
    home_tile = 0
    finished_tile = 59
    reward = 0

    knocked_pieces = 0

    enemies_already_won = False
    for player_i in range(1, 4):
        # first check knocking out enemies and if game already won by enemies
        # check if enemies return home - detect knocked opponents
        in_home_before = count_pieces_on_tile(player_no=player_i, state=begin_state, tile_no=home_tile)
        in_home_after = count_pieces_on_tile(player_no=player_i, state=new_state, tile_no=home_tile)
        if in_home_after > in_home_before:
            reward += 0.15
            knocked_pieces += 1  # debug only

        # check if any of the opponents won the game
        if count_pieces_on_tile(player_no=player_i, state=new_state, tile_no=finished_tile) == 4:
            enemies_already_won = True

    # check leaving the home for current player and finishing the game
    player_i = 0
    in_home_before = count_pieces_on_tile(player_no=player_i, state=begin_state, tile_no=home_tile)
    in_home_after = count_pieces_on_tile(player_no=player_i, state=new_state, tile_no=home_tile)
    if in_home_after < in_home_before:
        reward += 0.25

    if enemies_already_won:
        reward -= 1
    elif count_pieces_on_tile(player_no=player_i, state=begin_state, tile_no=finished_tile) == 4:
        reward += 1
    return reward
