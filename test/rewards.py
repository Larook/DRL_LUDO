

def count_pieces_on_tile(player_no, state, tile_no):
    value = state[player_no][tile_no]
    return value * 4


def get_max_reward_from_state(game, state, possible_actions):
    """ need to check all the movable pieces, and calculate all the possible rewards and get the maximum one"""
    from ANN_action_selection import get_state_after_action

    max_reward = 0
    for action in possible_actions:

        # get next state
        new_state = get_state_after_action(game, action)

        # get reward
        reward = get_reward(state, action, new_state, pieces_player_now=game.get_pieces()[game.current_player][game.current_player])
        # reward = get_reward(state, action, new_state)
        if reward >= max_reward:
            max_reward = reward

    return max_reward


def get_reward(begin_state, piece_to_move, new_state, pieces_player_now):
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

    # check the end of the game
    if enemies_already_won:
        reward -= 1
    elif count_pieces_on_tile(player_no=player_i, state=begin_state, tile_no=finished_tile) == 4:
        reward += 1

    # check if moved piece is the furthest away
    furthest_piece, furthest_dist = 0, 0
    # print("pieces_player_now", pieces_player_now)
    for piece in range(len(pieces_player_now)):
        # print("pieces_player_now[piece]", pieces_player_now[piece])
        if pieces_player_now[piece] >= furthest_dist:
            furthest_dist = pieces_player_now[piece]
            furthest_piece = piece
    # print("furthest_piece ", furthest_piece)
    if furthest_piece == piece_to_move and furthest_dist != 0:
        reward += 0.1
        # exit('chosen furthest one')
    return reward
