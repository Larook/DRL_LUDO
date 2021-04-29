

def count_pieces_on_tile(player_no, state, tile_no):
    value = state[player_no][tile_no]
    return value * 4


def get_max_reward_from_state(game, state, possible_actions):
    """ need to check all the movable pieces, and calculate all the possible rewards and get the maximum one"""
    from DQN_plays import get_state_after_action

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


def enemy_pieces_nearby(player_id, state, horizon):
    """ horizon - how far away pieces should be considered
    :return True when there are some pieces in danger """
    ids_where_pieces_are_safe = [0, 1, range(53,59)]

    player_state = state[player_id]
    enemy_states = []
    for i in range(len(state)):
        if i != player_id:
            enemy_states.append(state[i])

    tiles_with_player_pieces = []
    for tile_id, value in enumerate(player_state):
        if value != 0:
            # ok we have some of our pieces here
            tiles_with_player_pieces.append(tile_id)
    tiles_with_player_pieces = set(tiles_with_player_pieces)

    tiles_to_check = []
    for tile_id in tiles_with_player_pieces:
        # get the range of tiles to check
        min_tile = tile_id - horizon
        max_tile = tile_id + horizon

        # select only dangerous positions
        tiles_to_consider = list(range(min_tile, max_tile))
        tiles_to_consider = [num for num in tiles_to_consider if num >= 2 and num <= 52]

    for enemy_state in enemy_states:
        for i in tiles_to_consider:
            if enemy_state[i] > 0:
                # there is an enemy in our horizon!
                print("whole state:")
                print(state)
                return True
    return False


def get_reward(state_begin, piece_to_move, state_new, pieces_player_now):
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
        in_home_before = count_pieces_on_tile(player_no=player_i, state=state_begin, tile_no=home_tile)
        in_home_after = count_pieces_on_tile(player_no=player_i, state=state_new, tile_no=home_tile)
        if in_home_after > in_home_before:
            reward += 0.15
            knocked_pieces += 1  # debug only

        # check if any of the opponents won the game
        if count_pieces_on_tile(player_no=player_i, state=state_new, tile_no=finished_tile) == 4:
            enemies_already_won = True

    # check leaving the home for current player and finishing the game
    player_i = 0
    in_home_before = count_pieces_on_tile(player_no=player_i, state=state_begin, tile_no=home_tile)
    in_home_after = count_pieces_on_tile(player_no=player_i, state=state_new, tile_no=home_tile)
    if in_home_after < in_home_before:
        reward += 0.25

    # check the end of the game
    if enemies_already_won:
        reward -= 1
    elif count_pieces_on_tile(player_no=player_i, state=state_new, tile_no=finished_tile) == 4:
        # print("player 0 wins the game in this round")
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

    # 0.05 for forming a blockade
    player_i = 0
    # print("begin_state", state_begin[player_i])
    # print("new_state", state_new[player_i])
    if state_begin[player_i][0] != 1:
        state_diff = state_new[player_i] - state_begin[player_i]
        # print("state_diff", state_diff)
        for tile_id, value in enumerate(state_diff):
            # when new pawn moved and moved in actual dangerous position
            if value >= 0.25 and tile_id in range(1, 54):
                # if the actual number of the players on the tile increased
                pieces_there_after = count_pieces_on_tile(player_i, state_new, tile_id)
                if pieces_there_after > 1:
                    if pieces_there_after > count_pieces_on_tile(player_i, state_begin, tile_id):
                        reward += 0.05
                        # print("MADE A BLOCKADE, SIR!")
                        # exit("blockade check")

                        # if there are enemy pieces in range +- 6 add 0.2 more reward
                        #     • 0.2 for defending a vulnerable piece.
                        if enemy_pieces_nearby(player_id=player_i, state=state_new, horizon=6):
                            reward += 0.15
                            # print("WE WERE IN DANGER, SIR!")
                            # exit("life saving blockade")

    if reward < 0:
        print('ENEMY ENDED THE GAME, reward = ', reward)
    if reward >= 1:
        print('AI PLAYER ENDED THE GAME , reward', reward)
    return reward

def test_get_reward():
    return False
