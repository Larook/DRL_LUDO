import time

import config
from collections import namedtuple

from dqn_action_selection import get_state_after_action


def count_pieces_on_tile(player_no, state, tile_no):
    value = state[player_no][tile_no]
    return value * 4


def map_enemy_tile_id_to_player_0(i_enemy, tile_id):
    """ knowing enemy id and the tile that he sees
     check what is the tile in the coordinates of ai player """
    tile_ai = i_enemy*13 + tile_id
    if tile_ai >= 53:
        tile_ai -= 53
        tile_ai += 1
    if tile_id >= 54:
        tile_ai = 0
    return tile_ai


def get_max_reward_from_state(pieces_player_begin, dice, state_begin, possible_actions):
    """ need to check all the movable pieces, and calculate all the possible rewards and get the maximum one"""
    from DQN_plays import get_state_after_action_g

    max_reward = 0
    for action in possible_actions:

        # get next state
        # state_new = get_state_after_action_g(game, action)  # todo: rewrite to not using game
        state_new = get_state_after_action(pieces_player_begin, state_begin, dice, action)

        # get reward
        # reward = get_reward(dice, state, action, state_new, pieces_player_begin=pieces_player_begin)
        reward = get_reward(dice=dice, state_begin=state_begin, piece_to_move=action, state_new=state_new,
                            pieces_player_begin=pieces_player_begin, actual_action=False)  # immediate reward

        # reward = get_reward(state, action, state_new)
        if reward >= max_reward:
            max_reward = reward

    return max_reward


def enemy_pieces_nearby(player_id, state, horizon):
    """ horizon - how far away pieces should be considered
    :return True when there are some pieces in danger """
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
                # print("whole state:")
                # print(state)
                return True
    return False


def map_tile_id_p0_to_enemy(i_enemy, tile_id_p0_check):
    """ p0-1 -> e1-40, e2-27, e3-14
        p0-2 -> e2-41, e2-28, e3-15
    """
    if tile_id_p0_check == 0:
        return 0

    tile_enemy = i_enemy*(-13) + tile_id_p0_check
    if tile_enemy >= 53:
        tile_enemy -= 53
        # tile_enemy += 1
    if tile_enemy >= 54:
        tile_enemy = 0

    if tile_enemy < 1:
        tile_enemy += 53
        tile_enemy -= 1

    return tile_enemy


def did_loose_piece(state_begin, state_prev_new, pieces_player_begin):
    # check all the pieces and count how many in home
    if len(config.learning_info_data.whole_list) < 1 or int(config.learning_info_data.whole_list[-1]['round']) < 2:
        # if int(config.learning_info_data.whole_list[-1]['round']) < 2:
        # print("config.learning_info_data.whole_list['round']", config.learning_info_data.whole_list[-1]['round'])
        return False
    # print("config.learning_info_data.whole_list[-1]['round']", config.learning_info_data.whole_list[-1]['round'])
    no_piece_home_before = 0
    for i_piece, val_piece in enumerate(pieces_player_begin):
        if pieces_player_begin[i_piece] == config.home_tile:
            no_piece_home_before += 1

    # check the last positions of our pieces
    piece_positions_prev_new = []
    pieces_prev_new_home = 0
    # print("state_prev_new[0]", type(state_prev_new[0].tolist()), state_prev_new[0].tolist())
    for i_tile, value_tile in enumerate(state_prev_new[0]):
        # print("i_tile = ", i_tile, "value_tile = ", value_tile)
        if value_tile > 0:
            pieces_there = int(value_tile * 4)
            for i in range(pieces_there):
                piece_positions_prev_new.append(i_tile)

                if i_tile == 0:
                    pieces_prev_new_home += 1

    # print("state_prev_new[0]", state_prev_new[0])
    # print("state_begin[0]", state_begin[0])

    pieces_begin = pieces_player_begin.tolist()
    pieces_begin.sort()
    pieces_prev = piece_positions_prev_new
    pieces_prev.sort()

    # print("pieces_prev", pieces_prev)
    # print("pieces_begin", pieces_begin)
    # time.sleep(0.2)

    if pieces_begin != pieces_prev:
        # something changed in between!
        # check the number of pieces in home
        pieces_now_home = state_begin[0][config.home_tile]
        if pieces_prev_new_home > pieces_now_home:
            in_home_prev = pieces_prev.count(0)
            in_home_now = pieces_begin.count(0)
            if in_home_now > in_home_prev:

                # find the id of the tile that the piece was lost
                tiles_id_difference_p0 = list(set(pieces_prev).difference(set(pieces_begin)))
                # print("state_prev_new", state_prev_new)
                # print("state_begin", state_begin)

                # look in all of the enemies and check if the enemy moved there
                for i_enemy in range(1, len(state_begin)):

                    for tile_id_p0_check in tiles_id_difference_p0:

                        # state_enemy_from_player0 = get_enemy_mapped_state(i_enemy=i_enemy, state=state_begin)
                        enemy_begin_state = state_begin[i_enemy]
                        enemy_last_state = state_prev_new[i_enemy]
                        # todo:
                        # ok we have the tile where the enemy could have been - mapped to player_0
                        tile_id_player0_on_enemy = map_tile_id_p0_to_enemy(i_enemy, tile_id_p0_check)

                        # check if enemy moved to the tile
                        enemy_pieces_begin = enemy_begin_state[tile_id_player0_on_enemy]
                        enemy_pieces_last = enemy_last_state[tile_id_player0_on_enemy]

                        # print("pieces_begin", pieces_begin)
                        # print("tile_id_p0_check", tile_id_p0_check)
                        # print("state_prev_new", state_prev_new)
                        # print("state_begin", state_begin)
                        # print("state_begin[0][tile_id_p0_check]", state_begin[0][tile_id_p0_check])
                        # print("enemy_begin_state", enemy_begin_state)
                        # print("tile_id_player0_on_enemy", tile_id_player0_on_enemy)
                        # print("enemy_last_state[tile_id_player0_on_enemy]", enemy_last_state[tile_id_player0_on_enemy])
                        # print("enemy_begin_state[tile_id_player0_on_enemy]", enemy_begin_state[tile_id_player0_on_enemy])

                        # print("tile_id_p0_check = %d| state_prev_new[0][tile_id_p0_check]=%.2f->state_begin[0][tile_id_p0_check]=%.2f" % (tile_id_p0_check, state_prev_new[0][tile_id_p0_check], state_begin[0][tile_id_p0_check]))
                        # print("tile_id_p0_check = %d| i_enemy=%d, tile_id_player0_on_enemy=%d | enemy_pieces_last=%.2f->enemy_pieces_begin=%.2f" % (tile_id_p0_check, i_enemy, tile_id_player0_on_enemy, enemy_pieces_last, enemy_pieces_begin))
                        # exit()

                        # check if player lost the piece from tile
                        # if state_begin[i_enemy][tile_id_p0_check] > 0:
                        # if enemy_pieces_begin > 0:
                        if enemy_pieces_begin > 0 and enemy_pieces_begin > enemy_pieces_last:

                            # print("state_prev_new[i_enemy][tile_id_p0_check]", state_prev_new[i_enemy][tile_id_p0_check])
                            # print("state_begin[i_enemy][tile_id_p0_check]", state_begin[i_enemy][tile_id_p0_check])
                            # print("state_begin[i_enemy]", state_begin[i_enemy])
                            # print("state_prev_new[0][tile_id_p0_check]", state_prev_new[0][tile_id_p0_check])

                            # time.sleep(3)
                            # exit("piece lost")
                            # print("DETECTED LOOSING A PIECE!")
                            return True
                        else:
                            # print("Didnt detect loosing a piece!")
                            pass
                # print("checked all the enemies and couldnt find!")
                # exit("checked all the enemies and couldnt find!")
            else:
                # print("detected weird movement - probably using a star")
                pass
    else:
        pass
        # print("the positions are the same - all good")
        # time.sleep(1)

    return False


def get_reward(dice, state_begin, piece_to_move, state_new, pieces_player_begin, actual_action=False):
    """
        • 1.0 for winning a game.
        • 0.25 for releasing a piece from HOME.
        • 0.2 for defending a vulnerable piece.
        • 0.15 for knocking an opponent’s piece.
        • 0.1 for moving the piece that is closest to home - if moved from safe space then only 0.05
        • 0.05 for forming a blockade.
        • -0.25 for getting a piece knocked in the next turn.
        • -1.0 for losing a game.

        • go on globe (safe) 0.12
        • use a star (speed) 0.17
    """

    ids_where_pieces_are_safe = [1, range(53, 59)]
    player_i = 0
    state_diff = state_new[player_i] - state_begin[player_i]

    reward = 0

    knocked_pieces = 0

    enemies_already_won = False
    for player_i in range(1, 4):
        # first check knocking out enemies and if game already won by enemies
        # check if enemies return home - detect knocked opponents
        # in_home_before = count_pieces_on_tile(player_no=player_i, state=state_begin, tile_no=home_tile)
        in_home_before = state_begin[player_i][config.home_tile]
        in_home_after = state_new[player_i][config.home_tile]
        if in_home_after > in_home_before:
            reward += 0.15
            knocked_pieces += 1  # debug only
            if actual_action:
                config.rewards_detected['knock_opponent'] += 1
            # print("state_begin\n", state_begin)
            # print("state_new\n", state_new)
            # exit('Check kocking oponnent 39')

        # check if any of the opponents won the game
        # if count_pieces_on_tile(player_no=player_i, state=state_new, tile_no=config.finished_tile) == 4:
        if state_new[player_i][config.finished_tile] == 1:
            enemies_already_won = True

    # check leaving the home for current player and finishing the game
    player_i = 0
    # ai_pl_in_home_before = count_pieces_on_tile(player_no=player_i, state=state_begin, tile_no=config.home_tile)
    ai_pl_in_home_before = state_begin[0][config.home_tile]
    ai_pl_in_home_after = state_new[0][config.home_tile]
    # ai_pl_in_home_after = count_pieces_on_tile(player_no=player_i, state=state_new, tile_no=config.home_tile)

    # print("ai_pl_in_home_after", ai_pl_in_home_after)
    # print("ai_pl_in_home_before", ai_pl_in_home_before)

    if dice == 6:
        if state_new[0][config.home_tile] < state_begin[0][config.home_tile]:# and state_new[0][config.home_tile+1] > state_begin[0][config.home_tile+1]:
            reward += 0.25
            if actual_action:
                # print("releases before", config.rewards_detected['piece_release'])
                config.rewards_detected['piece_release'] += 1
                print("releases after", config.rewards_detected['piece_release'])

                # print("state_begin[0]", state_begin[0])
                # print("state_new[0]", state_new[0])
                # print("detected release!\n")
                # time.sleep(3)

                # if config.rewards_detected['piece_release'] >= 10:
                #     print("round now = ", len(config.learning_info_data.whole_list)+1 )
                #     # config.learning_info_data.whole_list['dice_now'], config.learning_info_data.whole_list['dice_now']
                #     print("config.rewards_detected['piece_release']", config.rewards_detected['piece_release'])
                #     print("state_begin[0]", state_begin[0])
                #     print("state_new[0]", state_new[0])
                #     exit("check print release")

    # check the end of the game
    if enemies_already_won:
        reward -= 1
        if actual_action:
            config.rewards_detected['ai_agent_lost'] += 1
    # elif count_pieces_on_tile(player_no=player_i, state=state_new, tile_no=config.finished_tile) == 4:
    elif state_new[player_i][config.finished_tile] == 1:
        # print("player 0 wins the game in this round")
        reward += 1
        if actual_action:
            config.rewards_detected['ai_agent_won'] += 1

    """ furthest piece away """
    # check if moved piece is the furthest away
    furthest_piece, furthest_dist = 0, 0
    # print("pieces_player_now", pieces_player_now)
    for piece in range(len(pieces_player_begin)):
        # print("pieces_player_now[piece]", pieces_player_now[piece])
        if pieces_player_begin[piece] >= furthest_dist and pieces_player_begin[piece] not in config.safe_corridor:
            furthest_dist = pieces_player_begin[piece]
            furthest_piece = piece
    # print("furthest_piece ", furthest_piece)
    if furthest_piece == piece_to_move and furthest_dist != 0:
        # if the piece was in safe zone - smaller reward
        piece_moved_from_safe_zone = False
        for tile_id, value in enumerate(state_diff):
            if value < 0 and tile_id in ids_where_pieces_are_safe:
                piece_moved_from_safe_zone = True
        if piece_moved_from_safe_zone:
            reward += 0.05
            if actual_action:
                config.rewards_detected['move_closest_safe'] += 1
        else:
            if not furthest_dist in config.safe_corridor:
                reward += 0.1
                if actual_action:
                    config.rewards_detected['move_closest_goal'] += 1
        # exit('chosen furthest one')

    """ 
    • -0.25 for getting a piece knocked in the next turn - next turn, not the next state
    for that will need to save the previous move's last state and see difference between state_new and state_begin of new turn 
    save the new state
    """
    player_i = 0
    # pieces_last = count_pieces_on_tile(player_i, config.last_turn_state_new, 0)
    # pieces_now = count_pieces_on_tile(player_i, state_begin, 0)

    pieces_last = config.last_turn_state_new[0][0]
    pieces_now = state_begin[0][0]

    # if pieces_now > pieces_last: #and len(config.learning_info_data.whole_list) >= 2:
    if did_loose_piece(state_begin=state_begin, state_prev_new=config.last_turn_state_new,
                       pieces_player_begin=pieces_player_begin):
        reward -= 0.25
        if actual_action:
            config.rewards_detected['getting_piece_knocked_next_turn'] += 1
            # if config.rewards_detected['getting_piece_knocked_next_turn'] > config.rewards_detected['piece_release']:
            # print("pieces_last", pieces_last, config.last_turn_state_new)
            # print("pieces_now", pieces_now, state_begin)

            # time.sleep(1)
                # exit("can not loose more pieces than released!")
        # print("config.last_turn_state_new \n", config.last_turn_state_new)
        # print("state_begin \n", state_begin)
        # exit("check loosing piece in next round")

    # 0.05 for forming a blockade
    player_i = 0
    # print("begin_state", state_begin[player_i])
    # print("new_state", state_new[player_i])
    if state_begin[player_i][0] != 1:
        # print("state_diff", state_diff)
        for tile_id, value in enumerate(state_diff):
            # when new pawn moved and moved in actual dangerous position
            if value >= 0.25 and tile_id in range(1, 54):
                # if the actual number of the players on the tile increased
                # pieces_there_after = count_pieces_on_tile(player_i, state_new, tile_id)
                pieces_there_after = state_new[player_i][tile_id] * 4

                if pieces_there_after > 1:
                    # if pieces_there_after > count_pieces_on_tile(player_i, state_begin, tile_id):
                    if pieces_there_after > state_begin[player_i][tile_id]:
                        reward += 0.05
                        if actual_action:
                            config.rewards_detected['forming_blockade'] += 1
                        # print("MADE A BLOCKADE, SIR!")
                        #
                        # print("state_begin[0]", state_begin[0])
                        # print("state_new[0]", state_new[0])
                        # exit("blockade check")

                        # if there are enemy pieces in range +- 6 add 0.2 more reward
                        #     • 0.2 for defending a vulnerable piece.
                        if enemy_pieces_nearby(player_id=player_i, state=state_new, horizon=6):
                            reward += 0.15
                            if actual_action:
                                config.rewards_detected['defend_vulnerable'] += 1
                            # print("WE WERE IN DANGER, SIR!")
                            # exit("life saving blockade")

    """ • go on globe (safe) 0.12 """
    for globe_tile in config.globe_tiles:
        # pieces_globe_begin = count_pieces_on_tile(player_i, state_begin, globe_tile)
        # pieces_globe_new = count_pieces_on_tile(player_i, state_new, globe_tile)

        pieces_globe_begin = state_begin[player_i][globe_tile]
        pieces_globe_new = state_new[player_i][globe_tile]

        if pieces_globe_new > pieces_globe_begin:
            reward += 0.12
            if actual_action:
                config.rewards_detected['moved_on_safe_globe'] += 1

    """ • using a star (speed) 0.17 """
    for star_tile in config.star_tiles:
        # pieces_star_begin = count_pieces_on_tile(player_i, state_begin, star_tile)
        # pieces_star_new = count_pieces_on_tile(player_i, state_new, star_tile)

        pieces_star_begin = state_begin[player_i][star_tile]
        pieces_star_new = state_new[player_i][star_tile]

        if pieces_star_new > pieces_star_begin:
            reward += 0.17
            if actual_action:
                config.rewards_detected['speed_boost_star'] += 1


    config.last_turn_state_new = state_new
    # exit("test")
    # return reward, config.rewards_detected
    return reward
