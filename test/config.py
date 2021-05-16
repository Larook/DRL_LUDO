from Learning_Info import Learning_Info


def init_rewards_couter_dict():
    return {'piece_release': 0, 'defend_vulnerable': 0, 'knock_opponent': 0,
                    'move_closest_goal': 0, 'move_closest_safe': 0, 'forming_blockade': 0,
                    'getting_piece_knocked_next_turn': 0,
                    'moved_on_safe_globe': 0,
                    'speed_boost_star': 0,
                    'ai_agent_won': 0, 'ai_agent_lost': 0,
            }

def init_start_state():
    start_state = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]
    return start_state


learning_info_data = Learning_Info()

last_turn_state_new = init_start_state()

epsilon_now = 0
rewards_detected = init_rewards_couter_dict()

""" tiles of the map """
home_tile = 0
finished_tile = 59

safe_corridor = [54, 55, 56, 57, 58, 59]

globe_tiles = [9, 14, 22, 35, 48]
star_tiles = [5, 12, 18, 25, 31, 38, 44, 51]

""" training the net """
GAMMA = 0.95