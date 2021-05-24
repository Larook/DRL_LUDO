import math

from Learning_Info import Learning_Info


""" tiles of the map """
home_tile = 0
finished_tile = 59

safe_corridor = [54, 55, 56, 57, 58, 59]

globe_tiles = [9, 14, 22, 35, 48]
star_tiles = [5, 12, 18, 25, 31, 38, 44, 51]



def get_epsilon_greedy(steps_done):
    """ when 1200 batch size then the net starts to be trained after epoch 12 """
    # steps_per_10_games = 3380
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 10000  # after 10 games eps_threshold=0.053

    steps_training_starts_after_1200_batches = 5100
    if steps_done <= steps_training_starts_after_1200_batches:
        eps_threshold = EPS_START
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-0.6 * (steps_done-steps_training_starts_after_1200_batches) / EPS_DECAY)  # platou after 200 games
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.4 * (steps_done-steps_training_starts_after_1200_batches) / EPS_DECAY)  # platou after 100 games
    # old         eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.7 * steps_done / EPS_DECAY)
    return eps_threshold


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


""" training the net """
batch_size = 1200
# batch_size = 100
epochs = 250
GAMMA = 0.95  # discount

network_sync_counter = 0
network_sync_freq = 500
# learning_rate_mlp = 5e-3
learning_rate_mlp = 1e-2  # bigger one
loss_avg_running_list = []

learning_info_data = Learning_Info()
last_turn_state_new = init_start_state()

steps_done = 0
epsilon_now = 0
rewards_detected = init_rewards_couter_dict()

""" human network pretrain """
losses_pretrain = []
# epochs_pretrain = 200
epochs_pretrain = 1000
pretrain_batch_size = 50
learning_rate_pretrain = 0.1  # big one

""" evaluation """
epochs_evaluate = 500
# epochs_evaluate = 5

