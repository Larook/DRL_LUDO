import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Learning_Info():
    # data_df

    def __init__(self):
        self.whole_list = []
        self.rewards_each_epoch = []

    def update(self, epoch_no, round_no, epochs_won, action_no, ai_player_i, begin_state, dice_now, action, new_state,
               reward, avg_reward, loss, rewards_info, epsilon_now):
        avg_reward_this_epoch = 0
        self.rewards_each_epoch.append(reward)
        winrate = epochs_won/epoch_no

        self.whole_list.append({'epoch_no': epoch_no, 'round': round_no, 'epochs_won': epochs_won, 'winrate': winrate,
                                'action_no': action_no,
                                'ai_player_i': ai_player_i, 'begin_state': begin_state, 'dice_now': dice_now,
                                'action': action, 'new_state': new_state, 'reward': reward, 'avg_reward': avg_reward,
                                'avg_reward_this_epoch': np.array(self.rewards_each_epoch).mean(),
                                'loss': loss,
                                'piece_release': rewards_info['piece_release'],
                                'knock_opponent': rewards_info['knock_opponent'],
                                'move_closest_goal': rewards_info['move_closest_goal'],
                                'move_closest_safe': rewards_info['move_closest_safe'],
                                'forming_blockade': rewards_info['forming_blockade'],
                                'defend_vulnerable': rewards_info['defend_vulnerable'],
                                'getting_piece_knocked_next_turn': rewards_info['getting_piece_knocked_next_turn'],
                                'moved_on_safe_globe': rewards_info['moved_on_safe_globe'],
                                'speed_boost_star': rewards_info['speed_boost_star'],
                                'ai_agent_won': rewards_info['ai_agent_won'],
                                'ai_agent_lost': rewards_info['ai_agent_lost'],
                                'epsilon_now': epsilon_now
                                })

    def save_to_csv(self, path, epoch_no):
        self.data_df = pd.DataFrame(self.whole_list)
        self.data_df.to_csv(path)
        self.rewards_each_epoch = []  # clean the table after the epoch ends and data is saved

    def save_plot_progress(self, bath_size, epoch_no, is_random_walk):

        fig, axes = plt.subplots(nrows=4, ncols=1)
        title = "batch = " + str(bath_size) + ", epochs = " + str(epoch_no)
        if is_random_walk:
            title = "randW_" + title
        plt.title(title)

        self.data_df.plot(y=['loss', 'avg_reward', 'avg_reward_this_epoch'], figsize=(30, 10), ax=axes[0])
        self.data_df.plot(y=['piece_release', 'knock_opponent', 'move_closest_goal', 'move_closest_safe', 'forming_blockade',
                   'defend_vulnerable', 'getting_piece_knocked_next_turn', 'ai_agent_won', 'ai_agent_lost'], figsize=(30, 10), ax=axes[1])
        winr = self.data_df.plot(y=['winrate', 'epsilon_now'], figsize=(30, 10), ax=axes[2])

        winr.hlines(0.25, winr.get_xticks().min(), winr.get_xticks().max(), linestyle='--', color='pink')
        self.data_df.plot(y=['epoch_no'], figsize=(30, 10), ax=axes[3])

        plt.savefig("results/plots/" + title + ".jpg")


