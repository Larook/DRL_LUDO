import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('learning_info_data_process.csv')

    fig, axes = plt.subplots(nrows=3, ncols=1)
    plt.title('batch = ??,  epochs = ??')

    df.plot(y=['loss', 'avg_reward', 'avg_reward_this_epoch'], figsize=(30, 10), ax=axes[0])
    # plt.xlabel('x-axis label')
    # plt.ylabel('y-axis label')

    df.plot(y=['piece_release', 'knock_opponent', 'move_closest_goal', 'move_closest_safe', 'forming_blockade',
               'defend_vulnerable', 'getting_piece_knocked_next_turn'], figsize=(30, 10), ax=axes[1])
    winr = df.plot(y=['winrate'], figsize=(30, 10), ax=axes[2])
    # winr.hlines(0.25, axes.get_xticks().min(), axes.get_xticks().max(), linestyle='--', color='pink')
    winr.hlines(0.25, winr.get_xticks().min(), winr.get_xticks().max(), linestyle='--', color='pink')


    # plt.savefig("20_04.jpg")
    plt.show()
