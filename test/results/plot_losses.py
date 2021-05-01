import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('learning_info_data_process.csv')

    fig, axes = plt.subplots(nrows=2, ncols=1)

    df.plot(y=['loss', 'avg_reward', 'avg_reward_this_epoch'], figsize=(30, 10), ax=axes[0])
    # plt.xlabel('x-axis label')
    # plt.ylabel('y-axis label')
    # plt.title('batch = ??,  epochs = ??')

    df.plot(y=['piece_release', 'knock_opponent', 'move_closest_goal', 'move_closest_safe', 'forming_blockade',
               'defend_vulnerable', 'getting_piece_knocked_next_turn'], figsize=(30, 10), ax=axes[1])


    # plt.savefig("20_04.jpg")
    plt.show()
