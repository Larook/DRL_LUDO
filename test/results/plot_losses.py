import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('learning_info_data_process.csv')

    # fig = plt.figure(figsize=(3, 6))
    df.plot(y=['loss', 'avg_reward', 'avg_reward_this_epoch'], figsize=(30, 10))
    plt.xlabel('x-axis label')
    plt.ylabel('y-axis label')
    plt.title('batch = ??,  epochs = ??')

    # plt.savefig("20_04.jpg")
    plt.show()
