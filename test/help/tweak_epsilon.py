import math
import random
import sys

from matplotlib import pyplot as plt
sys.path.append('../')
import config


def plot_epsilon_over_epochs():
    steps_per_10_games = 3380
    games_played = 101

    steps_done = games_played * steps_per_10_games / 10
    print("steps_done", steps_done)
    ys = []
    for steps_done in list(range(int(steps_done))):
        eps_threshold = config.get_epsilon_greedy(steps_done)
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.7 * steps_done / EPS_DECAY)
        ys.append(eps_threshold)


    # plt.plot(y=ys, x=steps_done/steps_per_10_games/games_played)
    plt.plot(ys)
    plt.axvline(x=steps_per_10_games * 1, color='r', label='axvline - full height')
    plt.axvline(x=steps_per_10_games * 5, color='b', label='axvline - full height')

    print("after 10 games: epsilon_now=", ys[steps_per_10_games * 1])
    # print("after 20 games: epsilon_now=", ys[steps_per_10_games * 2])
    # print("after 30 games: epsilon_now=", ys[steps_per_10_games * 3])
    # print("after 40 games: epsilon_now=", ys[steps_per_10_games * 4])
    print("after 100 games: epsilon_now=", ys[steps_per_10_games * 10])

    plt.show()


def check_epsilon():
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 10000
    steps_per_10_games = 3380
    steps_done = steps_per_10_games * 2

    best_counter = 0
    random_counter = 0
    for i in range(1000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.7 * steps_done / EPS_DECAY)
        print("sample = %f  |   eps_threshold = %f" % (sample, eps_threshold))

        if sample > eps_threshold:
            print("choose best action!")
            best_counter += 1
        else:
            print("choose random action")
            random_counter += 1

    print("random_counter = %d  |   best_counter = %d" % (random_counter, best_counter))


if __name__ == "__main__":
    plot_epsilon_over_epochs()
    # check_epsilon()
