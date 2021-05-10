import math

from matplotlib import pyplot as plt

if __name__ == "__main__":
    steps_per_10_games = 3380
    EPS_START = 0.9
    EPS_END = 0.05
    # EPS_DECAY = 1000  # after 10 games eps_threshold=0.0789
    EPS_DECAY = 10000  # after 10 games eps_threshold=0.053
    # EPS_DECAY = 18000  # after 10 games eps_threshold=0.754 -> after 100 games: 0.17999013613686377 and reaches EPS_END after 1000 plays

    games_played = 100
    steps_done = games_played * steps_per_10_games / 10
    ys = []
    for steps_done in list(range(int(steps_done))):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.7 * steps_done / EPS_DECAY)
        ys.append(eps_threshold)

    # plt.plot(y=ys, x=steps_done/steps_per_10_games/games_played)
    plt.plot(ys)
    plt.axvline(x=steps_per_10_games*1, color='r', label='axvline - full height')
    plt.axvline(x=steps_per_10_games*5, color='b', label='axvline - full height')

    print("after 10 games: epsilon_now=", ys[steps_per_10_games*1])
    print("after 20 games: epsilon_now=", ys[steps_per_10_games*2])
    print("after 30 games: epsilon_now=", ys[steps_per_10_games*3])
    print("after 40 games: epsilon_now=", ys[steps_per_10_games*4])
    print("after 50 games: epsilon_now=", ys[steps_per_10_games*5])

    plt.show()