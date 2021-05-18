from DQN_plays import dqn_approach


if __name__ == '__main__':
    # evaluation of only human leanred model
    dqn_approach(do_random_walk=False, load_model=True, train=False, start_with_human_model=True, use_gpu=False)