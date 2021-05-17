from DQN_plays import dqn_approach


if __name__ == '__main__':
    # evaluation
    dqn_approach(do_random_walk=False, load_model=True, train=False, use_pretrained=True, use_gpu=False)