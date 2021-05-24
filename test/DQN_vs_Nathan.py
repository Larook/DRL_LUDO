import pandas as pd

from DQN_plays import *
from Nathan.Qlearn import *
# from Nathan.Qlearn import Qplayer

players_win_counter = {0:0, 1:0, 2:0, 3:0}
players_win_list = []


def dqn_competition_vs_Nathan(do_random_walk, load_model, train, start_with_human_model, use_gpu):

    ai_agents = [0]  # which id of player should be played by ai?
    g = ludopy.Game()

    # load/create model of ANN

    """ q_net - network to get the Q of the current state """
    if load_model:
        q_net = Feedforward(try_cuda=use_gpu, input_size=242, hidden_size=21)

        if not start_with_human_model:
            # checkpoint = torch.load('results/models/model_test_48_epochs_2nets.pth')
            # checkpoint = torch.load('results/models/model_test_21_epochs_all_rewards.pth')
            # checkpoint = torch.load('results/models/model_test_99_epochs_batch_600.pth')
            # checkpoint = torch.load('results/models/model_final_epochs100_batch1200.pth')
            # checkpoint = torch.load('results/models/model_test_294_epochs_batch1200_games3.pth')
            # checkpoint = torch.load('results/models/model_test_282_epochs_4pretrained_new_ann_input.pth')
            checkpoint = torch.load('results/models/model_final_training_new_params.pth')

            print("will use the trained model")
            epoch_last = 282

        if start_with_human_model:
            # checkpoint = torch.load('results/models/pretrained_human_data_13_21_26_epochs.pth')
            # checkpoint = torch.load('results/models/pretrained_human_data_16_13_25_epochs.pth')
            # checkpoint = torch.load('results/models/pretrained_human_data_17_22_44_epochs.pth')
            # checkpoint = torch.load('results/models/pretrained_human_data_19_22_53_epochs_new_ann_input.pth')
            # checkpoint = torch.load('results/models/pretrained_human_data_19_23_5_epochs_new_ann_input.pth')

            # after fixing the networks update frequency
            checkpoint = torch.load('results/models/pretrained_human_epochs200_lr0.1__21_15_37.pth')
            print("will use human pretrained data")
            epoch_last = 1

        q_net.load_state_dict(checkpoint)
        q_net.eval()
    else:
        q_net = Feedforward(try_cuda=use_gpu, input_size=242, hidden_size=21)
        q_net.eval()
    if not train:
        epoch_last = 1

    """ target_net - this target network is used to generate target values or ground truth. 
    The weights of this network are held fixed for a fixed number of training steps after which these are updated with the weight of Main Network. 
    In this way, the distribution of our target return is also held fixed for some fixed iterations which increase training stability. """
    target_net = copy.deepcopy(q_net)
    target_net.train()


    # BATCH_SIZE = 600  # first try
    BATCH_SIZE = config.batch_size  # second try overnight
    if do_random_walk:
        BATCH_SIZE = 10000  # 1000

    memory = Memory(BATCH_SIZE)

    # counter info
    won_counter = 0
    time_epochs = []
    rewards_accumulated = []
    avg_time_epoch = 0
    avg_time_left = 0
    loss_avg = 0

    epochs = config.epochs + 1
    if not train:
        # for evaluation of model just play 200 times
        print('evaluation mode')
        epochs = config.epochs_evaluate
    else:
        print('training mode')
    config.steps_done = 0
    time.sleep(1)

    if load_model:
        epochs_elapsed = range(epoch_last, epochs)
    else:
        epochs_elapsed = range(1, epochs)

    """ QPLAYER """
    playerNathan = 2
    player_QN = Qplayer()
    Nathan_won_counter = 0


    for epoch in range(1, config.epochs_evaluate+1):
        time_turns = []
        time_epoch_start = time.time()

        """ main game loop """
        ai_player_seen_end = False
        there_is_a_winner = False
        config.last_turn_state_new = config.init_start_state()
        reward = 0

        if train:
            end_condition = ai_player_seen_end
        else:
            end_condition = there_is_a_winner

        while not end_condition:
            end_condition = there_is_a_winner
            time_turn_start = time.time()

            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

            # debug purposes only
            a_observation = {"player_i": player_i, "dice": dice, "move_pieces": move_pieces, "player_pieces": player_pieces, "enemy_pieces": enemy_pieces, "player_is_a_winner": player_is_a_winner, "there_is_a_winner": there_is_a_winner}

            if player_i not in ai_agents:

                """ Q learning move """
                if player_i == playerNathan:
                    player_QN.nextmove(player_i, player_pieces, enemy_pieces, dice, move_pieces)
                    action = player_QN.piece

                else:
                    """ enemy move """
                    if len(move_pieces) > 0:
                        action = perform_random_action(move_pieces)
                    else:
                        action = -1

                # state_begin = get_game_state(g.get_pieces()[0])
                # state_new = get_state_after_action(pieces_player_begin, state_begin, dice, action)
                state_new = get_state_after_action_g(g, action)

            else:
                """ select an action of AI player """
                # pieces = g.get_pieces()
                pieces_on_board = g.get_pieces()[player_i]
                # print("pieces_on_board", pieces_on_board)
                pieces_player_begin = pieces_on_board[player_i]
                # print("pieces_player_begin", pieces_player_begin)

                state_begin = get_game_state(pieces_on_board)
                if config.steps_done == 0:
                    config.last_turn_state_new = state_begin

                action, state_new, reward, rewards_accumulated, loss_avg = dqn_agent_action_selection(
                                                                            player_i=player_i,
                                                                            pieces_player_begin=pieces_player_begin,
                                                                            state_begin=state_begin,
                                                                            dice=dice,
                                                                            move_pieces=move_pieces,
                                                                            q_net=q_net,
                                                                            target_net=target_net,
                                                                            memory=memory,

                                                                            BATCH_SIZE=BATCH_SIZE,
                                                                            train=train,
                                                                            load_model=load_model,

                                                                            rewards_accumulated=rewards_accumulated
                                                                            )
                # reward = rewards_accumulated[-1]

                if player_i in ai_agents:
                    # if any(count_pieces_on_tile(player_no=player_id, state=state_new, tile_no=59) == 4 for player_id in range(0,4)):
                    if config.rewards_detected['ai_agent_lost'] > 0:
                        # if any(state_new[player_i][config.finished_tile] == 1 for player_id in range(0,4)):
                        ai_player_seen_end = True
                        print("ai_player_seen_end and lost!")

            """ perform action and end round """
            _, _, _, _, player_is_a_winner, there_is_a_winner = g.answer_observation(action)

            """ save information which player won """
            if there_is_a_winner:
                if player_is_a_winner:
                    if player_i == playerNathan:
                        Nathan_won_counter += 1
                if player_is_a_winner:
                    players_win_counter[player_i] += 1
                    players_win_list.append(players_win_counter)


            if config.rewards_detected['ai_agent_won'] > 0:
                if player_i in ai_agents:
                    won_counter += 1
                    print("<!!!> player_ai won!! won_counter=", won_counter)
                    ai_player_seen_end = True
                    # if do_random_walk:
                    #     exit("works! trying to reduce time of get_state_after_action()")




            config.steps_done += 1
            if player_i in ai_agents:
                avg_reward = np.array(rewards_accumulated).mean()
                config.learning_info_data.update(epoch_no=epoch, round_no=g.round, epochs_won=won_counter, ai_player_i=player_i,
                                                 action_no=config.steps_done, begin_state=state_begin, dice_now=dice, action=action,
                                                 new_state=state_new, reward=reward, avg_reward=avg_reward, loss=loss_avg,
                                                 rewards_occurrences=config.rewards_detected, epsilon_now=config.epsilon_now)

            time_turn_end = time.time()
            time_turns.append(time_turn_end - time_turn_start)
            avg_time_turn = np.mean(time_turns)
            # if steps_done % 10 == 0:
            if player_i == 0:
                print("epoch = %d | round = %d <avg_time_left = %.2f avg_time_epoch = %.2f | avg_time_turn = %.2f> "
                      "| won_counter = %d | steps_done = %d | action = %d | avg_reward = %f, loss_avg = %f "
                      "| epsilon = %f" % (epoch, g.round, avg_time_left, avg_time_epoch, avg_time_turn, won_counter,
                                          config.steps_done, action, avg_reward, loss_avg, config.epsilon_now))
                print("players_win_counter = ", players_win_counter, "    Nathan_won_counter", Nathan_won_counter)

            if len(players_win_list):
                df_wins = pd.DataFrame(players_win_list)
                df_wins.columns = ["player_0_DRL", "player_1_random", "player_2_Qlearning", "player_3_random"]
                df_wins.to_csv("results/PLAYERS_WON_COUNTER_FINAL_EVALUATION.csv")

        time_epoch_end = time.time()
        time_epochs.append(time_epoch_end-time_epoch_start)
        avg_time_epoch = np.mean(time_epochs)

        # save results after each epoch
        csv_name = 'results/learning_info_data_process'
        if not train:
            csv_name += '_evaluate_model'
        config.learning_info_data.save_to_csv(csv_name + '.csv', epoch_no=epoch)
        config.learning_info_data.save_plot_progress(bath_size=BATCH_SIZE, epoch_no=epoch, is_random_walk=do_random_walk)

        if epoch % 4 == 0 and train:
            print("saving ann model")
            torch.save(q_net.state_dict(), 'results/models/running/model_test_'+str(epoch)+"_epochs.pth")

        # if (won_counter % 10 == 0 and won_counter > 0 and not train) or (won_counter % 2 == 0 and won_counter > 0 and train):
        #     g.save_hist("results/videos_history/game_history.npy")
        #     g.save_hist_video("results/videos_history/game_ANN_last_win_3_won.mp4")

        # restart the game after each epoch
        ai_player_seen_end = False
        g.reset()
        g = ludopy.Game()
        avg_time_left = (epochs - epoch) * avg_time_epoch

        # reset rewards detected after epoch
        rewards_detected_reset()

    config.learning_info_data.save_to_csv('results/learning_info_data_x.csv', epoch_no=epoch)
    config.learning_info_data.save_plot_progress(bath_size=BATCH_SIZE, epoch_no=epoch, is_random_walk=do_random_walk)

    # Save history and ANN model
    if train:
        print("saving ann model")
        torch.save(q_net.state_dict(), 'results/models/running/model_final.pth')
    print("Saving history to numpy file")
    g.save_hist("results/videos_history/game_history.npy")
    # print("Saving game video")
    # g.save_hist_video("results/videos_history/game_ANN_test.mp4")

    return players_win_list



if __name__ == '__main__':

    players_win_list = dqn_competition_vs_Nathan(do_random_walk=False, load_model=True, train=False, start_with_human_model=True, use_gpu=False)

    # # save results to csv!
    # df_wins = pd.DataFrame(players_win_list)
    # df_wins.columns = ["player_0_DRL", "player_1_random", "player_2_Qlearning", "player_3_random"]
    # df_wins.to_csv("results/PLAYERS_WON_COUNTER_FINAL_EVALUATION.csv")