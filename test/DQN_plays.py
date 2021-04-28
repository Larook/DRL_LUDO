import ludopy

import copy
import math
import random
import time
import unittest
import sys
from collections import namedtuple

import numpy as np
import pandas as pd

from Feedforward import *
from rewards import *
from Memory import *
from Learning_Info import Learning_Info

losses = []

def get_game_state(pieces_seen_from_players):
    """
    state represented by  240 variables - for each player 60
    Each state (id of tile) can have values 0 - 1, where 0 means 0 pawns on the tile, and 1 means 4 pawns on tile

    :param pieces:
    :return:
    """
    # players = pieces[0]
    POSITIONS_PER_PLAYER = 60

    # for every player
    # loop through the pawns
    # save the positions
    # update state of the player
    state_all = np.empty([4, POSITIONS_PER_PLAYER], float)
    for index, pawn_positions in enumerate(pieces_seen_from_players):
        state = np.zeros(POSITIONS_PER_PLAYER)
        for pawn_id in pawn_positions:
            state[pawn_id] += 0.25
        state_all[index] = state

    # print("state_all", state_all)
    return state_all

def get_pawn_id_from_tile(tile_id, player_pieces):
    """
    get the id of a random pawn that is located on the tile_id
    Careful! this id of pawn might not be the movable pawn!!
    """
    # best_pawn_id = 100
    for i, pawn in enumerate(player_pieces):
        if pawn == tile_id:
            best_pawn_id = i
            return best_pawn_id

def get_reshaped_ann_input(begin_state, new_state, action):
    """ save STATE and ACTION into 1-dimensional np.array. This should be an input to a ANN """
    # look for the position of the given pawn before and after a move
    current_player = 0
    input_ann = np.array(begin_state)
    input_ann = input_ann.reshape((240, 1))

    action_tuple = (begin_state[current_player][action] / 60, new_state[current_player][action] / 60)

    # print(input_ann.shape)
    input_ann = np.append(input_ann, action_tuple)
    return input_ann


def get_estimated_Q_ann(model, state_now, action_tuple):
    """
    DQN network to select the action of optimal policy
    Just get the estimated Q from the Network
    :param state_papers_start:
    :return:
    """
    # TODO: flatten everything to 1 tensor
    input_ann = np.array(state_now)
    input_ann = input_ann.reshape((240, 1))
    # print(input_ann.shape)
    input_ann = np.append(input_ann, action_tuple)
    # print(input_ann.shape)

    input_ann = torch.FloatTensor(input_ann)
    # input_ann = torch.FloatTensor(action_tuple)
    Q_pred = model(input_ann)
    print("Q_pred = ", Q_pred)
    # t.max(1) will return largest column value of each row.
    # second column on max result is index of where max element was
    # found, so we pick action with the larger expected reward.
    # Q_pred_max = Q_pred.max(1)[1].view(1, 1)
    # print("Q_pred_max = ", Q_pred_max)
    return Q_pred


# TODO - works but probably too long execution
def get_state_after_action(game, pawn):
    """
    based on game.answer_observation()
    to feed the ANN we need state before and after action
    :param pawn:
    :return: state after action
    """

    # game = copy.copy(game)
    game = copy.deepcopy(game)

    # # Check if there is an observation pending
    # if not game.observation_pending:
    #     raise RuntimeError("There is no pending observation. "
    #                        "There has to be a pending observation has to be answered first")
    # Check if the given piece_to_move is among the current_move_pieces
    if len(game.current_move_pieces) and pawn not in game.current_move_pieces:
        raise RuntimeError("The piece given has to be among the given move_pieces")
    # If it is then move the piece
    elif len(game.current_move_pieces):
        new_enemys = game.players[game.current_player].move_piece(pawn, game.current_dice, game.current_enemys)

        # game.__set_enemy_pieces(game.current_player, new_enemys)  # does the same as lines below
        # Go through the enemies and set the changes in their pieces
        for e_i, e in enumerate(game.enemys_order[game.current_player]):
            game.players[e].set_pieces(new_enemys[e_i])

    # If there was no pieces that could be moved then nothing can be done
    # else:
    #     pass  # This line is present for readability

    # # Check if the player now is the winner
    # player_is_a_winner = game.players[game.current_player].player_winner()
    # if player_is_a_winner:
    #     # Check if player is the first winner
    #     if game.first_winner_was == -1:
    #         game.first_winner_was = game.current_player
    #     # Check if player has been added to game_winners
    #     if game.current_player not in game.game_winners:
    #         game.game_winners.append(game.current_player)

    # next_player = True
    # # In the first round the players has 3 attempts to get a piece out of home
    # if game.round == 1 and \
    #         all(p_piece == 0 for p_piece in game.players[game.current_player].get_pieces()) and \
    #         game.current_start_attempts < 3:
    #     game.current_start_attempts += 1
    #     next_player = False
    # else:
    #     game.current_start_attempts = 0
    # # If it is not in the first round a dice on 6 will give an extra move
    # if game.round != 1 and game.current_dice == 6:
    #     next_player = False

    # # Set the observation pending to false as the last given observation was handled
    # game.observation_pending = False

    # Get the environment after the move
    # after_obs = game.__gen_observation(game.current_player, roll_dice=False)

    player = game.players[game.current_player]
    # Get the pieces that can be moved with the current dice
    move_pieces = player.get_pieces_that_can_move(game.current_dice)
    game.current_move_pieces = move_pieces

    # # Get where the player's pieces are and the enemy's pieces are
    # player_pieces, enemy_pieces = game.get_pieces(game.current_player)
    # game.current_enemys = enemy_pieces

    # # Check if the player is a winner
    # player_is_a_winner = player.player_winner()
    # # Check if there is a winner
    # there_is_a_winner = any([p.player_winner() for p in game.players])

    # after_obs = dice, np.copy(move_pieces), np.copy(player_pieces), np.copy(
    #     enemy_pieces), player_is_a_winner, there_is_a_winner

    return get_game_state(game.get_pieces()[0])


# TODO train the network using the memory
def optimize_model(game, memory, ann_model, available_actions, BATCH_SIZE):

    """ prepeare training data """
    batch = memory.sample(memory.capacity)
    batchLen = len(batch)
    if batchLen < BATCH_SIZE:  # doesnt train when not enough samples in memory
        return 0

    # get the ANN inputs from batch
    ann_inputs, calculated_rewards = [], []
    for obs in batch:
        ann_inputs.append(get_reshaped_ann_input(begin_state=obs[0], new_state=obs[3], action=obs[1]))
        calculated_rewards.append(obs[2])

    ann_inputs = torch.tensor(ann_inputs).float()
    predicted_q = ann_model(ann_inputs)  # holds predictions for the states for each sample and will be used as a default target in the learning

    x = np.zeros((batchLen, ann_model.input_size))
    y = np.zeros((batchLen, 1))  # only one output

    for i in range(batchLen):
        obs = batch[i]
        s_ = obs[3]
        GAMMA = 0.95
        t = predicted_q[i] + GAMMA * get_max_reward_from_state(game, s_, available_actions)  # target
        x[i] = ann_inputs[i]  # state
        y[i] = t.detach().numpy()  # target - estimation of the Q(s,a) - if estimation is good -> close to the Q*(s,a)

    """ train the ann https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62 """
    optimizer = torch.optim.Adam(ann_model.parameters())  # Optimizers help the model find the minimum.
    losses_this_action = []
    for i in range(batchLen):
        output = ann_model(x[i])
        true_q_val = torch.tensor(y[i]).float()
        loss = F.smooth_l1_loss(output, true_q_val)  # L1 loss for regression applications
        loss.backward()

        optimizer.step()

        global losses
        losses.append({'loss': loss.item()})
        losses_this_action.append(loss.item())
        # print("loss.item() = ", loss.item())


    #     if batch_num % 40 == 0:
    #         print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
    # print('Epoch %d | Loss %6.2f' % (epoch, sum(losses) / len(losses)))
    loss_avg = np.mean(losses_this_action)
    return loss_avg


def action_selection(game, move_pieces, ann_model, begin_state, steps_done, is_random=False, show=False, exploit_model=False):
    """
    Use MLP to get the Q_value of chosen action and state
    choose the best action and return the new_state
    :param move_pieces:
    :param ann_model:
    :param begin_state:
    :param steps_done:
    :return:
    """
    # steps_done for 10 games ~= 3380
    EPS_START = 0.9
    EPS_END = 0.05
    # EPS_DECAY = 1000  # after 10 games eps_threshold=0.0789
    EPS_DECAY = 18000  # after 10 games eps_threshold=0.754 -> after 100 games: 0.17999013613686377 and reaches EPS_END after 1000 plays

    if len(move_pieces):
        action_q_l = []
        best_Q_val = -99999999999999

        for possible_action in move_pieces:
            # get piece with best Q
            before_new_state_t = time.time()
            new_state = get_state_after_action(game, possible_action)
            # print("<timing> get_state_after_action =", time.time() - before_new_state_t)

            input_ann = get_reshaped_ann_input(begin_state, new_state, possible_action)
            input_ann = torch.FloatTensor(input_ann)

            with torch.no_grad():
                Q_est = ann_model(input_ann)

            action_q_l.append((possible_action, Q_est))

        # epsilon greedy
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        # print("eps_threshold = ", eps_threshold)
        steps_done += 1

        if sample > eps_threshold or exploit_model:
            # choose best pawn
            for action, Q_v in action_q_l:
                if Q_v > best_Q_val:
                    piece_to_move = action
            if show:
                print("\tbest action selected")
            if is_random:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn

        else:
            # get random pawn
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
            new_state = get_state_after_action(game, piece_to_move)
            if show:
                print("\trandom action selected")

    else:
        piece_to_move = -1
        new_state = get_state_after_action(game, piece_to_move)
        if show:
            print("\tthe only possible action selected")

    return piece_to_move, new_state


def dqn_approach(do_random_walk, load_model, train, use_gpu):

    ai_agents = [0]  # which id of player should be played by ai?
    g = ludopy.Game()

    # load/create model of ANN

    if load_model:
        ann_model = Feedforward(try_cuda=use_gpu, input_size=242, hidden_size=21)

        checkpoint = torch.load('results/models/running/model_test_48_epochs.pth')
        ann_model.load_state_dict(checkpoint)
    else:
        ann_model = Feedforward(try_cuda=use_gpu, input_size=242, hidden_size=21)
    if not train:
        ann_model.model.eval()

    BATCH_SIZE = 600  # 1000
    if do_random_walk:
        BATCH_SIZE = 10000  # 1000

    memory = Memory(BATCH_SIZE)
    learning_info_data = Learning_Info()

    # counter info
    won_counter = 0
    time_epochs = []
    rewards_info = []
    avg_time_epoch = 0
    avg_time_left = 0

    epochs = 1000
    steps_done = 0
    for epoch in range(1, epochs):
        time_turns = []
        time_epoch_start = time.time()
        avg_time_turn = 0

        """ main game loop """
        ai_player_seen_end = False
        reward = 0

        while not ai_player_seen_end:
            time_turn_start = time.time()

            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
             there_is_a_winner), player_i = g.get_observation()
            pieces = g.get_pieces()

            if player_i not in ai_agents:
                if len(move_pieces) > 0:
                    action = move_pieces[np.random.randint(0, len(move_pieces))]  # randomly moves a pawn
                else:
                    action = -1
                new_state = get_state_after_action(g, action)
            else:
                """ select an action of AI player """
                reward = 0
                t_get_game_state = time.time()
                begin_state = get_game_state(pieces[player_i])
                # print("<timing> t_get_game_state =", time.time()-t_get_game_state)

                t_action_selection = time.time()
                action, new_state = action_selection(g, move_pieces, ann_model, begin_state, steps_done, is_random=do_random_walk, show=False, exploit_model=load_model)
                # print("<timing> t_action_selection =", time.time()-t_action_selection)

                t_get_reward = time.time()
                reward = get_reward(begin_state, action, new_state, pieces[player_i][player_i])  # immediate reward
                # print("<timing> t_get_reward =", time.time()-t_get_reward)

                # save round observation to the memory
                memory.add((begin_state, action, reward, new_state))

                """ perform one step of optimization with random batch from memory == TRAIN network """
                # if not use_model:
                if train or not load_model:
                    t_optimize_model = time.time()
                    loss_avg = optimize_model(g, memory, ann_model, move_pieces, BATCH_SIZE)
                else:
                    loss_avg = 0
                # print("<timing> t_optimize_model =", time.time()-t_optimize_model)
                rewards_info.append(reward)

                if player_i in ai_agents:
                    if any(count_pieces_on_tile(player_no=player_id, state=new_state, tile_no=59) == 4 for player_id in range(0,4)):
                        ai_player_seen_end = True

            """ perform action and end round """
            _, _, _, _, player_is_a_winner, there_is_a_winner = g.answer_observation(action)
            if reward >= 1:
                if player_i in ai_agents:
                    won_counter += 1
                    print("<!!!> player_ai won!! won_counter=", won_counter)
                    if do_random_walk:
                        exit("works! trying to reduce time of get_state_after_action()")

            steps_done += 1
            if player_i in ai_agents:
                avg_reward = np.array(rewards_info).mean()
                learning_info_data.append(epoch_no=epoch, epochs_won=won_counter, ai_player_i=player_i,
                                          action_no=steps_done, begin_state=begin_state, dice_now=dice, action=action,
                                          new_state=new_state, reward=reward, avg_reward=avg_reward, loss=loss_avg)

            time_turn_end = time.time()
            time_turns.append(time_turn_end - time_turn_start)
            avg_time_turn = np.mean(time_turns)
            if steps_done % 10 == 0:
                print("epoch = %d | round = %d "
                      "<avg_time_left = %.2f avg_time_epoch = %.2f | avg_time_turn = %.2f> "
                      "| won_counter = %d | steps_done = %d | action = %d | avg_reward = %f, loss_avg = %f" %
                      (epoch, g.round, avg_time_left, avg_time_epoch, avg_time_turn, won_counter, steps_done, action, avg_reward, loss_avg))
        time_epoch_end = time.time()
        time_epochs.append(time_epoch_end-time_epoch_start)
        avg_time_epoch = np.mean(time_epochs)

        # save results after each epoch
        learning_info_data.save_to_csv('results/learning_info_data_process.csv', epoch_no=epoch)
        learning_info_data.save_plot_progress(bath_size=BATCH_SIZE, epoch_no=epoch, is_random_walk=do_random_walk)

        if epoch % 3 == 0 and not load_model:
            print("saving ann model")
            torch.save(ann_model.state_dict(), 'results/models/running/model_test_'+str(epoch)+"_epochs.pth")

        if won_counter % 2 == 0 and won_counter > 0:
            g.save_hist("results/videos_history/game_history.npy")
            g.save_hist_video("results/videos_history/game_ANN_last_win_3_won.mp4")

        # restart the game after each epoch
        ai_player_seen_end = False
        g.reset()
        g = ludopy.Game()
        avg_time_left = (epochs - epoch) * avg_time_epoch

    df_losses = pd.DataFrame.from_records(losses)
    df_losses.to_csv('results/losses.csv')
    learning_info_data.save_to_csv('results/learning_info_data_x.csv', epoch_no=epoch)
    learning_info_data.save_plot_progress(bath_size=BATCH_SIZE, epoch_no=epoch)

    # Save history and ANN model
    if not load_model:
        print("saving ann model")
        torch.save(ann_model.state_dict(), 'results/models/model_final.pth')
    print("Saving history to numpy file")
    g.save_hist("videos_history/game_history.npy")
    print("Saving game video")
    g.save_hist_video("videos_history/game_ANN_test.mp4")


if __name__ == '__main__':
    # unittest.main()
    dqn_approach(do_random_walk=False, load_model=True, train=True, use_gpu=False)
