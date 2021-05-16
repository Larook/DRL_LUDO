import datetime
import os
import re

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

import config
import json
import ast

def get_fixed_row_from_str(row_d):
    """ problem is that csv is saved as string - need to get list from the string """
    for key in row_d:
        print(key, '->', type(row_d[key]), row_d[key])

        if key == 'reward':
            row_d[key] = float(row_d[key].strip('()').split(',')[0])
            print("type(row_d[key]) = ", type(row_d[key]), "row_d[key] ", row_d[key])
            # exit("check reward")

        else:

            if type(row_d[key]) == str:

                if len(row_d[key]) < 4:
                    row_d[key] = int(row_d[key])

                else:
                    print("fix str to list", type(row_d[key]), row_d[key])
                    a = row_d[key]
                    print("a", a)
                    # a = a.replace('[', '').replace(']', '')
                    # a_l = a.split(' ')
                    a = a.replace(' ', ',').replace('[ ','[').replace('[,','[').replace('\n','').replace('.]', ']').replace(',,',',').replace('.,',',').replace(',,',',').replace(',,,',',').replace(',]',']').replace(',,',',').replace(',]',']')
                    print("key = ", key, "a", a)

                    a_l = json.loads(a)

                    print("type(a_l) = %s a_l = %s" % (type(a_l), a_l))
                    # new_l = []
                    # for a_i in a_l:
                    #     print("a_i = ", a_i)
                    #
                    #     if type(a_i) != str:
                    #         new_l.append(float(a_i))
                    #         # pass
                    #     elif len(a_i) != 0:
                    #         if a_i.strip().endswith('.'):
                    #             a_i = a_i[:-1]
                    #         a_i = a_i.replace('\n', '')
                    #         a_i = a_i.strip()
                    #         # print('a_i "', a_i, '"', len(a_i))
                    #         if a_i.endswith('.'):
                    #             a_i = a_i[:-1]
                    #         a_i = a_i.strip()
                    #         if len(a_i) != 0:
                    #             new_l.append(float(a_i))
                    # print("new_l", new_l)
                    # # new_list =
                    # row_d[key] = new_l



                    row_d[key] = np.array(a_l)
                    # if key == "state_begin":
                    #     exit()
    return row_d


def pretrain_model(q_net, data):
    batch_size = 100
    # batch = data.sample(batch_size)
    random.shuffle(data)
    batch = data[1:batch_size]

    # get the ANN inputs from batch
    ann_inputs, calculated_rewards = [], []
    for obs in batch:
        print("obs", obs)
        ann_inputs.append(obs['ann_input'])
        # reward = float(obs['reward'].strip('()').split(',')[0])
        reward = float(obs['reward'][0])
        calculated_rewards.append(reward)

    ann_inputs = torch.tensor(ann_inputs).float()

    # holds predictions for the states for each sample and will be used as a default target in the learning
    predicted_q = q_net(ann_inputs)

    x = np.zeros((batch_size, q_net.input_size))
    y = np.zeros((batch_size, 1))  # only one output

    for i in range(len(batch)):
        obs = batch[i]
        print("obs_predicting", obs)

        s_ = obs['state_new']
        dice = obs['dice']
        pieces_player_begin = obs['pieces_player_begin']
        possible_actions = obs['available_actions']

        t = predicted_q[i] + config.GAMMA * get_max_reward_from_state(pieces_player_begin=pieces_player_begin, dice=dice, state_begin=s_, possible_actions=possible_actions)  # target
        x[i] = ann_inputs[i]  # state
        y[i] = t.detach().numpy()  # target - estimation of the Q(s,a) - if estimation is good -> close to the Q*(s,a)

    """ train the ann https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62 """
    optimizer = torch.optim.Adam(q_net.parameters())  # Optimizers help the model find the minimum.
    losses_this_action = []
    for i in range(batch_size):
        output = q_net(x[i])
        true_q_val = torch.tensor(y[i]).float()
        loss = F.smooth_l1_loss(output, true_q_val)  # L1 loss for regression applications
        loss.backward()

        optimizer.step()


    # save the model
    now = datetime.datetime.now()
    torch.save(q_net.state_dict(), 'results/models/pretrained_human_data_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) +  "_epochs.pth")
    pass


if __name__ == "__main__":
    whole_dataset = []

    # read all the csv and save it to one list
    for file in os.listdir('human_data'):
        print("game_data", file)
        if file.endswith('.csv'):
            df = pd.read_csv('human_data/' + file)
            df_dict_l = df.to_dict('records')
            for row_d in df_dict_l:
                row_d_lists = get_fixed_row_from_str(row_d)
                whole_dataset.append(row_d_lists)
            # print(whole_dataset)
            # exit()

    # for every row calculate the reward (reward = y)
    dataset_with_rewards = []
    for row_d in whole_dataset:
        print('row = ', row_d)
        ann_input = row_d['ann_input']
        # reward = get_reward(state_begin=row_d['begin_state'], piece_to_move=row_d['action'], state_new=row_d['new_state'], pieces_player_begin=row_d['pieces_player_begin'], actual_action=False)
        reward = get_reward(dice=row_d['dice'], state_begin=row_d['state_begin'], piece_to_move=row_d['action'], state_new=row_d['state_new'], pieces_player_begin=row_d['pieces_player_begin'], actual_action=False)
        row_d['reward'] = reward
        dataset_with_rewards.append(row_d)
        # exit()
    df_with_rewards = pd.DataFrame(dataset_with_rewards)
    df_with_rewards.to_csv('human_data/df_with_rewards.csv')

    # train the neural net
    # create new model of DQN
    q_net = Feedforward(try_cuda=False, input_size=242, hidden_size=21)
    q_net_trained = pretrain_model(q_net, data=dataset_with_rewards)






