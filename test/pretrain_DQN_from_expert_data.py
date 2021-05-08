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
        if type(row_d[key]) == str:

            if len(row_d[key]) < 4:
                row_d[key] = int(row_d[key])

            else:
                print("fix str to list", type(row_d[key]), row_d[key])
                # list_from_str = ast.literal_eval(row_d[key])
                # list_from_str = np.fromstring(row_d[key], int)
                # list_from_str = np.loads(row_d[key])

                # list_from_str = np.lib.npyio.format.read_array(row_d[key])
                # tmp = re.sub('\s*', '', row_d[key])  # get rid of spaces
                # a = row_d[key]
                # b = re.sub('[0-9] ', ', ', a)
                # c = b.replace('. ', ',').replace('.\n', ',').replace('0.25', '0.25,').replace('0.5', '0.5,').replace(
                #     '0.75', '0.75,')
                # d = c.replace('0, ]', '0]').replace('0.25, ]', '0.25]').replace('0.5, ]', '0.5]').replace('0.75, ]', '0.75]').replace('1, ]', '1]')


                a = row_d[key]
                a = a.replace('[', '').replace(']', '')

                a_l = a.split(' ')
                print("a_l", a_l)

                new_l = []
                # a_l = a_l.replace('[', '').replace(']', '')
                for a_i in a_l:
                    if len(a_i) != 0:
                        a_i = a_i.replace('\n', '')
                        a_i = a_i.strip()
                        print('a_i "', a_i, '"', len(a_i))
                        if a_i.endswith('.'):
                            a_i = a_i[:-1]
                        a_i = a_i.strip()
                        if len(a_i) != 0:
                            new_l.append(float(a_i))

                print("new_l", new_l)
                # new_list =
                row_d[key] = new_l
                # exit()


                # tmp = row_d[key].replace('. ', ',').replace('.\n', ',').replace('0.25', '0.25,').replace('0.5', '0.5,').replace('0.75', '0.75,')
                # # tmp = re.sub('[a-z]*@', 'ApD@', tmp)
                # tmp = tmp.replace('0, ]', '0]').replace('0.25, ]', '0.25]').replace('0.5, ]', '0.5]').replace('0.75, ]', '0.75]').replace('1, ]', '1]')
                # tmp = tmp.replace('\n', '').replace('] [', '], [')
                # print('tmp', tmp)
                # list_from_str = json.loads(tmp)
                # row_d[key] = list_from_str
                # # print("list_from_str", type(list_from_str), list_from_str)
                # # exit()

    return row_d


if __name__ == "__main__":
    whole_dataset = []

    # read all the csv and save it to one list
    for file in os.listdir('human_data'):
        print("game_data", file)
        if file.endswith('game.csv'):
            df = pd.read_csv('human_data/' + file)
            df_dict_l = df.to_dict('records')
            for row_d in df_dict_l:
                row_d_lists = get_fixed_row_from_str(row_d)
                whole_dataset.append(row_d_lists)
            # print(whole_dataset)
            # exit()

    # create new model of DQN
    q_net = Feedforward(try_cuda=False, input_size=242, hidden_size=21)

    # for every row calculate the reward (reward = y)
    for row_d in whole_dataset:
        print('row = ', row_d)
        exit()

    # train the neural net




