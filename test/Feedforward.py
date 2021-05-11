import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class Feedforward(torch.nn.Module):
    def __init__(self, try_cuda, input_size, hidden_size=20):
        super(Feedforward, self).__init__()
        if try_cuda and torch.cuda.is_available():
            print("WILL USE CUDA")
            print(torch.cuda.get_device_name(0))
            device_name = torch.device('cuda:0')
            self.cuda()
        else:
            print("WON'T USE CUDA")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 1 output unit that represents Q(s, a).
        x = torch.tensor(x).float()

        output = self.fc1(x)
        output = self.sigmoid(output)

        output = self.fc2(output)
        output = self.relu(output)
        return output


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


