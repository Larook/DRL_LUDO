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


def get_before_after_tile_id(pieces_player_begin, begin_state, new_state, action, dice):
    piece_selected_tile = pieces_player_begin[action]
    tile_piece_before = piece_selected_tile
    if piece_selected_tile == 0 and dice == 6:
        tile_piece_after = 1
    else:
        tile_piece_after = piece_selected_tile + dice
    return tile_piece_before, tile_piece_after


def get_reshaped_ann_input(begin_state, new_state, action, pieces_player_begin, dice):
    """ save STATE and ACTION into 1-dimensional np.array. This should be an input to a ANN """
    # look for the position of the given pawn before and after a move
    current_player = 0
    input_ann = np.array(begin_state)
    input_ann = input_ann.reshape((240, 1))
    """TODO:  To estimate the $Q(s,a)$ with a neural network, 
    it is needed for its input to consist the information of transitioning from the previous to the next state with
    visible action taken.
    
    Every action is represented as a tuple
    (x_0 / 60, x_f / 60), where x_0 is the initial position and x_f is the
    final position. The components are divided by 58 in order
    to obtain a number between 0 and 1
    """
    tile_piece_before, tile_piece_after = get_before_after_tile_id(pieces_player_begin, begin_state, new_state, action, dice)

    # action_tuple = (begin_state[current_player][action] / 60, new_state[current_player][action] / 60)
    action_tuple = (tile_piece_before / 59, tile_piece_after / 59)

    # print(input_ann.shape)
    input_ann = np.append(input_ann, action_tuple)
    return input_ann


