import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size=20):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # 1 output unit that represents Q(s, a).
        output = self.fc1(x)
        output = self.sigmoid(output)

        output = self.fc2(output)
        output = self.relu(output)
        return output



# class DQN(nn.Module):
#     """
#     Our model will be a convolutional neural network that takes in the difference between the current and previous
#     screen patches. It has two outputs, representing Q(s,left) and Q(s,right) (where s is the input to the network).
#     In effect, the network is trying to predict the expected return of taking each action given the current input.
#     """
#     def __init__(self, h, w, outputs):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)
#
#         # Number of linear input connections depends on output of conv2d layers
#         # and therefore the input image size, so need to compute it
#         def conv2d_size_out(size, kernel_size = 5, stride = 2):
#             return (size - (kernel_size - 1) - 1) // stride + 1  # // operator performs division without fraction part
#
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(in_features=linear_input_size, out_features=outputs)
#
#     # Called with either one element to determine next action or a batch during optimization
#     # returns tensor([ [ left0exp, right0exp ] ... ])
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.size(0), -1))
