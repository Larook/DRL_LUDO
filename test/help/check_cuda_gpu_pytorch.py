import torch
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# device_name = torch.device('cuda:0')



class Feedforward(torch.nn.Module):
    def __init__(self, try_cuda, input_size, hidden_size=20):
        super(Feedforward, self).__init__()
        if try_cuda and torch.cuda.is_available():
            print("will use cuda!")
            print(torch.cuda.get_device_name(0))
            device_name = torch.device('cuda:0')
            self.cuda()

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

model = Feedforward(try_cuda=True, input_size=242, hidden_size=21)



