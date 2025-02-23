import time
import numpy as np
import torch
import torchvision
from torch.ao.quantization.utils import activation_dtype
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# batch_image_count: 10000,  original size: 32*32*3
class Net(torch.nn.Module):
    def __init__(self, n_hidden_nodes, activation, keep_rate=0): # số lượng node của hidden layers, số lượng hidden layers, hàm activation
        # gọi constructor của Net
        super(Net, self).__init__()
        # định nghia cac truong
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        if keep_rate == 0:
            keep_rate = 0.5
        # set up layers n_hidden_nodes, layers, activation, keep_rate, drop, out
        self.fc1 = torch.nn.Linear(32*32*3,n_hidden_nodes[0])
        self.fc1_drop = torch.nn.Dropout(1-keep_rate)
        self.fc2 = torch.nn.Linear(n_hidden_nodes[0], n_hidden_nodes[1])
        self.fc2_drop = torch.nn.Dropout(1-keep_rate)
        self.out = torch.nn.Linear(n_hidden_nodes[1], 10)

    def forward(self,x):
        # (64,3,28,28) -> (64, 2352)
        x = x.view(-1,32*32*3)
        if self.activation == 'sigmoid':
            sigmoid = torch.nn.Sigmoid()
            x = sigmoid(self.fc1(x))
            x = self.fc1_drop(x)
            x = sigmoid(self.fc2(x))
            x = self.fc2_drop(x)
        elif self.activation == 'relu':
            x = torch.nn.functional.relu(self.fc1(x))
            x = self.fc1_drop(x)
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc2_drop(x)
        return torch.nn.functional.log_softmax(self.out(x))
