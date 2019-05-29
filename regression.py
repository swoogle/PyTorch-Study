#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# x data (tensor), shape=(100, 1)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
# noisy y data (tensor), shape=(100, 1)
y = 0.5 * x.pow(2) + 0.2 * torch.rand(x.size())

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # hidden layer
        self.hidden = nn.Linear(n_feature, n_hidden)
        # output layer
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # activation function for hidden layer
        x = F.relu(self.hidden(x))
        # linear output
        x = self.predict(x)
        return x


# define the network
net = Net(n_feature = 1, n_hidden = 10, n_output = 1)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.2)
# this is for regression mean squared loss
loss_func = torch.nn.MSELoss()

# something about plotting
plt.ion()

for t in range(200):
    # input x and predict based on x
    prediction = net(x)
    # must be (1. nn output, 2. target)
    loss = loss_func(prediction, y)
    
    # clear gradients for next train
    optimizer.zero_grad()
    # backpropagation, compute gradients
    loss.backward()
    # apply gradients
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
        plt.text(0.25, 0, 'Loss=%.4f' % loss.data.numpy(),
                 fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
