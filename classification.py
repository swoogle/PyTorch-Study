#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

"""
区分类型（分类）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# make fake data
n_data = torch.ones(100, 2)
# class0 x data (tensor), shape=(100, 2)
x0 = torch.normal(2 * n_data, 1)
# class0 y data (tensor), shape=(100, 1)
y0 = torch.zeros(100)
# class1 x data (tensor), shape=(100, 2)
x1 = torch.normal(-2 * n_data, 1)
# class1 y data (tensor), shape=(100, 1)
y1 = torch.ones(100)
# shape (200, 2) FloatTensor = 32-bit floating
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
# shape (200,) LongTensor = 64-bit integer
y = torch.cat((y0, y1), ).type(torch.LongTensor)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
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
net = Net(n_feature=2, n_hidden=10, n_output=2)
# another method define the network 
# net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
print(net) 

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# this is for regression mean squared loss
loss_func = nn.CrossEntropyLoss()

# something about plotting
plt.ion()

for t in range(100):
    # input x and predict based on x
    out = net(x)
    # must be (1. nn output, 2. target), the target label is NOT one-hotted
    loss = loss_func(out, y)

    # clear gradients for next train
    optimizer.zero_grad()
    # backpropagation, compute gradients
    loss.backward()
    # apply gradients
    optimizer.step()

    # plot and show learning process
    plt.cla()
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = y.data.numpy()
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[
        :, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
    accuracy = float((pred_y == target_y).astype(
        int).sum()) / float(target_y.size)
    plt.text(0.9, -5, 'Accuracy=%.2f' %
             accuracy, fontdict={'size': 20, 'color':  'red'})
    plt.pause(0.1)

plt.ioff()
plt.show()
