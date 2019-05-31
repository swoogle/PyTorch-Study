#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCHS = 12

# fake dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plot dataset
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

# put dateset into torch dataset
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset,
                         batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# default network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


def train():
    # different nets
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # different optimizers
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(
        net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(
        net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(
        net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    # record loss
    loss_records = [[], [], [], []]

    # training
    for epoch in range(EPOCHS):
        print('Epoch: ', epoch)
        # for each training step
        for step, (batch_x, batch_y) in enumerate(loader):
            for net, opt, loss_record in zip(nets, optimizers, loss_records):
                # get output for every net
                output = net(batch_x)
                # compute loss for every net
                loss = loss_func(output, batch_y)
                # clear gradients for next train
                opt.zero_grad()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                opt.step()
                # loss recoder
                loss_record.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, loss_record in enumerate(loss_records):
        plt.plot(loss_record, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()


if __name__ == '__main__':
    train()
