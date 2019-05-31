#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import sys
import torch
import torch.nn as nn
import os.path as op
import matplotlib.pyplot as plt

# fake data
# x data (tensor), shape=(100, 1)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# noisy y data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())


def save():
    # save net1
    net1 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # 2 ways to save the net
    # save entire net
    torch.save(net1, op.join(sys.path[0], 'model/net.pt'))
    # save only the parameters
    torch.save(net1.state_dict(), op.join(sys.path[0], 'model/net_params.pt'))


def restore_net():
    # restore entire net1 to net2
    net2 = torch.load(op.join(sys.path[0], 'model/net.pt'))
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    # restore only the parameters in net1 to net3
    net3 = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))

    # copy net1's parameters into net3
    net3.load_state_dict(torch.load(
        op.join(sys.path[0], 'model/net_params.pt')))
    prediction = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


def main():
    # save net1
    save()
    # restore entire net (may slow)
    restore_net()
    # restore only the net parameters
    restore_params()


if __name__ == "__main__":
    main()
