#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import torch
import torch.utils.data as Data

BATCH_SIZE = 5

# this is x data (torch tensor)
x = torch.linspace(1, 10, 10)
# this is y data (torch tensor)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2               # subprocesses for loading data
)


def show_batch():
    # train entire dataset 3 times
    for epoch in range(3):
        # for each training step
        for step, (batch_x, batch_y) in enumerate(loader):
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()
