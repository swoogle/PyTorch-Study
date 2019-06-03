#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

from __future__ import print_function
import os
import sys
import torch
import os.path as op
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms


# recurrent neural network
class RecuNet(nn.Module):
    def __init__(self, input_size, hidden_size, layers_num, classes_num):
        super(RecuNet, self).__init__()
        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.lstm = nn.LSTM(input_size, hidden_size, layers_num, batch_first=True)
        self.fc = nn.Linear(hidden_size, classes_num)

    def forward(self, x):
        # Forward propagate LSTM, out: tensor of shape (batch_size, seq_length, hidden_size)
        out, (h_n, h_c) = self.lstm(x, None) 
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Train the model
def train(model, device, train_loader, criterion, optimizer, epoch_num, sequence_length, input_size):
    model.train()
    total_step = len(train_loader)
    for epoch in range(epoch_num):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, epoch_num, i+1, total_step, loss.item()))


# Test the model
def test(model, device, test_loader, sequence_length, input_size):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted, 'prediction number')
            print(labels, 'real number')

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(
            100 * correct / total))


def main():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    sequence_length = 28
    input_size = 28
    hidden_size = 128
    layers_num = 2
    classes_num = 10
    batch_size = 100
    epochs_num = 5
    learning_rate = 0.01

    train_dataset = torchvision.datasets.MNIST(root=op.join(sys.path[0], 'data/'),
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root=op.join(sys.path[0], 'data/'),
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

    model = RecuNet(input_size, hidden_size, layers_num, classes_num).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train and test model
    train(model, device, train_loader, criterion, optimizer, epochs_num, sequence_length, input_size)
    test(model, device, test_loader, sequence_length, input_size)

   # Save the model
    torch.save(model, op.join(sys.path[0], 'model/mnist_rnn.pt'))


if __name__ == '__main__':
    main()
