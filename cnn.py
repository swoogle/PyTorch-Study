#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

import os
import sys
import torch
import os.path as op
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
EPOCHS_NUM = 5
CLASSES_NUM = 10
BATCH_SIZE = 100
LR = 0.001

train_dataset = torchvision.datasets.MNIST(root=op.join(sys.path[0], 'data/'),
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=op.join(sys.path[0], 'data/'),
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, classes_num=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(32*7*7, classes_num)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

convNet = ConvNet(CLASSES_NUM).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(convNet.parameters(), lr=LR)

# Train the convNet
total_step = len(train_loader)
for epoch in range(EPOCHS_NUM):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = convNet(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, EPOCHS_NUM, i+1, total_step, loss.item()))

# Test the convNet
convNet.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = convNet(images)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted, 'prediction number')
        print(labels, 'real number')

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))

# Save the convNet
torch.save(convNet, op.join(sys.path[0], 'model/cnn.pt'))