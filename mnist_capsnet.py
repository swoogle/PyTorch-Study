#! /usr/bin/env python3
#! _*_ coding: utf-8 _*_

from __future__ import print_function
import os
import sys
import torch
import os.path as op
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from capsnet import CapsNet, CapsuleLoss

# Train the model
def train(model, device, train_loader, criterion, optimizer, scheduler, epoch_num):
    model.train()
    for ep in range(epoch_num):
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
            logits, reconstruction = model(images)

            # Compute loss & accuracy
            loss = criterion(images, labels, logits, reconstruction)
            correct += torch.sum(
                torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss
            loss.backward()
            optimizer.step()
            print('Epoch {}, batch {}, loss: {:.4f}, accuracy: {:.4f}'.format(ep + 1,
                                                                      batch_id,
                                                                      total_loss / batch_id,
                                                                      accuracy))
            batch_id += 1
        scheduler.step(ep)
        print('Total loss for epoch {}: {:.4f}'.format(ep + 1, total_loss))

# Test the model
def test(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        # Add channels = 1
        images = images.to(device)
        # Categogrical encoding
        labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
        logits, reconstructions = model(images)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
        total += len(labels)
    print('Accuracy: {:.4f}%'.format(100 * correct / total))

def main():
    # Device configuration, check cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    BATCH_SIZE = 128
    EPOCHS_NUM = 50
    LR = 0.001
    GAMMA = 0.96
    
    transform = transforms.Compose([
        # shift by 2 pixels in either direction with zero padding.
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
        
    train_dataset = torchvision.datasets.MNIST(root=op.join(sys.path[0], 'data/'),
                                               train=True,
                                               transform=transform,
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root=op.join(sys.path[0], 'data/'),
                                              train=False,
                                              transform=transform)

    # Data loader
    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=BATCH_SIZE,
                                   num_workers=4,
                                   shuffle=True)

    test_loader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=4,
                                  shuffle=True)

    # Load model
    model = CapsNet().to(device)
    # Loss and optimizer
    criterion = CapsuleLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

    # Train and test model
    train(model, device, train_loader, criterion, optimizer, scheduler, EPOCHS_NUM)
    test(model, device, test_loader)

    # Save model
    torch.save(model, op.join(sys.path[0], 'model/mnist_capsnet.pt'))

if __name__ == '__main__':
    main()
