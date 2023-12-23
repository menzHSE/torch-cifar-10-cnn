# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """A simple CNN for CIFAR-10 / CIFAR-100. """

    def __init__(self, num_classes):
        super().__init__()
        # network layers
        self.conv1 = nn.Conv2d   (3,        32, 3, padding='same')
        self.conv2 = nn.Conv2d   (32,       32, 3, padding='same')
        self.pool  = nn.MaxPool2d(2,         2                   )
        self.conv3 = nn.Conv2d   (32,       64, 3, padding='same')
        self.conv4 = nn.Conv2d   (64,       64, 3, padding='same')
        self.conv5 = nn.Conv2d   (64,      128, 3, padding='same')
        self.conv6 = nn.Conv2d   (128,     128, 3, padding='same')
        self.fc1   = nn.Linear   (128*4*4, 128                   )
        self.fc2   = nn.Linear   (128,     num_classes           )

        self.drop  = nn.Dropout(0.25)

    # forward pass of the data "x"
    def forward(self, x):

        # Input: 3x32x32
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop(x)
        # Output: 32x16x16

        # Input: 32x16x16
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = self.drop(x)
        # Output: 64x8x8

        # Input: 64x8x8
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.pool(x)
        x = self.drop(x)
        # Output: 128x4x4

        # Input: 128x4x4

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        # "softmax" activation will be automatically applied in the cross entropy loss below,
        # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        return x
    
 
    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
        self.eval()