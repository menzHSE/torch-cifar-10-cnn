# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
import os
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
        self.conv3 = nn.Conv2d   (32,       64, 3, padding='same')
        self.conv4 = nn.Conv2d   (64,       64, 3, padding='same')
        self.conv5 = nn.Conv2d   (64,      128, 3, padding='same')
        self.conv6 = nn.Conv2d   (128,     128, 3, padding='same')

	# Poor man's ResNet ...
        # skip connections (learned 1x1 convolutions with stride=1)
        self.skip2 = nn.Conv2d( 32,  32, 1, stride=1, padding=0)
        self.skip4 = nn.Conv2d( 64,  64, 1, stride=1, padding=0)
        self.skip6 = nn.Conv2d(128, 128, 1, stride=1, padding=0)


        self.bn1  = nn.BatchNorm2d( 32)
        self.bn2  = nn.BatchNorm2d( 32)
        self.bn3  = nn.BatchNorm2d( 64)
        self.bn4  = nn.BatchNorm2d( 64)
        self.bn5  = nn.BatchNorm2d(128)
        self.bn6  = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(2,         2                   )

        self.fc1   = nn.Linear   (128*4*4, 128                   )
        self.fc2   = nn.Linear   (128,     num_classes           )

        self.drop  = nn.Dropout(0.25)

    # forward pass of the data "x"
    def forward(self, x):

	# Poor man's ResNet with residual connections

	# For some reason, residual connections work better in this
        # example with relu() applied before the addition and not after
        # the addition as in the original ResNet paper.


	# Input: 3x32x32
        x =                 F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.skip2(x) + F.leaky_relu(self.bn2(self.conv2(x))) # residual connection
        x = self.pool(x)
        x = self.drop(x)
        # Output: 32x16x16

        # Input: 32x16x16
        x =                 F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.skip4(x) + F.leaky_relu(self.bn4(self.conv4(x))) # residual connection
        x = self.pool(x)
        x = self.drop(x)
        # Output: 64x8x8

        # Input: 64x8x8
        x =                 F.leaky_relu(self.bn5(self.conv5(x)))
        x = self.skip6(x) + F.leaky_relu(self.bn6(self.conv6(x))) # residual connection
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
        # Extract the directory path from the file name
        dir_path = os.path.dirname(fname)

        # Check if the directory path is not empty
        if dir_path:
            # Check if the directory exists, and create it if it does not
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        # save the model
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))
        self.eval()
