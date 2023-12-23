# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is a simple CNN for CIFAR-10 / CIFAR-100 in PyTorch

import argparse
import time
import math
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import dataset
import model
import trainer
import device


# Train the CNN
def train(device, batch_size, num_epochs, learning_rate, cifar_version):

    # get the data
    train_loader, test_loader, _ = dataset.cifar(batch_size, cifar_version)

    # Instantiate the CNN
    cnn = model.CNN(num_classes=10 if cifar_version == "CIFAR-10" else 100)

    # print summary
    summary(cnn, input_size=(1, 3, 32, 32), row_settings=["var_names"])

    # create trainer
    # Loss function, see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    # Input in the loss should be the unnormalized logits for each class. Softmax is applied in the loss.
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(cnn.parameters(), lr=learning_rate)

    # Train the network
    fname_save_every_epoch = f"model_{cifar_version}"
    trainer_obj = trainer.Trainer(cnn, loss_fn, optimizer, device, fname_save_every_epoch, log_level=logging.INFO)
    trainer_obj.train(train_loader, test_loader, num_epochs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple CNN on CIFAR-10 / CIFAR_100 with PyTorch.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (cuda/mps) acceleration")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batchsize", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, choices=['CIFAR-10', 'CIFAR-100'], default='CIFAR-10', 
                        help="Select the dataset to use (CIFAR-10 or CIFAR-100)")

     
    args = parser.parse_args()
  
    # Autoselect the device to use
    # We transfer our model and data later to this device. If this is a GPU
    # PyTorch will take care of everything automatically.
    dev = torch.device('cpu')
    if not args.cpu:
        dev = device.autoselectDevice(verbose=1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dataset: {args.dataset}")

    train(dev, args.batchsize, args.epochs, args.lr, args.dataset)
