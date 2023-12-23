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
import tester
import device


def test(device, model_fname, cifar_version):
       
    # Load the training and test data
    batch_size = 32
    # get the data
    _, test_loader, _ = dataset.cifar(batch_size, cifar_version)


    # Load the model
    cnn = model.CNN(num_classes=10 if cifar_version == "CIFAR-10" else 100)
    cnn.load(model_fname)
    print(f"Loaded model for {cifar_version} from {model_fname}")

    # create tester
    tester_obj = tester.Tester(cnn, device)
    tester_obj.test(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test a simple CNN on CIFAR-10 / CIFAR-100 with PyTorch.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (cuda/mps) acceleration")     
    parser.add_argument('--model', type=str, required=True, help='Model filename *.pth')
    parser.add_argument("--dataset", type=str, choices=['CIFAR-10', 'CIFAR-100'], default='CIFAR-10', 
                        help="Select the dataset to use (CIFAR-10 or CIFAR-100)")

    args = parser.parse_args()

    # Autoselect the device to use
    # We transfer our model and data later to this device. If this is a GPU
    # PyTorch will take care of everything automatically.
    dev = torch.device('cpu')
    if not args.cpu:
        dev = device.autoselectDevice(verbose=1)

    test(dev, args.model, args.dataset)
