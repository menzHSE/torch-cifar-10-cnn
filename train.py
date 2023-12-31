# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is a simple CNN for CIFAR-10 / CIFAR-100 in PyTorch

import argparse
import time
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary

import dataset
import model
import trainer
import device

def weights_and_transforms_for_resnet(model):
    if model == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    elif model == "resnet34":
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
    elif model == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unknown model: {model}")
    return weights, weights.transforms()
        

# Train the CNN
def train(device, batch_size, num_epochs, learning_rate, cifar_version, finetune_model):

    # If we want to fine-tune, we obtain the model's weights and transforms
    weights = None
    custom_transforms = None

    if finetune_model in ["resnet18", "resnet34", "resnet50"]:
        model_input_size = 224
        weights, custom_transforms = weights_and_transforms_for_resnet(finetune_model)
    else:
        model_input_size = 32 # hard coded for now
        
    # get the data
    train_loader, test_loader, _ = dataset.cifar(batch_size, custom_transforms, cifar_version)

    # Instantiate the CNN
    num_classes=10 if cifar_version == "CIFAR-10" else 100
    if finetune_model in ["resnet18", "resnet34", "resnet50"]:
        cnn = model.CNNResnet(finetune_model, weights, num_classes)
    else:
        cnn = model.CNN(num_classes)

    # print summary and correctly flush the stream
    model_stats = summary(cnn, input_size=(1, 3, model_input_size, model_input_size), row_settings=["var_names"])
    print("", flush=True)
    time.sleep(1)

    # create trainer
    # Loss function, see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    # Input in the loss should be the unnormalized logits for each class. Softmax is applied in the loss.
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(cnn.parameters(), lr=learning_rate)

    # Train the network
    if finetune_model is not None:
        fname_save_every_epoch = f"models/model_{finetune_model}_{cifar_version}"
    else:
        fname_save_every_epoch = f"models/model_{cifar_version}"
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
    parser.add_argument("--finetune", type=str, choices=['resnet18', 'resnet34', 'resnet50'], default=None, 
                        help="Select the model for fine-tuning (resnet18, resnet34, resnet50), " +
                             "omit for training from scratch")

         
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
    if args.finetune:
        print(f"  Fine-tuning model: {args.finetune}")

    train(dev, args.batchsize, args.epochs, args.lr, args.dataset, args.finetune)
