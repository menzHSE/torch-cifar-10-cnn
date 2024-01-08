# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is a simple CNN for CIFAR-10 / CIFAR-100 in PyTorch

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchinfo import summary

import dataset
import device
import model
import tester


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


def test(device, model_fname, cifar_version, finetune_model):
    # Load the training and test data
    batch_size = 32

    # If we have fine-tuned, we obtain the model's weights and transforms
    custom_transforms = None

    if finetune_model in ["resnet18", "resnet34", "resnet50"]:
        weights, custom_transforms = weights_and_transforms_for_resnet(finetune_model)

    # get the data
    _, test_loader, _ = dataset.cifar(batch_size, custom_transforms, cifar_version)

    # Load the model
    # Instantiate the CNN
    num_classes = 10 if cifar_version == "CIFAR-10" else 100

    if finetune_model in ["resnet18", "resnet34", "resnet50"]:
        cnn = model.CNNResnet(finetune_model, weights, num_classes)
    else:
        cnn = model.CNN(num_classes)

    cnn.load(model_fname, device)
    print(f"Loaded model for {cifar_version} from {model_fname}")

    # create tester
    tester_obj = tester.Tester(cnn, device)
    tester_obj.test(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Test a simple CNN on CIFAR-10 / CIFAR-100 with PyTorch."
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU (cuda/mps) acceleration",
    )
    parser.add_argument("--model", type=str, required=True, help="Model filename *.pth")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CIFAR-10", "CIFAR-100"],
        default="CIFAR-10",
        help="Select the dataset to use (CIFAR-10 or CIFAR-100)",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        choices=["resnet18", "resnet34", "resnet50"],
        default=None,
        help="Select the model that has been fine-tuned(resnet18, resnet34, resnet50), "
        + "omit for testing the original model",
    )

    args = parser.parse_args()

    # Autoselect the device to use
    # We transfer our model and data later to this device. If this is a GPU
    # PyTorch will take care of everything automatically.
    dev = torch.device("cpu")
    if not args.cpu:
        dev = device.autoselectDevice(verbose=1)

    test(dev, args.model, args.dataset, args.finetune)
