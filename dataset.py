# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import os
import math
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

def cifar(batch_size, cifar_version="CIFAR-10", root="./data"):
    if (cifar_version == "CIFAR-10"):
        load_fn = torchvision.datasets.CIFAR10
        return cifar_loader(batch_size, load_fn, root)
    elif (cifar_version == "CIFAR-100"):
        load_fn = torchvision.datasets.CIFAR100
        return cifar_loader(batch_size, load_fn, root)
    else:
        raise ValueError(f"Unknown CIFAR version: {cifar_version}")
    

def cifar_loader(batch_size, load_fn, root):

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # resize to 64x64 pixels
         transforms.ToTensor()         # convert to tensor. This will also normalize pixels to 0-1
        ])

    # load train and test sets using torchvision
    tr   = load_fn(root=root, train=True,   download=True, transform=transform)
    test = load_fn(root=root, train=False,  download=True, transform=transform)
   
    # Data loaders
    train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size,
                                               shuffle=True, pin_memory=True, num_workers=2)
    
    test_loader  = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                               shuffle=False, pin_memory=True, num_workers=2)
    
    return train_loader, test_loader, tr.classes    


if __name__ == "__main__":

    batch_size = 32
    tr_loader, test_loader, classes = cifar(batch_size=batch_size, cifar_version="CIFAR-10")

    images, labels = next(iter(tr_loader)) 
    assert images.shape == (batch_size, 3, 32, 32), "Wrong training set size"
    assert labels.shape == (batch_size,), "Wrong training set size"


    images, labels = next(iter(test_loader))
    assert images.shape == (batch_size, 3, 32, 32), "Wrong training set size"
    assert labels.shape == (batch_size,), "Wrong training set size"
   
    print(classes)

    # Save an image as a sanity check
        
    # Normalize the image and convert to numpy array
    img_data = images[0].numpy() * 255
    img_data = np.transpose(img_data, (1, 2, 0)).astype(np.uint8)

    # Save the image using Pillow
    img = Image.fromarray(img_data)
    img.save("/tmp/trainTmp.png")


    print("Dataset prepared successfully!")
