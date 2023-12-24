# torch-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 CNN implementation in PyTorch. 

# Requirements
* torch
* torchvision
* torchinfo
* numpy
* Pillow

See `requirements.txt`



# Usage
```
$ python train.py -h
usage: Train a simple CNN on CIFAR-10 / CIFAR_100 with PyTorch. [-h] [--cpu] [--seed SEED] [--batchsize BATCHSIZE] [--epochs EPOCHS]
                                                            [--lr LR] [--dataset {CIFAR-10,CIFAR-100}]

options:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of Metal GPU acceleration
  --seed SEED           Random seed
  --batchsize BATCHSIZE
                        Batch size for training
  --epochs EPOCHS       Number of training epochs
  --lr LR               Learning rate
  --dataset {CIFAR-10,CIFAR-100}
                        Select the dataset to use (CIFAR-10 or CIFAR-100)
```

```
python test.py -h
usage: Test a simple CNN on CIFAR-10 / CIFAR-100 with PyTorch. [-h] [--cpu] --model MODEL [--dataset {CIFAR-10,CIFAR-100}]

options:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of Metal GPU acceleration
  --model MODEL         Model filename *.npz
  --dataset {CIFAR-10,CIFAR-100}
                        Select the dataset to use (CIFAR-10 or CIFAR-100)
```

# Examples

## CIFAR-10

### Train on CIFAR-10
`python train.py`

This uses an 80GB NVIDIA H100. 

```
$  python train.py --batchsize=256
Using device: cuda
NVIDIA H100 PCIe
Options:
  Device: GPU
  Seed: 0
  Batch size: 256
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-10
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 170498071/170498071 [00:06<00:00, 25683749.06it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
CNN (CNN)                                [1, 10]                   --
├─Conv2d (conv1)                         [1, 32, 32, 32]           896
├─Conv2d (conv2)                         [1, 32, 32, 32]           9,248
├─MaxPool2d (pool)                       [1, 32, 16, 16]           --
├─Dropout (drop)                         [1, 32, 16, 16]           --
├─Conv2d (conv3)                         [1, 64, 16, 16]           18,496
├─Conv2d (conv4)                         [1, 64, 16, 16]           36,928
├─MaxPool2d (pool)                       [1, 64, 8, 8]             --
├─Dropout (drop)                         [1, 64, 8, 8]             --
├─Conv2d (conv5)                         [1, 128, 8, 8]            73,856
├─Conv2d (conv6)                         [1, 128, 8, 8]            147,584
├─MaxPool2d (pool)                       [1, 128, 4, 4]            --
├─Dropout (drop)                         [1, 128, 4, 4]            --
├─Linear (fc1)                           [1, 128]                  262,272
├─Linear (fc2)                           [1, 10]                   1,290
==========================================================================================
Total params: 550,570
Trainable params: 550,570
Non-trainable params: 0
Total mult-adds (M): 39.01
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.92
Params size (MB): 2.20
Estimated Total Size (MB): 3.13
==========================================================================================
[Epoch   0] :  .... done (391 batches)
[Epoch   0] : | time:  5.518s | trainLoss:  1.858 | trainAccuracy:  0.311 | valLoss:  1.613 | valAccuracy:  0.414 | throughput:  64384.502 img/s |
[Epoch   1] :  .... done (391 batches)
[Epoch   1] : | time:  4.675s | trainLoss:  1.476 | trainAccuracy:  0.464 | valLoss:  1.385 | valAccuracy:  0.495 | throughput:  67336.097 img/s |
[Epoch   2] :  .... done (391 batches)
[Epoch   2] : | time:  4.659s | trainLoss:  1.324 | trainAccuracy:  0.521 | valLoss:  1.273 | valAccuracy:  0.540 | throughput:  67839.551 img/s |
[Epoch   3] :  .... done (391 batches)
[Epoch   3] : | time:  4.659s | trainLoss:  1.207 | trainAccuracy:  0.567 | valLoss:  1.137 | valAccuracy:  0.591 | throughput:  67536.684 img/s |
[Epoch   4] :  .... done (391 batches)
[Epoch   4] : | time:  4.686s | trainLoss:  1.110 | trainAccuracy:  0.602 | valLoss:  1.071 | valAccuracy:  0.619 | throughput:  68019.044 img/s |
[Epoch   5] :  .... done (391 batches)
[Epoch   5] : | time:  4.666s | trainLoss:  1.029 | trainAccuracy:  0.632 | valLoss:  1.006 | valAccuracy:  0.644 | throughput:  67443.947 img/s |
[Epoch   6] :  .... done (391 batches)
[Epoch   6] : | time:  4.683s | trainLoss:  0.963 | trainAccuracy:  0.660 | valLoss:  0.953 | valAccuracy:  0.663 | throughput:  67608.607 img/s |
[Epoch   7] :  .... done (391 batches)
[Epoch   7] : | time:  4.682s | trainLoss:  0.907 | trainAccuracy:  0.680 | valLoss:  0.891 | valAccuracy:  0.688 | throughput:  67719.909 img/s |
[Epoch   8] :  .... done (391 batches)
[Epoch   8] : | time:  4.667s | trainLoss:  0.853 | trainAccuracy:  0.700 | valLoss:  0.892 | valAccuracy:  0.687 | throughput:  67628.869 img/s |
[Epoch   9] :  .... done (391 batches)
[Epoch   9] : | time:  4.679s | trainLoss:  0.803 | trainAccuracy:  0.718 | valLoss:  0.847 | valAccuracy:  0.704 | throughput:  67975.429 img/s |
[Epoch  10] :  .... done (391 batches)
[Epoch  10] : | time:  4.683s | trainLoss:  0.760 | trainAccuracy:  0.733 | valLoss:  0.838 | valAccuracy:  0.715 | throughput:  67983.945 img/s |
[Epoch  11] :  .... done (391 batches)
[Epoch  11] : | time:  4.674s | trainLoss:  0.721 | trainAccuracy:  0.746 | valLoss:  0.794 | valAccuracy:  0.723 | throughput:  67957.664 img/s |
[Epoch  12] :  .... done (391 batches)
[Epoch  12] : | time:  4.653s | trainLoss:  0.689 | trainAccuracy:  0.761 | valLoss:  0.825 | valAccuracy:  0.715 | throughput:  68235.068 img/s |
[Epoch  13] :  .... done (391 batches)
[Epoch  13] : | time:  4.677s | trainLoss:  0.650 | trainAccuracy:  0.772 | valLoss:  0.764 | valAccuracy:  0.738 | throughput:  67766.572 img/s |
[Epoch  14] :  .... done (391 batches)
[Epoch  14] : | time:  4.673s | trainLoss:  0.612 | trainAccuracy:  0.786 | valLoss:  0.790 | valAccuracy:  0.732 | throughput:  68122.675 img/s |
[Epoch  15] :  .... done (391 batches)
[Epoch  15] : | time:  4.659s | trainLoss:  0.585 | trainAccuracy:  0.794 | valLoss:  0.762 | valAccuracy:  0.738 | throughput:  67197.899 img/s |
[Epoch  16] :  .... done (391 batches)
[Epoch  16] : | time:  4.664s | trainLoss:  0.558 | trainAccuracy:  0.801 | valLoss:  0.748 | valAccuracy:  0.747 | throughput:  67987.589 img/s |
[Epoch  17] :  .... done (391 batches)
[Epoch  17] : | time:  4.672s | trainLoss:  0.530 | trainAccuracy:  0.813 | valLoss:  0.733 | valAccuracy:  0.757 | throughput:  67732.255 img/s |
[Epoch  18] :  .... done (391 batches)
[Epoch  18] : | time:  4.686s | trainLoss:  0.509 | trainAccuracy:  0.820 | valLoss:  0.738 | valAccuracy:  0.756 | throughput:  67927.114 img/s |
[Epoch  19] :  .... done (391 batches)
[Epoch  19] : | time:  4.675s | trainLoss:  0.477 | trainAccuracy:  0.831 | valLoss:  0.744 | valAccuracy:  0.757 | throughput:  68103.918 img/s |
[Epoch  20] :  .... done (391 batches)
[Epoch  20] : | time:  4.703s | trainLoss:  0.452 | trainAccuracy:  0.840 | valLoss:  0.736 | valAccuracy:  0.763 | throughput:  67504.451 img/s |
[Epoch  21] :  .... done (391 batches)
[Epoch  21] : | time:  4.686s | trainLoss:  0.429 | trainAccuracy:  0.848 | valLoss:  0.755 | valAccuracy:  0.764 | throughput:  68384.435 img/s |
[Epoch  22] :  .... done (391 batches)
[Epoch  22] : | time:  4.684s | trainLoss:  0.407 | trainAccuracy:  0.855 | valLoss:  0.743 | valAccuracy:  0.767 | throughput:  68457.882 img/s |
[Epoch  23] :  .... done (391 batches)
[Epoch  23] : | time:  4.690s | trainLoss:  0.388 | trainAccuracy:  0.862 | valLoss:  0.751 | valAccuracy:  0.766 | throughput:  67976.986 img/s |
[Epoch  24] :  .... done (391 batches)
[Epoch  24] : | time:  4.676s | trainLoss:  0.367 | trainAccuracy:  0.871 | valLoss:  0.798 | valAccuracy:  0.759 | throughput:  67145.001 img/s |
[Epoch  25] :  .... done (391 batches)
[Epoch  25] : | time:  4.669s | trainLoss:  0.350 | trainAccuracy:  0.875 | valLoss:  0.757 | valAccuracy:  0.769 | throughput:  67844.659 img/s |
[Epoch  26] :  .... done (391 batches)
[Epoch  26] : | time:  4.665s | trainLoss:  0.335 | trainAccuracy:  0.881 | valLoss:  0.773 | valAccuracy:  0.770 | throughput:  67739.310 img/s |
[Epoch  27] :  .... done (391 batches)
[Epoch  27] : | time:  4.670s | trainLoss:  0.325 | trainAccuracy:  0.886 | valLoss:  0.836 | valAccuracy:  0.757 | throughput:  67632.124 img/s |
[Epoch  28] :  .... done (391 batches)
[Epoch  28] : | time:  4.664s | trainLoss:  0.307 | trainAccuracy:  0.891 | valLoss:  0.789 | valAccuracy:  0.766 | throughput:  67556.409 img/s |
[Epoch  29] :  .... done (391 batches)
[Epoch  29] : | time:  4.666s | trainLoss:  0.287 | trainAccuracy:  0.897 | valLoss:  0.850 | valAccuracy:  0.758 | throughput:  68036.042 img/s |
Training finished in 0:02:21.257716 hh:mm:ss.ms
```

### Test on CIFAR-10
```
$ python test.py --model=model_CIFAR-10_029.pth
Using device: cuda
NVIDIA H100 PCIe
Files already downloaded and verified
Files already downloaded and verified
Loaded model for CIFAR-10 from model_CIFAR-10_029.pth
Testing ....
Test Metrics (10000 test samples):
  - Accuracy: 0.758
Testing finished in 0:00:01.818539 hh:mm:ss.ms
```

## CIFAR-100

### Train on CIFAR-100
`python train.py --dataset CIFAR-100`

This uses an 80GB NVIDIA H100. 

```
$ python train.py --batchsize=256 --dataset=CIFAR-100
python train.py --dataset CIFAR-100
Using device: cuda
NVIDIA H100 PCIe
Options:
  Device: GPU
  Seed: 0
  Batch size: 256
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-100
Downloading https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz to ./data/cifar-100-python.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 169001437/169001437 [00:06<00:00, 24952360.15it/s]
Extracting ./data/cifar-100-python.tar.gz to ./data
Files already downloaded and verified
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
CNN (CNN)                                [1, 100]                  --
├─Conv2d (conv1)                         [1, 32, 32, 32]           896
├─Conv2d (conv2)                         [1, 32, 32, 32]           9,248
├─MaxPool2d (pool)                       [1, 32, 16, 16]           --
├─Dropout (drop)                         [1, 32, 16, 16]           --
├─Conv2d (conv3)                         [1, 64, 16, 16]           18,496
├─Conv2d (conv4)                         [1, 64, 16, 16]           36,928
├─MaxPool2d (pool)                       [1, 64, 8, 8]             --
├─Dropout (drop)                         [1, 64, 8, 8]             --
├─Conv2d (conv5)                         [1, 128, 8, 8]            73,856
├─Conv2d (conv6)                         [1, 128, 8, 8]            147,584
├─MaxPool2d (pool)                       [1, 128, 4, 4]            --
├─Dropout (drop)                         [1, 128, 4, 4]            --
├─Linear (fc1)                           [1, 128]                  262,272
├─Linear (fc2)                           [1, 100]                  12,900
==========================================================================================
Total params: 562,180
Trainable params: 562,180
Non-trainable params: 0
Total mult-adds (M): 39.02
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.92
Params size (MB): 2.25
Estimated Total Size (MB): 3.18
==========================================================================================

[Epoch   0] :  .... done (391 batches)
[Epoch   0] : | time:  5.527s | trainLoss:  4.242 | trainAccuracy:  0.054 | valLoss:  3.906 | valAccuracy:  0.108 | throughput:  66928.406 img/s |
[Epoch   1] :  .... done (391 batches)
[Epoch   1] : | time:  4.663s | trainLoss:  3.638 | trainAccuracy:  0.144 | valLoss:  3.487 | valAccuracy:  0.184 | throughput:  62364.129 img/s |
[Epoch   2] :  .... done (391 batches)
[Epoch   2] : | time:  4.658s | trainLoss:  3.315 | trainAccuracy:  0.202 | valLoss:  3.212 | valAccuracy:  0.237 | throughput:  63584.357 img/s |
[Epoch   3] :  .... done (391 batches)
[Epoch   3] : | time:  4.671s | trainLoss:  3.100 | trainAccuracy:  0.243 | valLoss:  3.062 | valAccuracy:  0.261 | throughput:  67809.647 img/s |
[Epoch   4] :  .... done (391 batches)
[Epoch   4] : | time:  4.696s | trainLoss:  2.932 | trainAccuracy:  0.275 | valLoss:  2.907 | valAccuracy:  0.289 | throughput:  67827.669 img/s |
[Epoch   5] :  .... done (391 batches)
[Epoch   5] : | time:  4.680s | trainLoss:  2.785 | trainAccuracy:  0.306 | valLoss:  2.846 | valAccuracy:  0.307 | throughput:  68021.259 img/s |
[Epoch   6] :  .... done (391 batches)
[Epoch   6] : | time:  4.680s | trainLoss:  2.668 | trainAccuracy:  0.329 | valLoss:  2.775 | valAccuracy:  0.325 | throughput:  67639.781 img/s |
[Epoch   7] :  .... done (391 batches)
[Epoch   7] : | time:  4.653s | trainLoss:  2.552 | trainAccuracy:  0.352 | valLoss:  2.679 | valAccuracy:  0.340 | throughput:  68193.057 img/s |
[Epoch   8] :  .... done (391 batches)
[Epoch   8] : | time:  4.679s | trainLoss:  2.457 | trainAccuracy:  0.371 | valLoss:  2.664 | valAccuracy:  0.344 | throughput:  68352.621 img/s |
[Epoch   9] :  .... done (391 batches)
[Epoch   9] : | time:  4.670s | trainLoss:  2.361 | trainAccuracy:  0.391 | valLoss:  2.579 | valAccuracy:  0.358 | throughput:  68568.771 img/s |
[Epoch  10] :  .... done (391 batches)
[Epoch  10] : | time:  4.669s | trainLoss:  2.280 | trainAccuracy:  0.409 | valLoss:  2.540 | valAccuracy:  0.371 | throughput:  68019.202 img/s |
[Epoch  11] :  .... done (391 batches)
[Epoch  11] : | time:  4.692s | trainLoss:  2.192 | trainAccuracy:  0.427 | valLoss:  2.537 | valAccuracy:  0.374 | throughput:  68549.152 img/s |
[Epoch  12] :  .... done (391 batches)
[Epoch  12] : | time:  4.678s | trainLoss:  2.113 | trainAccuracy:  0.445 | valLoss:  2.503 | valAccuracy:  0.381 | throughput:  68498.576 img/s |
[Epoch  13] :  .... done (391 batches)
[Epoch  13] : | time:  4.695s | trainLoss:  2.047 | trainAccuracy:  0.457 | valLoss:  2.493 | valAccuracy:  0.386 | throughput:  68395.384 img/s |
[Epoch  14] :  .... done (391 batches)
[Epoch  14] : | time:  4.672s | trainLoss:  1.980 | trainAccuracy:  0.473 | valLoss:  2.500 | valAccuracy:  0.392 | throughput:  67824.150 img/s |
[Epoch  15] :  .... done (391 batches)
[Epoch  15] : | time:  4.676s | trainLoss:  1.923 | trainAccuracy:  0.484 | valLoss:  2.488 | valAccuracy:  0.394 | throughput:  67766.080 img/s |
[Epoch  16] :  .... done (391 batches)
[Epoch  16] : | time:  4.673s | trainLoss:  1.858 | trainAccuracy:  0.500 | valLoss:  2.495 | valAccuracy:  0.405 | throughput:  68358.417 img/s |
[Epoch  17] :  .... done (391 batches)
[Epoch  17] : | time:  4.697s | trainLoss:  1.812 | trainAccuracy:  0.508 | valLoss:  2.507 | valAccuracy:  0.396 | throughput:  67827.711 img/s |
[Epoch  18] :  .... done (391 batches)
[Epoch  18] : | time:  4.667s | trainLoss:  1.759 | trainAccuracy:  0.519 | valLoss:  2.480 | valAccuracy:  0.407 | throughput:  67005.057 img/s |
[Epoch  19] :  .... done (391 batches)
[Epoch  19] : | time:  4.687s | trainLoss:  1.706 | trainAccuracy:  0.532 | valLoss:  2.495 | valAccuracy:  0.409 | throughput:  68419.747 img/s |
[Epoch  20] :  .... done (391 batches)
[Epoch  20] : | time:  4.715s | trainLoss:  1.649 | trainAccuracy:  0.545 | valLoss:  2.516 | valAccuracy:  0.403 | throughput:  68045.727 img/s |
[Epoch  21] :  .... done (391 batches)
[Epoch  21] : | time:  4.688s | trainLoss:  1.611 | trainAccuracy:  0.555 | valLoss:  2.503 | valAccuracy:  0.409 | throughput:  67146.109 img/s |
[Epoch  22] :  .... done (391 batches)
[Epoch  22] : | time:  4.698s | trainLoss:  1.573 | trainAccuracy:  0.564 | valLoss:  2.563 | valAccuracy:  0.407 | throughput:  68058.420 img/s |
[Epoch  23] :  .... done (391 batches)
[Epoch  23] : | time:  4.692s | trainLoss:  1.531 | trainAccuracy:  0.571 | valLoss:  2.535 | valAccuracy:  0.415 | throughput:  67868.473 img/s |
[Epoch  24] :  .... done (391 batches)
[Epoch  24] : | time:  4.676s | trainLoss:  1.493 | trainAccuracy:  0.580 | valLoss:  2.563 | valAccuracy:  0.411 | throughput:  68384.683 img/s |
[Epoch  25] :  .... done (391 batches)
[Epoch  25] : | time:  4.665s | trainLoss:  1.452 | trainAccuracy:  0.590 | valLoss:  2.564 | valAccuracy:  0.410 | throughput:  68297.743 img/s |
[Epoch  26] :  .... done (391 batches)
[Epoch  26] : | time:  4.687s | trainLoss:  1.422 | trainAccuracy:  0.598 | valLoss:  2.546 | valAccuracy:  0.410 | throughput:  67759.634 img/s |
[Epoch  27] :  .... done (391 batches)
[Epoch  27] : | time:  4.675s | trainLoss:  1.388 | trainAccuracy:  0.602 | valLoss:  2.586 | valAccuracy:  0.411 | throughput:  67656.384 img/s |
[Epoch  28] :  .... done (391 batches)
[Epoch  28] : | time:  4.666s | trainLoss:  1.355 | trainAccuracy:  0.611 | valLoss:  2.616 | valAccuracy:  0.415 | throughput:  68053.303 img/s |
[Epoch  29] :  .... done (391 batches)
[Epoch  29] : | time:  4.660s | trainLoss:  1.327 | trainAccuracy:  0.618 | valLoss:  2.597 | valAccuracy:  0.420 | throughput:  68052.415 img/s |
Training finished in 0:02:21.358824 hh:mm:ss.ms
```

### Test on CIFAR-100
```
$ python test.py --model=model_CIFAR-100_029.pth  --dataset=CIFAR-100
Using device: cuda
NVIDIA H100 PCIe
Files already downloaded and verified
Files already downloaded and verified
Loaded model for CIFAR-100 from model_CIFAR-100_029.pth
Testing ....
Test Metrics (10000 test samples):
  - Accuracy: 0.420
Testing finished in 0:00:01.839540 hh:mm:ss.ms
```
