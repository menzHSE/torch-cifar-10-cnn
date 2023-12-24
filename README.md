# torch-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 CNN implementation in PyTorch. 

See https://github.com/menzHSE/mlx-cifar-10-cnn for (more or less) the same model being trained using MLX. 

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

This uses an 80GB NVIDIA A100. 

```
$  python train.py --batchsize=512  --dataset CIFAR-10
Using device: cuda
NVIDIA A100 80GB PCIe
Options:
  Device: GPU
  Seed: 0
  Batch size: 512
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-10
Files already downloaded and verified
Files already downloaded and verified
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
CNN (CNN)                                [1, 10]                   --
├─Conv2d (conv1)                         [1, 32, 32, 32]           896
├─BatchNorm2d (bn1)                      [1, 32, 32, 32]           64
├─Conv2d (skip2)                         [1, 32, 32, 32]           1,056
├─Conv2d (conv2)                         [1, 32, 32, 32]           9,248
├─BatchNorm2d (bn2)                      [1, 32, 32, 32]           64
├─MaxPool2d (pool)                       [1, 32, 16, 16]           --
├─Dropout (drop)                         [1, 32, 16, 16]           --
├─Conv2d (conv3)                         [1, 64, 16, 16]           18,496
├─BatchNorm2d (bn3)                      [1, 64, 16, 16]           128
├─Conv2d (skip4)                         [1, 64, 16, 16]           4,160
├─Conv2d (conv4)                         [1, 64, 16, 16]           36,928
├─BatchNorm2d (bn4)                      [1, 64, 16, 16]           128
├─MaxPool2d (pool)                       [1, 64, 8, 8]             --
├─Dropout (drop)                         [1, 64, 8, 8]             --
├─Conv2d (conv5)                         [1, 128, 8, 8]            73,856
├─BatchNorm2d (bn5)                      [1, 128, 8, 8]            256
├─Conv2d (skip6)                         [1, 128, 8, 8]            16,512
├─Conv2d (conv6)                         [1, 128, 8, 8]            147,584
├─BatchNorm2d (bn6)                      [1, 128, 8, 8]            256
├─MaxPool2d (pool)                       [1, 128, 4, 4]            --
├─Dropout (drop)                         [1, 128, 4, 4]            --
├─Linear (fc1)                           [1, 128]                  262,272
├─Linear (fc2)                           [1, 10]                   1,290
==========================================================================================
Total params: 573,194
Trainable params: 573,194
Non-trainable params: 0
Total mult-adds (M): 42.22
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 2.29
Params size (MB): 2.29
Estimated Total Size (MB): 4.60
==========================================================================================

[Epoch   0] :  . done (98 batches)
[Epoch   0] : | time:  3.526s | trainLoss:  1.608 | trainAccuracy:  0.409 | valLoss:  1.510 | valAccuracy:  0.470 | throughput:  85182.313 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.296s | trainLoss:  1.193 | trainAccuracy:  0.572 | valLoss:  1.219 | valAccuracy:  0.578 | throughput:  81477.125 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.253s | trainLoss:  1.007 | trainAccuracy:  0.641 | valLoss:  1.002 | valAccuracy:  0.659 | throughput:  83386.336 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.261s | trainLoss:  0.885 | trainAccuracy:  0.686 | valLoss:  0.984 | valAccuracy:  0.668 | throughput:  78435.205 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.218s | trainLoss:  0.784 | trainAccuracy:  0.726 | valLoss:  0.909 | valAccuracy:  0.690 | throughput:  78821.117 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.189s | trainLoss:  0.713 | trainAccuracy:  0.748 | valLoss:  0.854 | valAccuracy:  0.711 | throughput:  82898.934 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  3.176s | trainLoss:  0.656 | trainAccuracy:  0.770 | valLoss:  0.700 | valAccuracy:  0.762 | throughput:  84698.412 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.348s | trainLoss:  0.610 | trainAccuracy:  0.787 | valLoss:  0.658 | valAccuracy:  0.780 | throughput:  67307.551 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.242s | trainLoss:  0.566 | trainAccuracy:  0.803 | valLoss:  0.651 | valAccuracy:  0.778 | throughput:  75215.313 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.088s | trainLoss:  0.524 | trainAccuracy:  0.815 | valLoss:  0.592 | valAccuracy:  0.801 | throughput:  85779.600 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.180s | trainLoss:  0.496 | trainAccuracy:  0.827 | valLoss:  0.599 | valAccuracy:  0.801 | throughput:  77115.881 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.140s | trainLoss:  0.465 | trainAccuracy:  0.839 | valLoss:  0.586 | valAccuracy:  0.803 | throughput:  85209.710 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.296s | trainLoss:  0.440 | trainAccuracy:  0.846 | valLoss:  0.573 | valAccuracy:  0.809 | throughput:  70195.239 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.096s | trainLoss:  0.411 | trainAccuracy:  0.858 | valLoss:  0.577 | valAccuracy:  0.805 | throughput:  80118.706 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.239s | trainLoss:  0.385 | trainAccuracy:  0.866 | valLoss:  0.561 | valAccuracy:  0.817 | throughput:  82645.635 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.077s | trainLoss:  0.368 | trainAccuracy:  0.872 | valLoss:  0.540 | valAccuracy:  0.823 | throughput:  85231.902 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.127s | trainLoss:  0.346 | trainAccuracy:  0.879 | valLoss:  0.582 | valAccuracy:  0.813 | throughput:  81466.161 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.100s | trainLoss:  0.332 | trainAccuracy:  0.884 | valLoss:  0.544 | valAccuracy:  0.824 | throughput:  84430.638 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.087s | trainLoss:  0.312 | trainAccuracy:  0.891 | valLoss:  0.537 | valAccuracy:  0.822 | throughput:  83123.882 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.382s | trainLoss:  0.300 | trainAccuracy:  0.895 | valLoss:  0.529 | valAccuracy:  0.829 | throughput:  68220.238 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.120s | trainLoss:  0.282 | trainAccuracy:  0.902 | valLoss:  0.540 | valAccuracy:  0.829 | throughput:  84667.004 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.102s | trainLoss:  0.270 | trainAccuracy:  0.906 | valLoss:  0.553 | valAccuracy:  0.821 | throughput:  83709.063 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.151s | trainLoss:  0.255 | trainAccuracy:  0.908 | valLoss:  0.547 | valAccuracy:  0.829 | throughput:  83719.587 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.215s | trainLoss:  0.246 | trainAccuracy:  0.914 | valLoss:  0.535 | valAccuracy:  0.835 | throughput:  71380.772 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.544s | trainLoss:  0.231 | trainAccuracy:  0.919 | valLoss:  0.546 | valAccuracy:  0.833 | throughput:  62765.940 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.187s | trainLoss:  0.226 | trainAccuracy:  0.920 | valLoss:  0.532 | valAccuracy:  0.836 | throughput:  83183.697 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.178s | trainLoss:  0.212 | trainAccuracy:  0.926 | valLoss:  0.570 | valAccuracy:  0.826 | throughput:  84219.808 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.548s | trainLoss:  0.199 | trainAccuracy:  0.929 | valLoss:  0.534 | valAccuracy:  0.839 | throughput:  61236.396 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.097s | trainLoss:  0.194 | trainAccuracy:  0.931 | valLoss:  0.563 | valAccuracy:  0.833 | throughput:  81725.215 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.169s | trainLoss:  0.186 | trainAccuracy:  0.935 | valLoss:  0.566 | valAccuracy:  0.832 | throughput:  82906.049 img/s |
Training finished in 0:01:36.842110 hh:mm:ss.ms
```

### Test on CIFAR-10
```
$ python test.py --model=models/model_CIFAR-10_029.pth
Using device: cuda
NVIDIA A100 80GB PCIe
Files already downloaded and verified
Files already downloaded and verified
Loaded model for CIFAR-10 from models/model_CIFAR-10_029.pth
Testing ....
Test Metrics (10000 test samples):
  - Accuracy: 0.832
Testing finished in 0:00:02.602382 hh:mm:ss.ms
```

## CIFAR-100

### Train on CIFAR-100
`python train.py --dataset CIFAR-100`

This uses an 80GB NVIDIA A100. 

```
$ python train.py --batchsize=512  --dataset CIFAR-100
Using device: cuda
NVIDIA A100 80GB PCIe
Options:
  Device: GPU
  Seed: 0
  Batch size: 512
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-100
Files already downloaded and verified
Files already downloaded and verified
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
CNN (CNN)                                [1, 100]                  --
├─Conv2d (conv1)                         [1, 32, 32, 32]           896
├─BatchNorm2d (bn1)                      [1, 32, 32, 32]           64
├─Conv2d (skip2)                         [1, 32, 32, 32]           1,056
├─Conv2d (conv2)                         [1, 32, 32, 32]           9,248
├─BatchNorm2d (bn2)                      [1, 32, 32, 32]           64
├─MaxPool2d (pool)                       [1, 32, 16, 16]           --
├─Dropout (drop)                         [1, 32, 16, 16]           --
├─Conv2d (conv3)                         [1, 64, 16, 16]           18,496
├─BatchNorm2d (bn3)                      [1, 64, 16, 16]           128
├─Conv2d (skip4)                         [1, 64, 16, 16]           4,160
├─Conv2d (conv4)                         [1, 64, 16, 16]           36,928
├─BatchNorm2d (bn4)                      [1, 64, 16, 16]           128
├─MaxPool2d (pool)                       [1, 64, 8, 8]             --
├─Dropout (drop)                         [1, 64, 8, 8]             --
├─Conv2d (conv5)                         [1, 128, 8, 8]            73,856
├─BatchNorm2d (bn5)                      [1, 128, 8, 8]            256
├─Conv2d (skip6)                         [1, 128, 8, 8]            16,512
├─Conv2d (conv6)                         [1, 128, 8, 8]            147,584
├─BatchNorm2d (bn6)                      [1, 128, 8, 8]            256
├─MaxPool2d (pool)                       [1, 128, 4, 4]            --
├─Dropout (drop)                         [1, 128, 4, 4]            --
├─Linear (fc1)                           [1, 128]                  262,272
├─Linear (fc2)                           [1, 100]                  12,900
==========================================================================================
Total params: 584,804
Trainable params: 584,804
Non-trainable params: 0
Total mult-adds (M): 42.23
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 2.30
Params size (MB): 2.34
Estimated Total Size (MB): 4.65
==========================================================================================

[Epoch   0] :  . done (98 batches)
[Epoch   0] : | time:  3.584s | trainLoss:  4.249 | trainAccuracy:  0.061 | valLoss:  4.013 | valAccuracy:  0.102 | throughput:  82478.823 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.092s | trainLoss:  3.501 | trainAccuracy:  0.172 | valLoss:  3.497 | valAccuracy:  0.198 | throughput:  81611.806 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.161s | trainLoss:  3.090 | trainAccuracy:  0.247 | valLoss:  3.162 | valAccuracy:  0.248 | throughput:  85580.846 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.233s | trainLoss:  2.796 | trainAccuracy:  0.301 | valLoss:  2.852 | valAccuracy:  0.305 | throughput:  80819.013 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.207s | trainLoss:  2.557 | trainAccuracy:  0.350 | valLoss:  2.555 | valAccuracy:  0.365 | throughput:  85973.316 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.274s | trainLoss:  2.363 | trainAccuracy:  0.388 | valLoss:  2.541 | valAccuracy:  0.366 | throughput:  69291.423 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  4.489s | trainLoss:  2.209 | trainAccuracy:  0.420 | valLoss:  2.330 | valAccuracy:  0.406 | throughput:  81887.730 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.167s | trainLoss:  2.077 | trainAccuracy:  0.452 | valLoss:  2.283 | valAccuracy:  0.420 | throughput:  76645.366 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.071s | trainLoss:  1.973 | trainAccuracy:  0.476 | valLoss:  2.145 | valAccuracy:  0.445 | throughput:  81345.379 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.232s | trainLoss:  1.876 | trainAccuracy:  0.495 | valLoss:  2.059 | valAccuracy:  0.463 | throughput:  72574.444 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.298s | trainLoss:  1.791 | trainAccuracy:  0.516 | valLoss:  2.066 | valAccuracy:  0.460 | throughput:  76553.477 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.096s | trainLoss:  1.717 | trainAccuracy:  0.536 | valLoss:  1.961 | valAccuracy:  0.490 | throughput:  83437.530 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.335s | trainLoss:  1.651 | trainAccuracy:  0.550 | valLoss:  1.970 | valAccuracy:  0.487 | throughput:  75454.560 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.117s | trainLoss:  1.588 | trainAccuracy:  0.566 | valLoss:  1.907 | valAccuracy:  0.501 | throughput:  83609.810 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.170s | trainLoss:  1.533 | trainAccuracy:  0.578 | valLoss:  1.931 | valAccuracy:  0.497 | throughput:  77835.877 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.471s | trainLoss:  1.478 | trainAccuracy:  0.590 | valLoss:  1.913 | valAccuracy:  0.501 | throughput:  72709.992 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.244s | trainLoss:  1.432 | trainAccuracy:  0.602 | valLoss:  1.867 | valAccuracy:  0.513 | throughput:  73051.119 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.120s | trainLoss:  1.375 | trainAccuracy:  0.616 | valLoss:  1.873 | valAccuracy:  0.518 | throughput:  84118.105 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.184s | trainLoss:  1.332 | trainAccuracy:  0.627 | valLoss:  1.809 | valAccuracy:  0.526 | throughput:  83339.351 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.242s | trainLoss:  1.286 | trainAccuracy:  0.639 | valLoss:  1.848 | valAccuracy:  0.523 | throughput:  80841.820 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.252s | trainLoss:  1.248 | trainAccuracy:  0.647 | valLoss:  1.802 | valAccuracy:  0.531 | throughput:  75277.649 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.186s | trainLoss:  1.203 | trainAccuracy:  0.661 | valLoss:  1.773 | valAccuracy:  0.537 | throughput:  81183.488 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.376s | trainLoss:  1.175 | trainAccuracy:  0.665 | valLoss:  1.768 | valAccuracy:  0.532 | throughput:  70206.393 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.227s | trainLoss:  1.141 | trainAccuracy:  0.674 | valLoss:  1.800 | valAccuracy:  0.536 | throughput:  68309.179 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.263s | trainLoss:  1.103 | trainAccuracy:  0.684 | valLoss:  1.826 | valAccuracy:  0.543 | throughput:  78856.839 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.223s | trainLoss:  1.066 | trainAccuracy:  0.694 | valLoss:  1.794 | valAccuracy:  0.540 | throughput:  81747.635 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.194s | trainLoss:  1.045 | trainAccuracy:  0.697 | valLoss:  1.772 | valAccuracy:  0.549 | throughput:  85508.511 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.218s | trainLoss:  1.012 | trainAccuracy:  0.707 | valLoss:  1.781 | valAccuracy:  0.547 | throughput:  83865.283 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.204s | trainLoss:  0.978 | trainAccuracy:  0.716 | valLoss:  1.789 | valAccuracy:  0.547 | throughput:  75674.619 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.131s | trainLoss:  0.962 | trainAccuracy:  0.722 | valLoss:  1.798 | valAccuracy:  0.548 | throughput:  84423.293 img/s |
Training finished in 0:01:38.424386 hh:mm:ss.ms
```

### Test on CIFAR-100
```
$ python test.py --model=models/model_CIFAR-100_029.pth  --dataset=CIFAR-100
Using device: cuda
NVIDIA A100 80GB PCIe
Files already downloaded and verified
Files already downloaded and verified
Loaded model for CIFAR-100 from models/model_CIFAR-100_029.pth
Testing ....
Test Metrics (10000 test samples):
  - Accuracy: 0.548
Testing finished in 0:00:02.517724 hh:mm:ss.ms
```
