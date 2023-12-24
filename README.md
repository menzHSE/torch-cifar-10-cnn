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
$  python train.py --batchsize=512 --dataset=CIFAR-10
Using device: cuda
NVIDIA A100 80GB PCIe
Options:
  Device: GPU
  Seed: 0
  Batch size: 512
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
CNN (CNN)                                [1, 100]                  --
├─Conv2d (conv1)                         [1, 32, 32, 32]           896
├─BatchNorm2d (bn1)                      [1, 32, 32, 32]           64
├─Conv2d (conv2)                         [1, 32, 32, 32]           9,248
├─BatchNorm2d (bn2)                      [1, 32, 32, 32]           64
├─MaxPool2d (pool)                       [1, 32, 16, 16]           --
├─Dropout (drop)                         [1, 32, 16, 16]           --
├─Conv2d (conv3)                         [1, 64, 16, 16]           18,496
├─BatchNorm2d (bn3)                      [1, 64, 16, 16]           128
├─Conv2d (conv4)                         [1, 64, 16, 16]           36,928
├─BatchNorm2d (bn4)                      [1, 64, 16, 16]           128
├─MaxPool2d (pool)                       [1, 64, 8, 8]             --
├─Dropout (drop)                         [1, 64, 8, 8]             --
├─Conv2d (conv5)                         [1, 128, 8, 8]            73,856
├─BatchNorm2d (bn5)                      [1, 128, 8, 8]            256
├─Conv2d (conv6)                         [1, 128, 8, 8]            147,584
├─BatchNorm2d (bn6)                      [1, 128, 8, 8]            256
├─MaxPool2d (pool)                       [1, 128, 4, 4]            --
├─Dropout (drop)                         [1, 128, 4, 4]            --
├─Linear (fc1)                           [1, 128]                  262,272
├─Linear (fc2)                           [1, 100]                  12,900
==========================================================================================
Total params: 563,076
Trainable params: 563,076
Non-trainable params: 0
Total mult-adds (M): 39.02
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 1.84
Params size (MB): 2.25
Estimated Total Size (MB): 4.10
==========================================================================================
[Epoch   0] :  . done (98 batches)
[Epoch   0] : | time:  3.854s | trainLoss:  1.608 | trainAccuracy:  0.409 | valLoss:  1.415 | valAccuracy:  0.502 | Throughput: 106679.902 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.151s | trainLoss:  1.209 | trainAccuracy:  0.565 | valLoss:  1.152 | valAccuracy:  0.601 | Throughput: 106750.067 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.092s | trainLoss:  1.027 | trainAccuracy:  0.635 | valLoss:  1.095 | valAccuracy:  0.622 | Throughput: 106657.651 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.195s | trainLoss:  0.910 | trainAccuracy:  0.676 | valLoss:  1.103 | valAccuracy:  0.630 | Throughput:  99152.257 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.190s | trainLoss:  0.835 | trainAccuracy:  0.707 | valLoss:  1.047 | valAccuracy:  0.644 | Throughput: 107223.327 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.163s | trainLoss:  0.770 | trainAccuracy:  0.727 | valLoss:  0.865 | valAccuracy:  0.704 | Throughput:  92843.730 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  3.200s | trainLoss:  0.720 | trainAccuracy:  0.746 | valLoss:  0.788 | valAccuracy:  0.735 | Throughput:  99081.978 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.223s | trainLoss:  0.674 | trainAccuracy:  0.763 | valLoss:  0.796 | valAccuracy:  0.729 | Throughput:  84405.895 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.241s | trainLoss:  0.647 | trainAccuracy:  0.774 | valLoss:  0.704 | valAccuracy:  0.763 | Throughput: 109105.675 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.135s | trainLoss:  0.614 | trainAccuracy:  0.786 | valLoss:  0.683 | valAccuracy:  0.771 | Throughput:  98001.947 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.214s | trainLoss:  0.587 | trainAccuracy:  0.795 | valLoss:  0.617 | valAccuracy:  0.792 | Throughput: 105054.901 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.179s | trainLoss:  0.566 | trainAccuracy:  0.801 | valLoss:  0.657 | valAccuracy:  0.783 | Throughput: 109316.610 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.362s | trainLoss:  0.541 | trainAccuracy:  0.812 | valLoss:  0.599 | valAccuracy:  0.803 | Throughput:  82360.556 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.185s | trainLoss:  0.523 | trainAccuracy:  0.818 | valLoss:  0.636 | valAccuracy:  0.787 | Throughput: 109027.703 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.203s | trainLoss:  0.508 | trainAccuracy:  0.822 | valLoss:  0.601 | valAccuracy:  0.800 | Throughput: 104464.256 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.242s | trainLoss:  0.485 | trainAccuracy:  0.830 | valLoss:  0.637 | valAccuracy:  0.793 | Throughput:  89491.639 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.427s | trainLoss:  0.473 | trainAccuracy:  0.834 | valLoss:  0.571 | valAccuracy:  0.814 | Throughput:  81084.162 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.114s | trainLoss:  0.460 | trainAccuracy:  0.839 | valLoss:  0.568 | valAccuracy:  0.815 | Throughput: 109545.113 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.414s | trainLoss:  0.447 | trainAccuracy:  0.842 | valLoss:  0.607 | valAccuracy:  0.803 | Throughput:  80587.006 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.135s | trainLoss:  0.429 | trainAccuracy:  0.851 | valLoss:  0.540 | valAccuracy:  0.821 | Throughput: 109971.437 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.178s | trainLoss:  0.421 | trainAccuracy:  0.851 | valLoss:  0.555 | valAccuracy:  0.820 | Throughput: 106959.791 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.268s | trainLoss:  0.410 | trainAccuracy:  0.855 | valLoss:  0.594 | valAccuracy:  0.809 | Throughput: 103097.934 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.315s | trainLoss:  0.399 | trainAccuracy:  0.860 | valLoss:  0.538 | valAccuracy:  0.823 | Throughput:  92234.467 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.216s | trainLoss:  0.386 | trainAccuracy:  0.865 | valLoss:  0.547 | valAccuracy:  0.820 | Throughput: 104308.267 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.259s | trainLoss:  0.379 | trainAccuracy:  0.865 | valLoss:  0.509 | valAccuracy:  0.831 | Throughput:  89400.632 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.182s | trainLoss:  0.365 | trainAccuracy:  0.870 | valLoss:  0.521 | valAccuracy:  0.828 | Throughput: 111933.308 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.254s | trainLoss:  0.360 | trainAccuracy:  0.872 | valLoss:  0.525 | valAccuracy:  0.832 | Throughput:  82986.530 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.594s | trainLoss:  0.349 | trainAccuracy:  0.879 | valLoss:  0.545 | valAccuracy:  0.828 | Throughput:  75988.367 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.174s | trainLoss:  0.338 | trainAccuracy:  0.880 | valLoss:  0.538 | valAccuracy:  0.825 | Throughput:  99147.813 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.141s | trainLoss:  0.329 | trainAccuracy:  0.885 | valLoss:  0.558 | valAccuracy:  0.826 | Throughput: 108885.591 img/s |
Training finished in 0:01:37.803991 hh:mm:ss.ms
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
  - Accuracy: 0.826
Testing finished in 0:00:02.694355 hh:mm:ss.ms
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
Downloading https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz to ./data/cifar-100-python.tar.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 169001437/169001437 [00:06<00:00, 24952360.15it/s]
Extracting ./data/cifar-100-python.tar.gz to ./data
Files already downloaded and verified
==========================================================================================
Layer (type (var_name))                  Output Shape              Param #
==========================================================================================
CNN (CNN)                                [1, 100]                  --
├─Conv2d (conv1)                         [1, 32, 32, 32]           896
├─BatchNorm2d (bn1)                      [1, 32, 32, 32]           64
├─Conv2d (conv2)                         [1, 32, 32, 32]           9,248
├─BatchNorm2d (bn2)                      [1, 32, 32, 32]           64
├─MaxPool2d (pool)                       [1, 32, 16, 16]           --
├─Dropout (drop)                         [1, 32, 16, 16]           --
├─Conv2d (conv3)                         [1, 64, 16, 16]           18,496
├─BatchNorm2d (bn3)                      [1, 64, 16, 16]           128
├─Conv2d (conv4)                         [1, 64, 16, 16]           36,928
├─BatchNorm2d (bn4)                      [1, 64, 16, 16]           128
├─MaxPool2d (pool)                       [1, 64, 8, 8]             --
├─Dropout (drop)                         [1, 64, 8, 8]             --
├─Conv2d (conv5)                         [1, 128, 8, 8]            73,856
├─BatchNorm2d (bn5)                      [1, 128, 8, 8]            256
├─Conv2d (conv6)                         [1, 128, 8, 8]            147,584
├─BatchNorm2d (bn6)                      [1, 128, 8, 8]            256
├─MaxPool2d (pool)                       [1, 128, 4, 4]            --
├─Dropout (drop)                         [1, 128, 4, 4]            --
├─Linear (fc1)                           [1, 128]                  262,272
├─Linear (fc2)                           [1, 100]                  12,900
==========================================================================================
Total params: 563,076
Trainable params: 563,076
Non-trainable params: 0
Total mult-adds (M): 39.02
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 1.84
Params size (MB): 2.25
Estimated Total Size (MB): 4.10
==========================================================================================

[Epoch   0] :  . done (98 batches)
[Epoch   0] : | time:  3.454s | trainLoss:  4.231 | trainAccuracy:  0.067 | valLoss:  3.955 | valAccuracy:  0.108 | throughput:  96789.882 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.156s | trainLoss:  3.480 | trainAccuracy:  0.173 | valLoss:  3.412 | valAccuracy:  0.203 | throughput: 101664.794 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.185s | trainLoss:  3.115 | trainAccuracy:  0.241 | valLoss:  3.337 | valAccuracy:  0.222 | throughput:  97546.742 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.162s | trainLoss:  2.832 | trainAccuracy:  0.293 | valLoss:  3.111 | valAccuracy:  0.262 | throughput: 109670.662 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.117s | trainLoss:  2.620 | trainAccuracy:  0.335 | valLoss:  2.648 | valAccuracy:  0.334 | throughput: 109027.878 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.137s | trainLoss:  2.449 | trainAccuracy:  0.371 | valLoss:  2.593 | valAccuracy:  0.349 | throughput: 111375.682 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  3.302s | trainLoss:  2.317 | trainAccuracy:  0.398 | valLoss:  2.569 | valAccuracy:  0.358 | throughput:  92767.178 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.368s | trainLoss:  2.214 | trainAccuracy:  0.420 | valLoss:  2.380 | valAccuracy:  0.393 | throughput:  88522.693 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.236s | trainLoss:  2.119 | trainAccuracy:  0.442 | valLoss:  2.223 | valAccuracy:  0.429 | throughput:  99760.393 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.166s | trainLoss:  2.041 | trainAccuracy:  0.459 | valLoss:  2.212 | valAccuracy:  0.427 | throughput:  96154.842 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.162s | trainLoss:  1.973 | trainAccuracy:  0.475 | valLoss:  2.108 | valAccuracy:  0.449 | throughput: 107409.208 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.044s | trainLoss:  1.907 | trainAccuracy:  0.486 | valLoss:  2.057 | valAccuracy:  0.470 | throughput: 105035.024 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.160s | trainLoss:  1.851 | trainAccuracy:  0.500 | valLoss:  2.102 | valAccuracy:  0.456 | throughput: 108768.224 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.112s | trainLoss:  1.806 | trainAccuracy:  0.512 | valLoss:  2.126 | valAccuracy:  0.455 | throughput:  98156.392 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.100s | trainLoss:  1.760 | trainAccuracy:  0.520 | valLoss:  1.996 | valAccuracy:  0.476 | throughput: 112546.685 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.176s | trainLoss:  1.713 | trainAccuracy:  0.531 | valLoss:  1.904 | valAccuracy:  0.498 | throughput:  93088.746 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.504s | trainLoss:  1.672 | trainAccuracy:  0.543 | valLoss:  1.869 | valAccuracy:  0.506 | throughput:  77395.262 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.306s | trainLoss:  1.629 | trainAccuracy:  0.551 | valLoss:  1.886 | valAccuracy:  0.501 | throughput:  89673.822 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.079s | trainLoss:  1.596 | trainAccuracy:  0.559 | valLoss:  1.837 | valAccuracy:  0.517 | throughput: 102152.237 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.186s | trainLoss:  1.552 | trainAccuracy:  0.570 | valLoss:  1.781 | valAccuracy:  0.525 | throughput: 107620.777 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.204s | trainLoss:  1.524 | trainAccuracy:  0.574 | valLoss:  1.786 | valAccuracy:  0.528 | throughput:  87966.526 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.197s | trainLoss:  1.489 | trainAccuracy:  0.585 | valLoss:  1.742 | valAccuracy:  0.539 | throughput:  90604.443 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.476s | trainLoss:  1.463 | trainAccuracy:  0.590 | valLoss:  1.760 | valAccuracy:  0.531 | throughput:  77966.798 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.167s | trainLoss:  1.427 | trainAccuracy:  0.601 | valLoss:  1.750 | valAccuracy:  0.535 | throughput: 104512.915 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.271s | trainLoss:  1.402 | trainAccuracy:  0.607 | valLoss:  1.787 | valAccuracy:  0.534 | throughput: 104117.689 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.174s | trainLoss:  1.380 | trainAccuracy:  0.611 | valLoss:  1.728 | valAccuracy:  0.546 | throughput: 101893.671 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.149s | trainLoss:  1.350 | trainAccuracy:  0.619 | valLoss:  1.723 | valAccuracy:  0.541 | throughput: 104508.179 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.170s | trainLoss:  1.329 | trainAccuracy:  0.624 | valLoss:  1.689 | valAccuracy:  0.549 | throughput:  95529.540 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.128s | trainLoss:  1.302 | trainAccuracy:  0.629 | valLoss:  1.702 | valAccuracy:  0.547 | throughput:  99823.018 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.090s | trainLoss:  1.280 | trainAccuracy:  0.637 | valLoss:  1.751 | valAccuracy:  0.540 | throughput: 110360.605 img/s |
Training finished in 0:01:36.648186 hh:mm:ss.ms
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
  - Accuracy: 0.540
Testing finished in 0:00:02.582369 hh:mm:ss.ms
```
