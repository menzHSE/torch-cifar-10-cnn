# torch-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 ResNet-like CNN implementation in PyTorch. 

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
[Epoch   0] : | time:  3.463s | trainLoss:  1.608 | trainAccuracy:  0.409 | valLoss:  1.547 | valAccuracy:  0.460 | throughput:  86253.256 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.141s | trainLoss:  1.193 | trainAccuracy:  0.573 | valLoss:  1.258 | valAccuracy:  0.567 | throughput:  83937.610 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.375s | trainLoss:  1.008 | trainAccuracy:  0.642 | valLoss:  0.969 | valAccuracy:  0.668 | throughput:  68932.503 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.196s | trainLoss:  0.886 | trainAccuracy:  0.687 | valLoss:  0.954 | valAccuracy:  0.677 | throughput:  73139.131 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.349s | trainLoss:  0.787 | trainAccuracy:  0.724 | valLoss:  0.924 | valAccuracy:  0.686 | throughput:  70358.619 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.202s | trainLoss:  0.717 | trainAccuracy:  0.748 | valLoss:  0.840 | valAccuracy:  0.717 | throughput:  73180.432 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  3.227s | trainLoss:  0.660 | trainAccuracy:  0.769 | valLoss:  0.699 | valAccuracy:  0.764 | throughput:  85500.814 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.094s | trainLoss:  0.612 | trainAccuracy:  0.787 | valLoss:  0.662 | valAccuracy:  0.778 | throughput:  86023.888 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.067s | trainLoss:  0.567 | trainAccuracy:  0.803 | valLoss:  0.630 | valAccuracy:  0.786 | throughput:  86963.361 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.234s | trainLoss:  0.527 | trainAccuracy:  0.816 | valLoss:  0.606 | valAccuracy:  0.796 | throughput:  81349.960 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.127s | trainLoss:  0.498 | trainAccuracy:  0.828 | valLoss:  0.604 | valAccuracy:  0.798 | throughput:  86073.199 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.115s | trainLoss:  0.469 | trainAccuracy:  0.836 | valLoss:  0.584 | valAccuracy:  0.803 | throughput:  80480.693 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.151s | trainLoss:  0.442 | trainAccuracy:  0.846 | valLoss:  0.577 | valAccuracy:  0.806 | throughput:  77756.301 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.211s | trainLoss:  0.415 | trainAccuracy:  0.855 | valLoss:  0.564 | valAccuracy:  0.811 | throughput:  73234.967 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.118s | trainLoss:  0.385 | trainAccuracy:  0.865 | valLoss:  0.549 | valAccuracy:  0.817 | throughput:  82200.094 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.241s | trainLoss:  0.371 | trainAccuracy:  0.869 | valLoss:  0.543 | valAccuracy:  0.821 | throughput:  75561.320 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.442s | trainLoss:  0.351 | trainAccuracy:  0.877 | valLoss:  0.552 | valAccuracy:  0.821 | throughput:  64772.159 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.284s | trainLoss:  0.335 | trainAccuracy:  0.883 | valLoss:  0.542 | valAccuracy:  0.827 | throughput:  71640.955 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.151s | trainLoss:  0.318 | trainAccuracy:  0.890 | valLoss:  0.556 | valAccuracy:  0.820 | throughput:  77383.137 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.132s | trainLoss:  0.303 | trainAccuracy:  0.894 | valLoss:  0.531 | valAccuracy:  0.830 | throughput:  86589.410 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.293s | trainLoss:  0.283 | trainAccuracy:  0.901 | valLoss:  0.536 | valAccuracy:  0.830 | throughput:  67708.022 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.192s | trainLoss:  0.272 | trainAccuracy:  0.905 | valLoss:  0.551 | valAccuracy:  0.821 | throughput:  80829.243 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.226s | trainLoss:  0.253 | trainAccuracy:  0.911 | valLoss:  0.555 | valAccuracy:  0.827 | throughput:  72563.493 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.178s | trainLoss:  0.247 | trainAccuracy:  0.913 | valLoss:  0.558 | valAccuracy:  0.832 | throughput:  86425.289 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.156s | trainLoss:  0.235 | trainAccuracy:  0.918 | valLoss:  0.565 | valAccuracy:  0.823 | throughput:  85690.960 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.200s | trainLoss:  0.227 | trainAccuracy:  0.921 | valLoss:  0.545 | valAccuracy:  0.835 | throughput:  86722.386 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.311s | trainLoss:  0.216 | trainAccuracy:  0.925 | valLoss:  0.579 | valAccuracy:  0.825 | throughput:  86752.040 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.223s | trainLoss:  0.203 | trainAccuracy:  0.929 | valLoss:  0.523 | valAccuracy:  0.843 | throughput:  73632.212 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.493s | trainLoss:  0.191 | trainAccuracy:  0.933 | valLoss:  0.585 | valAccuracy:  0.828 | throughput:  60934.829 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.488s | trainLoss:  0.189 | trainAccuracy:  0.933 | valLoss:  0.557 | valAccuracy:  0.837 | throughput:  63697.672 img/s |
Training finished in 0:01:36.280061 hh:mm:ss.ms
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
  - Accuracy: 0.837
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
[Epoch   0] : | time:  3.519s | trainLoss:  4.250 | trainAccuracy:  0.061 | valLoss:  4.013 | valAccuracy:  0.100 | throughput:  81421.198 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.121s | trainLoss:  3.515 | trainAccuracy:  0.170 | valLoss:  3.454 | valAccuracy:  0.202 | throughput:  84310.820 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.311s | trainLoss:  3.106 | trainAccuracy:  0.243 | valLoss:  3.162 | valAccuracy:  0.247 | throughput:  71636.242 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.231s | trainLoss:  2.809 | trainAccuracy:  0.300 | valLoss:  2.830 | valAccuracy:  0.310 | throughput:  85042.933 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.245s | trainLoss:  2.567 | trainAccuracy:  0.350 | valLoss:  2.596 | valAccuracy:  0.351 | throughput:  72137.381 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.221s | trainLoss:  2.373 | trainAccuracy:  0.388 | valLoss:  2.517 | valAccuracy:  0.370 | throughput:  74949.552 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  3.259s | trainLoss:  2.218 | trainAccuracy:  0.422 | valLoss:  2.347 | valAccuracy:  0.404 | throughput:  78963.998 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.460s | trainLoss:  2.080 | trainAccuracy:  0.451 | valLoss:  2.241 | valAccuracy:  0.428 | throughput:  67380.274 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.284s | trainLoss:  1.978 | trainAccuracy:  0.472 | valLoss:  2.173 | valAccuracy:  0.438 | throughput:  72844.612 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.160s | trainLoss:  1.875 | trainAccuracy:  0.498 | valLoss:  2.064 | valAccuracy:  0.461 | throughput:  82040.979 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.293s | trainLoss:  1.790 | trainAccuracy:  0.516 | valLoss:  2.057 | valAccuracy:  0.464 | throughput:  74010.440 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.594s | trainLoss:  1.713 | trainAccuracy:  0.535 | valLoss:  1.942 | valAccuracy:  0.492 | throughput:  61697.813 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.220s | trainLoss:  1.643 | trainAccuracy:  0.552 | valLoss:  1.983 | valAccuracy:  0.491 | throughput:  83290.275 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.316s | trainLoss:  1.580 | trainAccuracy:  0.567 | valLoss:  1.910 | valAccuracy:  0.501 | throughput:  84072.799 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.266s | trainLoss:  1.525 | trainAccuracy:  0.580 | valLoss:  1.898 | valAccuracy:  0.507 | throughput:  79228.463 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.231s | trainLoss:  1.467 | trainAccuracy:  0.593 | valLoss:  1.876 | valAccuracy:  0.511 | throughput:  85412.637 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.188s | trainLoss:  1.420 | trainAccuracy:  0.606 | valLoss:  1.861 | valAccuracy:  0.515 | throughput:  85235.887 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.500s | trainLoss:  1.368 | trainAccuracy:  0.619 | valLoss:  1.882 | valAccuracy:  0.510 | throughput:  61683.322 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.238s | trainLoss:  1.324 | trainAccuracy:  0.627 | valLoss:  1.821 | valAccuracy:  0.530 | throughput:  84189.020 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.209s | trainLoss:  1.275 | trainAccuracy:  0.642 | valLoss:  1.833 | valAccuracy:  0.531 | throughput:  80853.284 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.139s | trainLoss:  1.240 | trainAccuracy:  0.650 | valLoss:  1.798 | valAccuracy:  0.534 | throughput:  86022.746 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.213s | trainLoss:  1.198 | trainAccuracy:  0.661 | valLoss:  1.732 | valAccuracy:  0.547 | throughput:  85616.647 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.170s | trainLoss:  1.165 | trainAccuracy:  0.667 | valLoss:  1.732 | valAccuracy:  0.546 | throughput:  84710.322 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.539s | trainLoss:  1.131 | trainAccuracy:  0.676 | valLoss:  1.782 | valAccuracy:  0.545 | throughput:  59764.028 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.131s | trainLoss:  1.086 | trainAccuracy:  0.687 | valLoss:  1.803 | valAccuracy:  0.546 | throughput:  86034.478 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.359s | trainLoss:  1.056 | trainAccuracy:  0.695 | valLoss:  1.772 | valAccuracy:  0.552 | throughput:  79477.761 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.276s | trainLoss:  1.032 | trainAccuracy:  0.702 | valLoss:  1.772 | valAccuracy:  0.548 | throughput:  72641.042 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.251s | trainLoss:  1.002 | trainAccuracy:  0.709 | valLoss:  1.778 | valAccuracy:  0.552 | throughput:  80770.211 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.461s | trainLoss:  0.970 | trainAccuracy:  0.720 | valLoss:  1.747 | valAccuracy:  0.558 | throughput:  62484.855 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.574s | trainLoss:  0.950 | trainAccuracy:  0.721 | valLoss:  1.773 | valAccuracy:  0.549 | throughput:  61059.334 img/s |
Training finished in 0:01:39.189674 hh:mm:ss.ms
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
  - Accuracy: 0.549
Testing finished in 0:00:02.517724 hh:mm:ss.ms
```
