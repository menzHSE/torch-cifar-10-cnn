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
[Epoch   0] : | time:  3.551s | trainLoss:  1.584 | trainAccuracy:  0.420 | valLoss:  1.489 | valAccuracy:  0.477 | throughput:  81041.956 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.155s | trainLoss:  1.177 | trainAccuracy:  0.582 | valLoss:  1.206 | valAccuracy:  0.583 | throughput:  84726.369 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.151s | trainLoss:  1.004 | trainAccuracy:  0.643 | valLoss:  0.974 | valAccuracy:  0.662 | throughput:  85411.309 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.095s | trainLoss:  0.894 | trainAccuracy:  0.685 | valLoss:  0.873 | valAccuracy:  0.701 | throughput:  85244.838 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.157s | trainLoss:  0.806 | trainAccuracy:  0.715 | valLoss:  0.948 | valAccuracy:  0.677 | throughput:  82846.666 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.118s | trainLoss:  0.746 | trainAccuracy:  0.739 | valLoss:  0.871 | valAccuracy:  0.703 | throughput:  85983.275 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  3.148s | trainLoss:  0.693 | trainAccuracy:  0.756 | valLoss:  0.753 | valAccuracy:  0.742 | throughput:  80765.536 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.201s | trainLoss:  0.650 | trainAccuracy:  0.772 | valLoss:  0.725 | valAccuracy:  0.751 | throughput:  78502.994 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.118s | trainLoss:  0.613 | trainAccuracy:  0.785 | valLoss:  0.698 | valAccuracy:  0.764 | throughput:  86471.838 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.226s | trainLoss:  0.578 | trainAccuracy:  0.798 | valLoss:  0.634 | valAccuracy:  0.781 | throughput:  75138.468 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.240s | trainLoss:  0.554 | trainAccuracy:  0.805 | valLoss:  0.598 | valAccuracy:  0.796 | throughput:  77112.618 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.280s | trainLoss:  0.528 | trainAccuracy:  0.814 | valLoss:  0.587 | valAccuracy:  0.803 | throughput:  79943.540 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.324s | trainLoss:  0.504 | trainAccuracy:  0.822 | valLoss:  0.573 | valAccuracy:  0.809 | throughput:  70398.792 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.138s | trainLoss:  0.480 | trainAccuracy:  0.833 | valLoss:  0.561 | valAccuracy:  0.806 | throughput:  78206.028 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.057s | trainLoss:  0.457 | trainAccuracy:  0.839 | valLoss:  0.578 | valAccuracy:  0.803 | throughput:  84194.908 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.326s | trainLoss:  0.438 | trainAccuracy:  0.846 | valLoss:  0.543 | valAccuracy:  0.816 | throughput:  72150.664 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.086s | trainLoss:  0.423 | trainAccuracy:  0.853 | valLoss:  0.564 | valAccuracy:  0.814 | throughput:  81589.677 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.274s | trainLoss:  0.404 | trainAccuracy:  0.858 | valLoss:  0.516 | valAccuracy:  0.826 | throughput:  77829.112 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.244s | trainLoss:  0.386 | trainAccuracy:  0.864 | valLoss:  0.513 | valAccuracy:  0.830 | throughput:  72253.525 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.090s | trainLoss:  0.375 | trainAccuracy:  0.869 | valLoss:  0.514 | valAccuracy:  0.826 | throughput:  81256.334 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.238s | trainLoss:  0.364 | trainAccuracy:  0.872 | valLoss:  0.559 | valAccuracy:  0.817 | throughput:  70272.215 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.151s | trainLoss:  0.350 | trainAccuracy:  0.877 | valLoss:  0.509 | valAccuracy:  0.830 | throughput:  85616.862 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.151s | trainLoss:  0.337 | trainAccuracy:  0.880 | valLoss:  0.538 | valAccuracy:  0.821 | throughput:  84787.211 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.277s | trainLoss:  0.327 | trainAccuracy:  0.884 | valLoss:  0.497 | valAccuracy:  0.837 | throughput:  68601.885 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.196s | trainLoss:  0.310 | trainAccuracy:  0.891 | valLoss:  0.494 | valAccuracy:  0.840 | throughput:  80438.191 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.311s | trainLoss:  0.306 | trainAccuracy:  0.892 | valLoss:  0.507 | valAccuracy:  0.840 | throughput:  66596.482 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.200s | trainLoss:  0.286 | trainAccuracy:  0.900 | valLoss:  0.505 | valAccuracy:  0.837 | throughput:  73301.517 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.140s | trainLoss:  0.280 | trainAccuracy:  0.901 | valLoss:  0.513 | valAccuracy:  0.834 | throughput:  85618.121 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.204s | trainLoss:  0.271 | trainAccuracy:  0.903 | valLoss:  0.489 | valAccuracy:  0.844 | throughput:  81046.684 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.211s | trainLoss:  0.263 | trainAccuracy:  0.907 | valLoss:  0.550 | valAccuracy:  0.834 | throughput:  82296.701 img/s |
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
  - Accuracy: 0.834
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
[Epoch   0] : | time:  3.553s | trainLoss:  4.211 | trainAccuracy:  0.066 | valLoss:  4.005 | valAccuracy:  0.104 | throughput:  82384.896 img/s |
[Epoch   1] :  . done (98 batches)
[Epoch   1] : | time:  3.148s | trainLoss:  3.501 | trainAccuracy:  0.173 | valLoss:  3.437 | valAccuracy:  0.204 | throughput:  86395.857 img/s |
[Epoch   2] :  . done (98 batches)
[Epoch   2] : | time:  3.226s | trainLoss:  3.117 | trainAccuracy:  0.240 | valLoss:  3.081 | valAccuracy:  0.263 | throughput:  80705.533 img/s |
[Epoch   3] :  . done (98 batches)
[Epoch   3] : | time:  3.428s | trainLoss:  2.840 | trainAccuracy:  0.292 | valLoss:  2.887 | valAccuracy:  0.301 | throughput:  67807.370 img/s |
[Epoch   4] :  . done (98 batches)
[Epoch   4] : | time:  3.280s | trainLoss:  2.623 | trainAccuracy:  0.335 | valLoss:  2.591 | valAccuracy:  0.351 | throughput:  75559.218 img/s |
[Epoch   5] :  . done (98 batches)
[Epoch   5] : | time:  3.165s | trainLoss:  2.462 | trainAccuracy:  0.369 | valLoss:  2.528 | valAccuracy:  0.359 | throughput:  83110.168 img/s |
[Epoch   6] :  . done (98 batches)
[Epoch   6] : | time:  3.252s | trainLoss:  2.332 | trainAccuracy:  0.395 | valLoss:  2.439 | valAccuracy:  0.381 | throughput:  83298.803 img/s |
[Epoch   7] :  . done (98 batches)
[Epoch   7] : | time:  3.185s | trainLoss:  2.217 | trainAccuracy:  0.419 | valLoss:  2.346 | valAccuracy:  0.402 | throughput:  80363.605 img/s |
[Epoch   8] :  . done (98 batches)
[Epoch   8] : | time:  3.124s | trainLoss:  2.117 | trainAccuracy:  0.439 | valLoss:  2.198 | valAccuracy:  0.429 | throughput:  88093.301 img/s |
[Epoch   9] :  . done (98 batches)
[Epoch   9] : | time:  3.152s | trainLoss:  2.032 | trainAccuracy:  0.457 | valLoss:  2.140 | valAccuracy:  0.441 | throughput:  85660.886 img/s |
[Epoch  10] :  . done (98 batches)
[Epoch  10] : | time:  3.195s | trainLoss:  1.966 | trainAccuracy:  0.473 | valLoss:  2.141 | valAccuracy:  0.444 | throughput:  83853.821 img/s |
[Epoch  11] :  . done (98 batches)
[Epoch  11] : | time:  3.228s | trainLoss:  1.898 | trainAccuracy:  0.488 | valLoss:  2.064 | valAccuracy:  0.461 | throughput:  76172.942 img/s |
[Epoch  12] :  . done (98 batches)
[Epoch  12] : | time:  3.177s | trainLoss:  1.836 | trainAccuracy:  0.503 | valLoss:  1.971 | valAccuracy:  0.486 | throughput:  82018.693 img/s |
[Epoch  13] :  . done (98 batches)
[Epoch  13] : | time:  3.109s | trainLoss:  1.775 | trainAccuracy:  0.518 | valLoss:  1.913 | valAccuracy:  0.495 | throughput:  85336.153 img/s |
[Epoch  14] :  . done (98 batches)
[Epoch  14] : | time:  3.171s | trainLoss:  1.730 | trainAccuracy:  0.529 | valLoss:  1.953 | valAccuracy:  0.490 | throughput:  85540.909 img/s |
[Epoch  15] :  . done (98 batches)
[Epoch  15] : | time:  3.192s | trainLoss:  1.683 | trainAccuracy:  0.538 | valLoss:  1.916 | valAccuracy:  0.498 | throughput:  80763.866 img/s |
[Epoch  16] :  . done (98 batches)
[Epoch  16] : | time:  3.144s | trainLoss:  1.641 | trainAccuracy:  0.549 | valLoss:  1.850 | valAccuracy:  0.505 | throughput:  86594.761 img/s |
[Epoch  17] :  . done (98 batches)
[Epoch  17] : | time:  3.224s | trainLoss:  1.598 | trainAccuracy:  0.557 | valLoss:  1.890 | valAccuracy:  0.503 | throughput:  72335.193 img/s |
[Epoch  18] :  . done (98 batches)
[Epoch  18] : | time:  3.444s | trainLoss:  1.554 | trainAccuracy:  0.570 | valLoss:  1.825 | valAccuracy:  0.518 | throughput:  70519.472 img/s |
[Epoch  19] :  . done (98 batches)
[Epoch  19] : | time:  3.118s | trainLoss:  1.517 | trainAccuracy:  0.576 | valLoss:  1.803 | valAccuracy:  0.523 | throughput:  86103.886 img/s |
[Epoch  20] :  . done (98 batches)
[Epoch  20] : | time:  3.304s | trainLoss:  1.483 | trainAccuracy:  0.587 | valLoss:  1.823 | valAccuracy:  0.517 | throughput:  78449.367 img/s |
[Epoch  21] :  . done (98 batches)
[Epoch  21] : | time:  3.200s | trainLoss:  1.447 | trainAccuracy:  0.595 | valLoss:  1.742 | valAccuracy:  0.538 | throughput:  82706.356 img/s |
[Epoch  22] :  . done (98 batches)
[Epoch  22] : | time:  3.159s | trainLoss:  1.416 | trainAccuracy:  0.602 | valLoss:  1.736 | valAccuracy:  0.540 | throughput:  86920.052 img/s |
[Epoch  23] :  . done (98 batches)
[Epoch  23] : | time:  3.172s | trainLoss:  1.388 | trainAccuracy:  0.607 | valLoss:  1.785 | valAccuracy:  0.534 | throughput:  86830.040 img/s |
[Epoch  24] :  . done (98 batches)
[Epoch  24] : | time:  3.282s | trainLoss:  1.351 | trainAccuracy:  0.620 | valLoss:  1.756 | valAccuracy:  0.543 | throughput:  68667.128 img/s |
[Epoch  25] :  . done (98 batches)
[Epoch  25] : | time:  3.202s | trainLoss:  1.316 | trainAccuracy:  0.627 | valLoss:  1.710 | valAccuracy:  0.550 | throughput:  82810.959 img/s |
[Epoch  26] :  . done (98 batches)
[Epoch  26] : | time:  3.307s | trainLoss:  1.295 | trainAccuracy:  0.633 | valLoss:  1.699 | valAccuracy:  0.552 | throughput:  72265.411 img/s |
[Epoch  27] :  . done (98 batches)
[Epoch  27] : | time:  3.127s | trainLoss:  1.263 | trainAccuracy:  0.639 | valLoss:  1.699 | valAccuracy:  0.551 | throughput:  86624.581 img/s |
[Epoch  28] :  . done (98 batches)
[Epoch  28] : | time:  3.144s | trainLoss:  1.232 | trainAccuracy:  0.645 | valLoss:  1.664 | valAccuracy:  0.558 | throughput:  81490.799 img/s |
[Epoch  29] :  . done (98 batches)
[Epoch  29] : | time:  3.146s | trainLoss:  1.207 | trainAccuracy:  0.656 | valLoss:  1.715 | valAccuracy:  0.551 | throughput:  85611.276 img/s |
Training finished in 0:01:36.768437 hh:mm:ss.ms
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
  - Accuracy: 0.551
Testing finished in 0:00:02.517724 hh:mm:ss.ms
```
