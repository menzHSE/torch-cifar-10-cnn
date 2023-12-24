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

[Epoch   0] :  .. done (196 batches)
[Epoch   0] : | time:  6.588s | trainLoss:  1.967 | trainAccuracy:  0.270 | valLoss:  1.770 | valAccuracy:  0.375 | throughput: 130877.626 img/s |
[Epoch   1] :  .. done (196 batches)
[Epoch   1] : | time:  4.628s | trainLoss:  1.589 | trainAccuracy:  0.421 | valLoss:  1.485 | valAccuracy:  0.471 | throughput: 133802.976 img/s |
[Epoch   2] :  .. done (196 batches)
[Epoch   2] : | time:  4.586s | trainLoss:  1.440 | trainAccuracy:  0.479 | valLoss:  1.392 | valAccuracy:  0.506 | throughput: 130137.737 img/s |
[Epoch   3] :  .. done (196 batches)
[Epoch   3] : | time:  4.581s | trainLoss:  1.352 | trainAccuracy:  0.512 | valLoss:  1.324 | valAccuracy:  0.530 | throughput: 130311.775 img/s |
[Epoch   4] :  .. done (196 batches)
[Epoch   4] : | time:  4.618s | trainLoss:  1.278 | trainAccuracy:  0.541 | valLoss:  1.254 | valAccuracy:  0.557 | throughput: 131700.278 img/s |
[Epoch   5] :  .. done (196 batches)
[Epoch   5] : | time:  4.559s | trainLoss:  1.208 | trainAccuracy:  0.568 | valLoss:  1.193 | valAccuracy:  0.582 | throughput: 130449.208 img/s |
[Epoch   6] :  .. done (196 batches)
[Epoch   6] : | time:  4.605s | trainLoss:  1.158 | trainAccuracy:  0.588 | valLoss:  1.120 | valAccuracy:  0.604 | throughput: 132074.401 img/s |
[Epoch   7] :  .. done (196 batches)
[Epoch   7] : | time:  4.617s | trainLoss:  1.095 | trainAccuracy:  0.610 | valLoss:  1.073 | valAccuracy:  0.622 | throughput: 131978.492 img/s |
[Epoch   8] :  .. done (196 batches)
[Epoch   8] : | time:  4.597s | trainLoss:  1.043 | trainAccuracy:  0.631 | valLoss:  1.044 | valAccuracy:  0.637 | throughput: 132040.753 img/s |
[Epoch   9] :  .. done (196 batches)
[Epoch   9] : | time:  4.608s | trainLoss:  0.997 | trainAccuracy:  0.648 | valLoss:  1.035 | valAccuracy:  0.641 | throughput: 132215.981 img/s |
[Epoch  10] :  .. done (196 batches)
[Epoch  10] : | time:  4.614s | trainLoss:  0.953 | trainAccuracy:  0.665 | valLoss:  1.000 | valAccuracy:  0.655 | throughput: 129597.310 img/s |
[Epoch  11] :  .. done (196 batches)
[Epoch  11] : | time:  4.596s | trainLoss:  0.908 | trainAccuracy:  0.681 | valLoss:  0.924 | valAccuracy:  0.679 | throughput: 132150.477 img/s |
[Epoch  12] :  .. done (196 batches)
[Epoch  12] : | time:  4.606s | trainLoss:  0.872 | trainAccuracy:  0.696 | valLoss:  0.897 | valAccuracy:  0.688 | throughput: 123408.232 img/s |
[Epoch  13] :  .. done (196 batches)
[Epoch  13] : | time:  4.615s | trainLoss:  0.830 | trainAccuracy:  0.709 | valLoss:  0.890 | valAccuracy:  0.693 | throughput: 132708.182 img/s |
[Epoch  14] :  .. done (196 batches)
[Epoch  14] : | time:  4.606s | trainLoss:  0.801 | trainAccuracy:  0.718 | valLoss:  0.881 | valAccuracy:  0.696 | throughput: 131591.732 img/s |
[Epoch  15] :  .. done (196 batches)
[Epoch  15] : | time:  4.594s | trainLoss:  0.770 | trainAccuracy:  0.729 | valLoss:  0.864 | valAccuracy:  0.704 | throughput: 130495.688 img/s |
[Epoch  16] :  .. done (196 batches)
[Epoch  16] : | time:  4.594s | trainLoss:  0.741 | trainAccuracy:  0.739 | valLoss:  0.837 | valAccuracy:  0.713 | throughput: 133671.470 img/s |
[Epoch  17] :  .. done (196 batches)
[Epoch  17] : | time:  4.594s | trainLoss:  0.714 | trainAccuracy:  0.750 | valLoss:  0.839 | valAccuracy:  0.715 | throughput: 132751.989 img/s |
[Epoch  18] :  .. done (196 batches)
[Epoch  18] : | time:  4.578s | trainLoss:  0.681 | trainAccuracy:  0.764 | valLoss:  0.832 | valAccuracy:  0.718 | throughput: 132079.015 img/s |
[Epoch  19] :  .. done (196 batches)
[Epoch  19] : | time:  4.606s | trainLoss:  0.660 | trainAccuracy:  0.771 | valLoss:  0.838 | valAccuracy:  0.720 | throughput: 130263.915 img/s |
[Epoch  20] :  .. done (196 batches)
[Epoch  20] : | time:  4.575s | trainLoss:  0.635 | trainAccuracy:  0.780 | valLoss:  0.827 | valAccuracy:  0.724 | throughput: 132863.951 img/s |
[Epoch  21] :  .. done (196 batches)
[Epoch  21] : | time:  4.608s | trainLoss:  0.611 | trainAccuracy:  0.786 | valLoss:  0.803 | valAccuracy:  0.729 | throughput: 134081.966 img/s |
[Epoch  22] :  .. done (196 batches)
[Epoch  22] : | time:  4.566s | trainLoss:  0.587 | trainAccuracy:  0.795 | valLoss:  0.834 | valAccuracy:  0.727 | throughput: 127359.423 img/s |
[Epoch  23] :  .. done (196 batches)
[Epoch  23] : | time:  4.578s | trainLoss:  0.564 | trainAccuracy:  0.802 | valLoss:  0.823 | valAccuracy:  0.729 | throughput: 123822.193 img/s |
[Epoch  24] :  .. done (196 batches)
[Epoch  24] : | time:  4.618s | trainLoss:  0.543 | trainAccuracy:  0.808 | valLoss:  0.835 | valAccuracy:  0.730 | throughput: 125872.396 img/s |
[Epoch  25] :  .. done (196 batches)
[Epoch  25] : | time:  4.615s | trainLoss:  0.524 | trainAccuracy:  0.815 | valLoss:  0.847 | valAccuracy:  0.729 | throughput: 124479.005 img/s |
[Epoch  26] :  .. done (196 batches)
[Epoch  26] : | time:  4.655s | trainLoss:  0.507 | trainAccuracy:  0.821 | valLoss:  0.827 | valAccuracy:  0.733 | throughput: 125991.018 img/s |
[Epoch  27] :  .. done (196 batches)
[Epoch  27] : | time:  4.561s | trainLoss:  0.485 | trainAccuracy:  0.831 | valLoss:  0.833 | valAccuracy:  0.737 | throughput: 131916.093 img/s |
[Epoch  28] :  .. done (196 batches)
[Epoch  28] : | time:  4.599s | trainLoss:  0.463 | trainAccuracy:  0.837 | valLoss:  0.832 | valAccuracy:  0.737 | throughput: 131616.945 img/s |
[Epoch  29] :  .. done (196 batches)
[Epoch  29] : | time:  4.593s | trainLoss:  0.449 | trainAccuracy:  0.841 | valLoss:  0.828 | valAccuracy:  0.743 | throughput: 131435.191 img/s |
Training finished in 0:02:20.131106 hh:mm:ss.ms
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
  - Accuracy: 0.743
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

[Epoch   0] :  .. done (196 batches)
[Epoch   0] : | time:  5.337s | trainLoss:  4.359 | trainAccuracy:  0.043 | valLoss:  4.026 | valAccuracy:  0.101 | throughput: 133925.980 img/s |
[Epoch   1] :  .. done (196 batches)
[Epoch   1] : | time:  4.565s | trainLoss:  3.782 | trainAccuracy:  0.124 | valLoss:  3.686 | valAccuracy:  0.159 | throughput: 126685.282 img/s |
[Epoch   2] :  .. done (196 batches)
[Epoch   2] : | time:  4.557s | trainLoss:  3.528 | trainAccuracy:  0.167 | valLoss:  3.480 | valAccuracy:  0.200 | throughput: 130387.115 img/s |
[Epoch   3] :  .. done (196 batches)
[Epoch   3] : | time:  4.558s | trainLoss:  3.319 | trainAccuracy:  0.208 | valLoss:  3.316 | valAccuracy:  0.223 | throughput: 130556.513 img/s |
[Epoch   4] :  .. done (196 batches)
[Epoch   4] : | time:  4.579s | trainLoss:  3.164 | trainAccuracy:  0.234 | valLoss:  3.165 | valAccuracy:  0.251 | throughput: 129780.800 img/s |
[Epoch   5] :  .. done (196 batches)
[Epoch   5] : | time:  4.554s | trainLoss:  3.027 | trainAccuracy:  0.261 | valLoss:  3.061 | valAccuracy:  0.277 | throughput: 128915.521 img/s |
[Epoch   6] :  .. done (196 batches)
[Epoch   6] : | time:  4.546s | trainLoss:  2.912 | trainAccuracy:  0.283 | valLoss:  2.960 | valAccuracy:  0.290 | throughput: 130964.879 img/s |
[Epoch   7] :  .. done (196 batches)
[Epoch   7] : | time:  4.532s | trainLoss:  2.813 | trainAccuracy:  0.303 | valLoss:  2.895 | valAccuracy:  0.306 | throughput: 129597.845 img/s |
[Epoch   8] :  .. done (196 batches)
[Epoch   8] : | time:  4.528s | trainLoss:  2.718 | trainAccuracy:  0.322 | valLoss:  2.844 | valAccuracy:  0.318 | throughput: 131585.296 img/s |
[Epoch   9] :  .. done (196 batches)
[Epoch   9] : | time:  4.550s | trainLoss:  2.626 | trainAccuracy:  0.342 | valLoss:  2.781 | valAccuracy:  0.331 | throughput: 130066.078 img/s |
[Epoch  10] :  .. done (196 batches)
[Epoch  10] : | time:  4.537s | trainLoss:  2.539 | trainAccuracy:  0.358 | valLoss:  2.705 | valAccuracy:  0.348 | throughput: 131852.050 img/s |
[Epoch  11] :  .. done (196 batches)
[Epoch  11] : | time:  4.548s | trainLoss:  2.464 | trainAccuracy:  0.372 | valLoss:  2.689 | valAccuracy:  0.353 | throughput: 131316.803 img/s |
[Epoch  12] :  .. done (196 batches)
[Epoch  12] : | time:  4.583s | trainLoss:  2.390 | trainAccuracy:  0.388 | valLoss:  2.656 | valAccuracy:  0.353 | throughput: 130145.153 img/s |
[Epoch  13] :  .. done (196 batches)
[Epoch  13] : | time:  4.543s | trainLoss:  2.323 | trainAccuracy:  0.402 | valLoss:  2.632 | valAccuracy:  0.363 | throughput: 131971.924 img/s |
[Epoch  14] :  .. done (196 batches)
[Epoch  14] : | time:  4.551s | trainLoss:  2.258 | trainAccuracy:  0.416 | valLoss:  2.648 | valAccuracy:  0.364 | throughput: 130585.220 img/s |
[Epoch  15] :  .. done (196 batches)
[Epoch  15] : | time:  4.569s | trainLoss:  2.201 | trainAccuracy:  0.428 | valLoss:  2.571 | valAccuracy:  0.371 | throughput: 128175.536 img/s |
[Epoch  16] :  .. done (196 batches)
[Epoch  16] : | time:  4.561s | trainLoss:  2.139 | trainAccuracy:  0.440 | valLoss:  2.585 | valAccuracy:  0.381 | throughput: 131303.428 img/s |
[Epoch  17] :  .. done (196 batches)
[Epoch  17] : | time:  4.581s | trainLoss:  2.084 | trainAccuracy:  0.451 | valLoss:  2.548 | valAccuracy:  0.390 | throughput: 130143.127 img/s |
[Epoch  18] :  .. done (196 batches)
[Epoch  18] : | time:  4.579s | trainLoss:  2.024 | trainAccuracy:  0.466 | valLoss:  2.549 | valAccuracy:  0.387 | throughput: 128558.155 img/s |
[Epoch  19] :  .. done (196 batches)
[Epoch  19] : | time:  4.550s | trainLoss:  1.973 | trainAccuracy:  0.479 | valLoss:  2.548 | valAccuracy:  0.395 | throughput: 131537.833 img/s |
[Epoch  20] :  .. done (196 batches)
[Epoch  20] : | time:  4.569s | trainLoss:  1.929 | trainAccuracy:  0.487 | valLoss:  2.578 | valAccuracy:  0.389 | throughput: 130492.413 img/s |
[Epoch  21] :  .. done (196 batches)
[Epoch  21] : | time:  4.572s | trainLoss:  1.874 | trainAccuracy:  0.498 | valLoss:  2.566 | valAccuracy:  0.394 | throughput: 130709.781 img/s |
[Epoch  22] :  .. done (196 batches)
[Epoch  22] : | time:  4.562s | trainLoss:  1.831 | trainAccuracy:  0.505 | valLoss:  2.541 | valAccuracy:  0.393 | throughput: 130452.736 img/s |
[Epoch  23] :  .. done (196 batches)
[Epoch  23] : | time:  4.560s | trainLoss:  1.798 | trainAccuracy:  0.517 | valLoss:  2.559 | valAccuracy:  0.398 | throughput: 131744.400 img/s |
[Epoch  24] :  .. done (196 batches)
[Epoch  24] : | time:  4.543s | trainLoss:  1.754 | trainAccuracy:  0.523 | valLoss:  2.544 | valAccuracy:  0.394 | throughput: 131091.097 img/s |
[Epoch  25] :  .. done (196 batches)
[Epoch  25] : | time:  4.559s | trainLoss:  1.711 | trainAccuracy:  0.535 | valLoss:  2.589 | valAccuracy:  0.394 | throughput: 130771.858 img/s |
[Epoch  26] :  .. done (196 batches)
[Epoch  26] : | time:  4.550s | trainLoss:  1.677 | trainAccuracy:  0.541 | valLoss:  2.587 | valAccuracy:  0.401 | throughput: 119292.514 img/s |
[Epoch  27] :  .. done (196 batches)
[Epoch  27] : | time:  4.560s | trainLoss:  1.637 | trainAccuracy:  0.548 | valLoss:  2.586 | valAccuracy:  0.401 | throughput: 127623.376 img/s |
[Epoch  28] :  .. done (196 batches)
[Epoch  28] : | time:  4.564s | trainLoss:  1.601 | trainAccuracy:  0.560 | valLoss:  2.639 | valAccuracy:  0.399 | throughput: 131426.970 img/s |
[Epoch  29] :  .. done (196 batches)
[Epoch  29] : | time:  4.574s | trainLoss:  1.566 | trainAccuracy:  0.567 | valLoss:  2.610 | valAccuracy:  0.399 | throughput: 128966.126 img/s |
Training finished in 0:02:18.093191 hh:mm:ss.ms

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
  - Accuracy: 0.399
Testing finished in 0:00:01.839540 hh:mm:ss.ms
```
