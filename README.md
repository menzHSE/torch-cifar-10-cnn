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

This uses an 80GB NVIDIA A100. 

```
$  python train.py
Using device: cuda
NVIDIA A100 80GB PCIe
Options:
  Device: GPU
  Seed: 0
  Batch size: 128
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
[Epoch   0] : | time: 16.984s | trainLoss:  1.861 | trainAccuracy:  0.310 | valLoss:  1.575 | valAccuracy:  0.424 | Throughput:  33628.809 img/s |
[Epoch   1] :  .... done (391 batches)
[Epoch   1] : | time:  3.512s | trainLoss:  1.485 | trainAccuracy:  0.463 | valLoss:  1.393 | valAccuracy:  0.491 | Throughput:  30652.835 img/s |
[Epoch   2] :  .... done (391 batches)
[Epoch   2] : | time:  3.241s | trainLoss:  1.339 | trainAccuracy:  0.515 | valLoss:  1.265 | valAccuracy:  0.546 | Throughput:  37853.836 img/s |
[Epoch   3] :  .... done (391 batches)
[Epoch   3] : | time:  3.324s | trainLoss:  1.225 | trainAccuracy:  0.560 | valLoss:  1.162 | valAccuracy:  0.589 | Throughput:  30052.086 img/s |
[Epoch   4] :  .... done (391 batches)
[Epoch   4] : | time:  3.175s | trainLoss:  1.127 | trainAccuracy:  0.596 | valLoss:  1.089 | valAccuracy:  0.614 | Throughput:  38988.496 img/s |
[Epoch   5] :  .... done (391 batches)
[Epoch   5] : | time:  3.291s | trainLoss:  1.042 | trainAccuracy:  0.629 | valLoss:  1.020 | valAccuracy:  0.639 | Throughput:  36773.098 img/s |
[Epoch   6] :  .... done (391 batches)
[Epoch   6] : | time:  3.310s | trainLoss:  0.976 | trainAccuracy:  0.653 | valLoss:  0.964 | valAccuracy:  0.659 | Throughput:  36509.443 img/s |
[Epoch   7] :  .... done (391 batches)
[Epoch   7] : | time:  3.206s | trainLoss:  0.913 | trainAccuracy:  0.680 | valLoss:  0.909 | valAccuracy:  0.679 | Throughput:  37675.749 img/s |
[Epoch   8] :  .... done (391 batches)
[Epoch   8] : | time:  3.284s | trainLoss:  0.863 | trainAccuracy:  0.697 | valLoss:  0.878 | valAccuracy:  0.691 | Throughput:  33247.502 img/s |
[Epoch   9] :  .... done (391 batches)
[Epoch   9] : | time:  3.421s | trainLoss:  0.815 | trainAccuracy:  0.714 | valLoss:  0.871 | valAccuracy:  0.688 | Throughput:  32885.911 img/s |
[Epoch  10] :  .... done (391 batches)
[Epoch  10] : | time:  3.291s | trainLoss:  0.769 | trainAccuracy:  0.727 | valLoss:  0.824 | valAccuracy:  0.710 | Throughput:  38367.823 img/s |
[Epoch  11] :  .... done (391 batches)
[Epoch  11] : | time:  3.080s | trainLoss:  0.728 | trainAccuracy:  0.744 | valLoss:  0.797 | valAccuracy:  0.725 | Throughput:  38407.942 img/s |
[Epoch  12] :  .... done (391 batches)
[Epoch  12] : | time:  3.185s | trainLoss:  0.694 | trainAccuracy:  0.756 | valLoss:  0.828 | valAccuracy:  0.717 | Throughput:  37978.836 img/s |
[Epoch  13] :  .... done (391 batches)
[Epoch  13] : | time:  3.101s | trainLoss:  0.657 | trainAccuracy:  0.768 | valLoss:  0.766 | valAccuracy:  0.740 | Throughput:  39445.433 img/s |
[Epoch  14] :  .... done (391 batches)
[Epoch  14] : | time:  3.223s | trainLoss:  0.627 | trainAccuracy:  0.779 | valLoss:  0.772 | valAccuracy:  0.734 | Throughput:  36641.787 img/s |
[Epoch  15] :  .... done (391 batches)
[Epoch  15] : | time:  3.163s | trainLoss:  0.590 | trainAccuracy:  0.792 | valLoss:  0.753 | valAccuracy:  0.745 | Throughput:  33410.927 img/s |
[Epoch  16] :  .... done (391 batches)
[Epoch  16] : | time:  3.228s | trainLoss:  0.558 | trainAccuracy:  0.805 | valLoss:  0.728 | valAccuracy:  0.752 | Throughput:  35762.769 img/s |
[Epoch  17] :  .... done (391 batches)
[Epoch  17] : | time:  3.205s | trainLoss:  0.533 | trainAccuracy:  0.813 | valLoss:  0.723 | valAccuracy:  0.756 | Throughput:  35364.081 img/s |
[Epoch  18] :  .... done (391 batches)
[Epoch  18] : | time:  3.081s | trainLoss:  0.504 | trainAccuracy:  0.821 | valLoss:  0.736 | valAccuracy:  0.757 | Throughput:  33879.809 img/s |
[Epoch  19] :  .... done (391 batches)
[Epoch  19] : | time:  3.251s | trainLoss:  0.476 | trainAccuracy:  0.830 | valLoss:  0.734 | valAccuracy:  0.758 | Throughput:  35239.992 img/s |
[Epoch  20] :  .... done (391 batches)
[Epoch  20] : | time:  3.128s | trainLoss:  0.453 | trainAccuracy:  0.841 | valLoss:  0.735 | valAccuracy:  0.757 | Throughput:  34609.982 img/s |
[Epoch  21] :  .... done (391 batches)
[Epoch  21] : | time:  3.170s | trainLoss:  0.430 | trainAccuracy:  0.847 | valLoss:  0.735 | valAccuracy:  0.764 | Throughput:  33156.163 img/s |
[Epoch  22] :  .... done (391 batches)
[Epoch  22] : | time:  3.191s | trainLoss:  0.408 | trainAccuracy:  0.855 | valLoss:  0.747 | valAccuracy:  0.763 | Throughput:  36893.595 img/s |
[Epoch  23] :  .... done (391 batches)
[Epoch  23] : | time:  3.080s | trainLoss:  0.390 | trainAccuracy:  0.863 | valLoss:  0.751 | valAccuracy:  0.763 | Throughput:  35040.866 img/s |
[Epoch  24] :  .... done (391 batches)
[Epoch  24] : | time:  3.393s | trainLoss:  0.367 | trainAccuracy:  0.870 | valLoss:  0.745 | valAccuracy:  0.771 | Throughput:  33444.080 img/s |
[Epoch  25] :  .... done (391 batches)
[Epoch  25] : | time:  3.348s | trainLoss:  0.353 | trainAccuracy:  0.873 | valLoss:  0.755 | valAccuracy:  0.766 | Throughput:  34724.576 img/s |
[Epoch  26] :  .... done (391 batches)
[Epoch  26] : | time:  3.234s | trainLoss:  0.331 | trainAccuracy:  0.881 | valLoss:  0.775 | valAccuracy:  0.769 | Throughput:  37748.088 img/s |
[Epoch  27] :  .... done (391 batches)
[Epoch  27] : | time:  3.134s | trainLoss:  0.316 | trainAccuracy:  0.888 | valLoss:  0.754 | valAccuracy:  0.771 | Throughput:  36358.492 img/s |
[Epoch  28] :  .... done (391 batches)
[Epoch  28] : | time:  3.284s | trainLoss:  0.303 | trainAccuracy:  0.891 | valLoss:  0.809 | valAccuracy:  0.769 | Throughput:  35445.559 img/s |
[Epoch  29] :  .... done (391 batches)
[Epoch  29] : | time:  3.254s | trainLoss:  0.291 | trainAccuracy:  0.896 | valLoss:  0.799 | valAccuracy:  0.770 | Throughput:  36884.863 img/s |
Training finished in 0:01:51.102646 hh:mm:ss.ms
```

### Test on CIFAR-10
```
$ python test.py --model=model_CIFAR-10_029.pth
Using device: cuda
NVIDIA A100 80GB PCIe
Files already downloaded and verified
Files already downloaded and verified
Loaded model for CIFAR-10 from model_CIFAR-10_029.pth
Testing ....
Test Metrics (10000 test samples):
  - Accuracy: 0.770
Testing finished in 0:00:02.605735 hh:mm:ss.ms
```

## CIFAR-100

### Train on CIFAR-100
`python train.py --dataset CIFAR-100`

This uses an 80GB NVIDIA A100. 

```
$ python train.py  --dataset=CIFAR-100
python train.py --dataset CIFAR-100
Using device: cuda
NVIDIA A100 80GB PCIe
Options:
  Device: GPU
  Seed: 0
  Batch size: 128
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
[Epoch   0] : | time:  3.592s | trainLoss:  4.243 | trainAccuracy:  0.053 | valLoss:  3.885 | valAccuracy:  0.114 | Throughput:  33358.793 img/s |
[Epoch   1] :  .... done (391 batches)
[Epoch   1] : | time:  3.361s | trainLoss:  3.618 | trainAccuracy:  0.148 | valLoss:  3.474 | valAccuracy:  0.188 | Throughput:  35651.599 img/s |
[Epoch   2] :  .... done (391 batches)
[Epoch   2] : | time:  3.121s | trainLoss:  3.303 | trainAccuracy:  0.205 | valLoss:  3.212 | valAccuracy:  0.232 | Throughput:  39670.841 img/s |
[Epoch   3] :  .... done (391 batches)
[Epoch   3] : | time:  3.284s | trainLoss:  3.091 | trainAccuracy:  0.244 | valLoss:  3.033 | valAccuracy:  0.267 | Throughput:  37392.047 img/s |
[Epoch   4] :  .... done (391 batches)
[Epoch   4] : | time:  3.185s | trainLoss:  2.926 | trainAccuracy:  0.275 | valLoss:  2.934 | valAccuracy:  0.288 | Throughput:  36484.106 img/s |
[Epoch   5] :  .... done (391 batches)
[Epoch   5] : | time:  3.230s | trainLoss:  2.785 | trainAccuracy:  0.304 | valLoss:  2.829 | valAccuracy:  0.305 | Throughput:  39128.082 img/s |
[Epoch   6] :  .... done (391 batches)
[Epoch   6] : | time:  3.179s | trainLoss:  2.657 | trainAccuracy:  0.331 | valLoss:  2.757 | valAccuracy:  0.321 | Throughput:  39242.447 img/s |
[Epoch   7] :  .... done (391 batches)
[Epoch   7] : | time:  3.205s | trainLoss:  2.545 | trainAccuracy:  0.354 | valLoss:  2.669 | valAccuracy:  0.343 | Throughput:  37550.685 img/s |
[Epoch   8] :  .... done (391 batches)
[Epoch   8] : | time:  3.158s | trainLoss:  2.440 | trainAccuracy:  0.377 | valLoss:  2.585 | valAccuracy:  0.364 | Throughput:  37427.010 img/s |
[Epoch   9] :  .... done (391 batches)
[Epoch   9] : | time:  3.239s | trainLoss:  2.345 | trainAccuracy:  0.396 | valLoss:  2.591 | valAccuracy:  0.361 | Throughput:  36962.480 img/s |
[Epoch  10] :  .... done (391 batches)
[Epoch  10] : | time:  3.226s | trainLoss:  2.263 | trainAccuracy:  0.413 | valLoss:  2.502 | valAccuracy:  0.373 | Throughput:  38853.829 img/s |
[Epoch  11] :  .... done (391 batches)
[Epoch  11] : | time:  3.266s | trainLoss:  2.175 | trainAccuracy:  0.431 | valLoss:  2.465 | valAccuracy:  0.387 | Throughput:  37917.850 img/s |
[Epoch  12] :  .... done (391 batches)
[Epoch  12] : | time:  3.219s | trainLoss:  2.108 | trainAccuracy:  0.444 | valLoss:  2.511 | valAccuracy:  0.386 | Throughput:  38362.656 img/s |
[Epoch  13] :  .... done (391 batches)
[Epoch  13] : | time:  3.146s | trainLoss:  2.032 | trainAccuracy:  0.461 | valLoss:  2.445 | valAccuracy:  0.395 | Throughput:  37668.505 img/s |
[Epoch  14] :  .... done (391 batches)
[Epoch  14] : | time:  3.180s | trainLoss:  1.965 | trainAccuracy:  0.477 | valLoss:  2.457 | valAccuracy:  0.398 | Throughput:  39990.809 img/s |
[Epoch  15] :  .... done (391 batches)
[Epoch  15] : | time:  3.457s | trainLoss:  1.897 | trainAccuracy:  0.492 | valLoss:  2.461 | valAccuracy:  0.399 | Throughput:  34414.344 img/s |
[Epoch  16] :  .... done (391 batches)
[Epoch  16] : | time:  3.214s | trainLoss:  1.838 | trainAccuracy:  0.504 | valLoss:  2.416 | valAccuracy:  0.409 | Throughput:  40104.814 img/s |
[Epoch  17] :  .... done (391 batches)
[Epoch  17] : | time:  3.215s | trainLoss:  1.794 | trainAccuracy:  0.513 | valLoss:  2.445 | valAccuracy:  0.412 | Throughput:  37948.732 img/s |
[Epoch  18] :  .... done (391 batches)
[Epoch  18] : | time:  3.141s | trainLoss:  1.727 | trainAccuracy:  0.524 | valLoss:  2.430 | valAccuracy:  0.411 | Throughput:  38467.844 img/s |
[Epoch  19] :  .... done (391 batches)
[Epoch  19] : | time:  3.249s | trainLoss:  1.684 | trainAccuracy:  0.537 | valLoss:  2.422 | valAccuracy:  0.420 | Throughput:  37616.772 img/s |
[Epoch  20] :  .... done (391 batches)
[Epoch  20] : | time:  3.154s | trainLoss:  1.629 | trainAccuracy:  0.549 | valLoss:  2.446 | valAccuracy:  0.416 | Throughput:  38231.580 img/s |
[Epoch  21] :  .... done (391 batches)
[Epoch  21] : | time:  3.332s | trainLoss:  1.581 | trainAccuracy:  0.560 | valLoss:  2.444 | valAccuracy:  0.419 | Throughput:  37472.352 img/s |
[Epoch  22] :  .... done (391 batches)
[Epoch  22] : | time:  3.230s | trainLoss:  1.531 | trainAccuracy:  0.572 | valLoss:  2.446 | valAccuracy:  0.416 | Throughput:  38628.134 img/s |
[Epoch  23] :  .... done (391 batches)
[Epoch  23] : | time:  3.168s | trainLoss:  1.493 | trainAccuracy:  0.580 | valLoss:  2.464 | valAccuracy:  0.424 | Throughput:  37345.034 img/s |
[Epoch  24] :  .... done (391 batches)
[Epoch  24] : | time:  3.184s | trainLoss:  1.455 | trainAccuracy:  0.588 | valLoss:  2.479 | valAccuracy:  0.421 | Throughput:  38034.317 img/s |
[Epoch  25] :  .... done (391 batches)
[Epoch  25] : | time:  3.283s | trainLoss:  1.410 | trainAccuracy:  0.598 | valLoss:  2.442 | valAccuracy:  0.424 | Throughput:  38234.314 img/s |
[Epoch  26] :  .... done (391 batches)
[Epoch  26] : | time:  3.124s | trainLoss:  1.379 | trainAccuracy:  0.608 | valLoss:  2.499 | valAccuracy:  0.424 | Throughput:  38497.559 img/s |
[Epoch  27] :  .... done (391 batches)
[Epoch  27] : | time:  3.138s | trainLoss:  1.350 | trainAccuracy:  0.616 | valLoss:  2.484 | valAccuracy:  0.430 | Throughput:  38484.853 img/s |
[Epoch  28] :  .... done (391 batches)
[Epoch  28] : | time:  3.153s | trainLoss:  1.316 | trainAccuracy:  0.620 | valLoss:  2.518 | valAccuracy:  0.433 | Throughput:  39567.019 img/s |
[Epoch  29] :  .... done (391 batches)
[Epoch  29] : | time:  3.383s | trainLoss:  1.280 | trainAccuracy:  0.629 | valLoss:  2.577 | valAccuracy:  0.423 | Throughput:  35265.599 img/s |
Training finished in 0:01:37.227602 hh:mm:ss.ms

```

### Test on CIFAR-100
```
$ python test.py --model=model_CIFAR-100_029.pth  --dataset=CIFAR-100
Using device: cuda
NVIDIA A100 80GB PCIe
Files already downloaded and verified
Files already downloaded and verified
Loaded model for CIFAR-100 from model_CIFAR-100_029.pth
Testing ....
Test Metrics (10000 test samples):
  - Accuracy: 0.423
Testing finished in 0:00:02.488181 hh:mm:ss.ms
```
