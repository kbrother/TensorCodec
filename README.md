# TensorCodec: Compact Lossy Compression of Tensors without Strong Data Assumptions
This repository is the official implementation of TensorCodec: Compact Lossy Compression of Tensors without Strong Data Assumptions (anonymized).

## Requirements
* To run the provided codes, you need to install `PyTorch`. Since the installation commands for the package rely on the environments, please visit the page (https://pytorch.org/get-started/locally/) for guideline to install the package.

* The code should be run on the folder (`./`) which includes the `TensorCodec` folder. The dataset files should be located in `./input`.

## Initializing the orders
Initialization of the orders of the tensor is implemented in ```init_order.py```.

### Positional argument
* `-sp`, `--save_path`: path for saving the original tensor.
* `-lp`, `--load_path`: path for saving the reordered tensor.

### Example commands and results
```
  python TensorCodec/init_order.py -lp input/action_orig.npy -sp input/action.npy

  order: 0, loss before: 2387.244534244066, loss after: 2387.244534244066
  order: 1, loss before: 46618.97891356207, loss after: 7292.559081679066
  order: 2, loss before: 20461.291436824664, loss after: 14315.363870162731
  Total elapsed time: 8.323100328445435
```

## Running TensorCodec
Training (compressing) and evaluating (decompressing) process are implemented in ```main.py```.
### Positional argument
* `action`: `train` for compressing the matrix. `test` for checking the reconstruction loss of the trained model.
* `-d`, `--dataset`: data to be compressed

### Optional arguments (common)
* `-de`, `--device`: GPU id(s) for execution.
* `-rk`, `--rank`: rank of TT cores.
* `-hs`, `--hidden_size`: size of the hidden dimension.
* `-m`, `--model`: type of the model (gru, lstm, mha). The default is lstm.
* `-nb`, `--num_batch`: the number of mini-batches for training.
* `-b`, `--batch_size`: the number of entries of the tensor which are processed simultaneosly in GPUs.
  
### Optional arguments (for training)
* `-lr`, `--lr`: learning rate.
* `-e`, `--epoch`: maximum epoch numbers.
* `-sp`, `--save_path`: path for saving the parameters of the trained model and the new orders of the indices of the tensor (bijective function from indices of the reordered tensor to the indices of the original tensor).
* `-tol`, `--tol`: tolerance for training.
 
### Example command
```
  # Training
  python TensorCodec/main.py train -d action -de 0 1 2 3 -rk 6 -hs 8 -sp output/action_r6_h8 -e 5000 -lr 1 -m lstm -nb 100 -t 100 -b 2097152

  # Evaluating
  python TensorCodec/main.py test -d action -de 0 1 2 3 -rk 6 -hs 8
```

## Evaluating the trained model
### Command
* We uploaded the trained model for the 3 smallest tensors among 3-order tensors and the smallest tensor among 4-order tensors in the folder 'trained model'.
* The hyperparameters (rank and hidden dimension) correspond to the models with the fewest parameters shown in Figure 3 of the main paper for all datasets.
* You can run the code with the following commands. Note that the device option should be changed depending on the available GPUs.
```
  python TensorCodec/main.py test -d action -de 0 1 2 3 -rk 6 -hs 8
  python TensorCodec/main.py test -d airquality -de 0 1 2 3 -rk 7 -hs 11 
  python TensorCodec/main.py test -d uber -de 0 1 2 3 -rk 8 -hs 7
  python TensorCodec/main.py test -d nyc -de 0 1 2 3 -rk 2 -hs 5
```

### Expected results
||action|airquality|uber|nyc|
|-|-|-|-|-|
|Fitness|0.65|0.648|0.669|0.558|
|Compressed Size (bytes)|11686|26031|11870|5227|

## Real-world datasets we used
|Name|shape|Density|Source|Link|
|-|-|-|-|-|
|Uber|183 x 24 x 1,140|0.138|[FROSTT](http://frostt.io/)|[Link](https://www.dropbox.com/sh/n1wv6sad7pdtvs2/AABo7r1d42btdmfkyf46FTNOa?dl=0)|
|Air Quality|5,600 x 362 x 6|0.917|[Air Korea](https://www.airkorea.or.kr/web/)|[Link](https://www.dropbox.com/sh/mph9ynz21dbjplc/AACnfvEWNC2V3vAKKB5d__Bga?dl=0)|
|Action|100 x 570 x 567|0.393|[Multivariate LSTM-FCNs](https://github.com/titu1994/MLSTM-FCN)|[Link](https://www.dropbox.com/sh/vz4dw1a1hu5i2d3/AAA18RxIZbYZKkqPEQxJRq7da?dl=0)|
|PEMS-SF|963 X 144 X 440|0.999|[The UEA & UCR Time Series Classification Repository](https://www.timeseriesclassification.com/)|[Link](https://www.dropbox.com/sh/ps0wiyrsjhfvbas/AACX_JYd4skwFcS4dYs20aiJa?dl=0)|
|Activity|337 x 570 x 320|0.569|[Multivariate LSTM-FCNs](https://github.com/titu1994/MLSTM-FCN)|[Link](https://www.dropbox.com/sh/4rsd4gp9em4vz8i/AABfjRdCkpAR8HSpjwRJgvkfa?dl=0)|
|Stock|1,317 x 88 x 916|0.816|[Zoom-Tucker](https://github.com/jungijang/KoreaStockData)|[Link](https://www.dropbox.com/sh/m812qgv1t5zjris/AAAxTz-fVItQbsLBhlGVJiRGa?dl=0)|
|NYC|265 X 265 X 28 X 35|0.118|[New York City Government](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)|[Link](https://www.dropbox.com/sh/bv2rrj6gp73z7sr/AACM2ZIvm8Bg3RuyY2mlObMia?dl=0)|
|Absorb|192 x 228 x 30 x 120|1.000|[Climate Data at the National Center for Atmospheric Research](https://www.earthsystemgrid.org)|[Link](https://www.dropbox.com/sh/5k9du01d1yvgnke/AAANN2QL0KxAmM3rRdH_yt82a?dl=0)|
