# TensorCodec: Compact Lossy Compression of Tensors without Strong Data Assumptions
This repository is the official implementation of TensorCodec: Compact Lossy Compression of Tensors without Strong Data Assumptions (anonymized).

## Requirements
* To run the provided codes, you need to install `PyTorch`. Since the installation commands for the package rely on the environments, please visit the page (https://pytorch.org/get-started/locally/) for guideline to install the package.

* The code should be run on the folder (`./`) which includes the `TensorCodec` folder. The dataset files should be located in `./input`.

## Arguments for training and evluation
### Positional argument
* `action`: `train` for compressing the matrix.
* `-d`, `--dataset`: data to be compressed

### Optional arguments (common)
* `-de`, `--device`: GPU id(s) for execution.
* `-rk`, `--rank`: rank of TT cores.
* `-hs`, `--hidden_size`: hidden size of LSTM.

### Optional arguments (for training)
* `-lr`, `--lr`: learning rate.
* `-e`, `--epoch`: maximum numbers of epoch.
* `-b', `--batch_size`: number of entries of tensors parallely processed by GPU when computing the loss.
* `-nb`, `--num_batch`: the number of mini-batches used to compress a tensor (train a model).
* `-sp`, `--save_path`: path for saving the parameters of the trained model and the new orders of the indices of the tensor (bijecive fuction from indices of the reordred tensor to the indices of the original tensor).
* `-tol`, `--tol`: tolerance for training.

## Real-world datasets we used

|Name|shape|Density|Source|Link|
|-|-|-|-|-|
|Uber|183 x 24 x 1,140|0.138|[FROSTT](http://frostt.io/)|[Link](https://www.dropbox.com/s/cvf6nrkmmwfe00w/uber_orig.npy?dl=0)|
|Air Quality|5,600 x 362 x 6|0.917|[Air Korea](https://www.airkorea.or.kr/web/)|[Link](https://www.dropbox.com/s/0a6ys590taefj8t/airquality_orig.npy?dl=0)|
|Action|100 x 570 x 567|0.393|[Multivariate LSTM-FCNs](https://github.com/titu1994/MLSTM-FCN)|[Link](https://www.dropbox.com/s/q1fpw14tjpxfa0x/action_orig.npy?dl=0)|
|PEMS-SF|963 X 144 X 440|0.999|[The UEA & UCR Time Series Classification Repository](https://www.timeseriesclassification.com/)|[Link](https://www.dropbox.com/s/n4kj8ajw9j8mefa/pems_orig.npy?dl=0)|
|Activity|337 x 570 x 320|0.569|[Multivariate LSTM-FCNs](https://github.com/titu1994/MLSTM-FCN)|[Link](https://www.dropbox.com/s/muoyj7a1utajhs9/activity_orig.npy?dl=0)|
|Stock|1,317 x 88 x 916|0.816|[Zoom-Tucker](https://github.com/jungijang/KoreaStockData)|[Link](https://www.dropbox.com/s/kpesk1nn0a92lrq/kstock_orig.npy?dl=0)|
|NYC|265 X 265 X 28 X 35|0.118|[New York City Government](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)|[Link](https://www.dropbox.com/s/9n9llimrxvyln76/nyc_orig.npy?dl=0)|
|Absorb|192 x 228 x 30 x 120|1.000|[Climate Data at the National Center for Atmospheric Research](https://www.earthsystemgrid.org)|[Link](https://www.dropbox.com/s/wpmyt5wt5ca08i5/absorb_orig.npy?dl=0)|
