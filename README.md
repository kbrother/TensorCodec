# TensorCodec: Compact Lossy Compression of Tensors without Strong Assumptions on Data Properties
This repository is the official implementation of TensorCodec: Compact Lossy Compression of Tensors without Strong Assumptions on Data Properties (anonymized).

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
|Uber|183 x 24 x 1,140|0.138|[FROSTT](http://frostt.io/)|[Link](https://drive.google.com/file/d/1kWIA55li_QnPKCjLCg8M_v-ddPcpv8Ba/view?usp=sharing)|
|Air Quality|5,600 x 362 x 6|0.917|[Air Korea](https://www.airkorea.or.kr/web/)|[Link](https://drive.google.com/file/d/1nXvlIikJGM5J9cSJhJdOwNI7iiUNNBae/view?usp=sharing)|
|Action|100 x 570 x 567|0.393|[Fazle Karim](https://github.com/titu1994/MLSTM-FCN)|[Link](https://drive.google.com/file/d/1Sjxn0Iwh6N91gPtXarAUCmAr--CiuTf7/view?usp=sharing)|
|PEMS-SF|963 X 144 X 440|0.999|[The UEA & UCR Time Series Classification Repository](http://www.timeseriesclassification.com)|[Link](https://drive.google.com/file/d/1pzWV8oLAQFbLS9lVEZz27gO-3Fr_qoR_/view?usp=sharing)|
|Activity|337 x 570 x 320|0.569|[Fazle Karim](https://github.com/titu1994/MLSTM-FCN)|[Link](https://drive.google.com/file/d/1NyaquNHCX7NfATsOb3561TXyGlNMR4A-/view?usp=sharing)|
|Stock|1,317 x 88 x 916|0.816|[Jungi Jang](https://github.com/jungijang/KoreaStockData)|[Link](https://drive.google.com/file/d/1GYrDjybptRGw6TXux2L68-RMJKwHhJBW/view?usp=sharing)|
|NYC|265 X 265 X 28 X 35|0.118|[New York City Government](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)|[Link](https://drive.google.com/file/d/1Fof0v0E1BUeCMvtL5uQalAPeDZk9_tge/view?usp=sharing)|
|Absorb|192 x 228 x 30 x 120|1.000|[Climate Data at the National Center for Atmospheric Research](https://www.earthsystemgrid.org)|[Link](https://drive.google.com/file/d/1bL7eBNeC-EBu4HlqVh2hTltS3eWDfuyB/view?usp=sharing)|
