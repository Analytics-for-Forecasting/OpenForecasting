# UniV-Forecasting

An open source univariate time series forecasting framework that provides following features:

* An automation framework intergrated with data preprocess, hyper-parameters setting, hyper-parameters tuning, model training, model evaluation, and experiment logging.
* An easy user-replaced model packing paradigm compatible with both statistical, stochasitc, and training models.
* Ready-to-use forecasting models, supported with both GPU acceleration or CPU only.
  * Strong basline models including CNN, RNN (Elman, GRU, LSTM), DeepAR, and Conv-LSTM.
  * Classic statistical and machine learning models including ARIMA, Holt-Winter, MLP, and MSVR.
  * Our proposed deep learning models.

### Main Dependence

* python >= 3.6
* pytorch = 1.9.1
* CUDA (as required as pytorch, if using GPU)
* ray = 1.6.0 (as requried by the specific optimizaiton algorithm, if using TaskTuner)
* scikit-learn = 1.0.2

### Provided models

* Strong deep neural networks.
* Classic statistical and machine learning models.
* Promising neural networks with random weights.
* Our proposed models.

## Acknowledgement

- This work was done under the direction of our supervisor Prof. Yukun Bao.
