# UniV-Forecasting

An open source univariate time series forecasting framework that provides following features:

* An automation framework intergrated with data preprocess, hyper-parameters setting, hyper-parameters tuning, model training, model evaluation, and experiment logging.
* An easy user-replaced model packing paradigm compatible with both statistical, stochasitc, and training models.
* Ready-to-use forecasting models, supported with both GPU acceleration or CPU only.
  * Strong basline models including CNN, RNN (Elman, GRU, LSTM), DeepAR, and Conv-LSTM.
  * Classic statistical and machine learning models including ARIMA, Holt-Winter, MLP, and MSVR.
  * Our proposed deep learning models.

### Main Dependence

---

* python >= 3.6
* pytorch = 1.9.1
* CUDA (as required as pytorch, if using GPU)
* ray = 1.6.0 (as requried by the specific optimizaiton algorithm, if using TaskTuner)
* scikit-learn = 1.0.2

### Provided models

---

* Strong deep neural networks.
* Classic statistical and machine learning models.
* Promising neural networks with random weights.
* Our proposed models.

The training models we implemented are referred to these papers.

| Model                  | Paper                                                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| DeepAR                 | [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)                   |
| ConvRNN                | [Autoregressive Convolutional Recurrent Neural Network for Univariate and Multivariate Time](https://arxiv.org/abs/1903.02540) |
| RNN (Elman, GRU, LSTM) | [Recurrent neural networks for time series forecasting: current status and future directions](https://www.sciencedirect.com/science/article/pii/S0169207020300996)                                 |
| CNN                    | [Convolutional neural networks for energy time series forecasting](https://ieeexplore.ieee.org/abstract/document/8489399/)                                                            |
| MLP                    | [PSO-MISMO modeling strategy for MultiStep-ahead time series prediction](https://ieeexplore.ieee.org/abstract/document/6553147/)                                                      |

The stochastic models we implemented are referred to these papers.

| Model    | Paper                                                                                                                                                           |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RVFL     | [A review on neural networks with random weights](https://www.sciencedirect.com/science/article/pii/S0925231217314613)                                                                                                                 |
| IELM     | [Extreme learning machine: theory and applications](https://www.sciencedirect.com/science/article/abs/pii/S0925231206000385)                                                                                                               |
| SCN      | [Stochastic configuration networks: fundamentals and algorithms](https://ieeexplore.ieee.org/abstract/document/8013920/)                                                                                                  |
| ESN      | [Optimization and applications of echo state networks with leaky-integrator neurons](https://www.sciencedirect.com/science/article/abs/pii/S089360800700041X)                                                                              |
| GESN     | [Growing echo-state network with multiple subreservoirs](https://ieeexplore.ieee.org/abstract/document/7386673/)                                                                                                          |
| DESN     | [Design of deep echo state networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608018302223)                                                                                                                              |
| PSO-GESN | [PSO-based growing echo state network](https://www.sciencedirect.com/science/article/abs/pii/S1568494619305551)                                                                                                                            |

Our proposed models are corresponding to these papers.

| Model    | Paper                                                                                                            |
| -------- | ---------------------------------------------------------------------------------------------------------------- |
| [MSVR](https://github.com/Analytics-for-Forecasting/msvr)    | [Multi-step-ahead time series prediction using multiple-output support vector regression](https://www.sciencedirect.com/science/article/abs/pii/S092523121300917X) |
| [ESM-CNN](https://github.com/XinzeZhang/TimeSeriesForecasting-torch)  | [Error-feedback stochastic modeling strategy for time series forecasting with convolutional neural networks](https://www.sciencedirect.com/science/article/abs/pii/S1568494619305551)       |
| ETO-SDNN | Growing stochastic deep neural network for time series forecasting with error-feedback triple-phase optimization |

### Acknowledgement
---

* This framework were began to be built by Xinze Zhang while he was a Ph.D student, supervised by Prof. Yukun Bao, in the school of Management, Huazhong university of Science and Technology (HUST).

Notice
* The DeepAR provided in this repository is modified based on the work of [TimeSeries](https://github.com/zhykoties/TimeSeries). Yunkai Zhang, Qiao Jianga, and Xueying Ma are original authors of [TimeSeries](https://github.com/zhykoties/TimeSeries).
* The ConvRNN provided in this repository is modified based on the work of [ConvRNN](https://github.com/KurochkinAlexey/ConvRNN), KurochkinAlexey, Fess13 are original authors of [ConvRNN](https://github.com/KurochkinAlexey/ConvRNN).
* The PSO-GESN provided in this repository is modified based on the source code created by [Qi Sima](https://github.com/simaqi18).