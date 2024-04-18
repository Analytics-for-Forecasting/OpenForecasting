# OpenForecasting

An open source time series forecasting framework that provides following features:

* A general framework intergrated with data preprocess, hyper-parameters setting, hyper-parameters tuning, model training, model evaluation, and experiment logging.
* An easy user-replaced model coding paradigm compatible with both statistical, stochastic, and training models.
* Ready-to-use forecasting models, supported with both GPU acceleration or CPU only.
* As for now, we only support univariable time series forecasting. In the future, the multivariable time serires forecasting will be officially provided.

The experiments need to be configured by the python files in the folder `exp`. To replicate or run the experiment in the `exp` folder, e.g., `exp/encoder/demo.py`, just execute:

```bash
python exp/encoder/demo.py
```

or 

```bash
python main.py -cuda -test -datafolder exp/encoder -dataset demo -exp_name RL -H 2 -model rnn -rep_times 1
```

## Main Dependence

To install the dependence of the running environment, using the following commands:

```bash
cd _requirement
conda create --name amc --file packages.txt
conda activate amc
conda install pip
pip install -r requirements.txt
```

For the follower in China, we suggest to config the mirror for conda and pip.

##### Conda mirror


Create the `.condarc` file if it does not exist.

```

touch ~/.condarc

```

Then copy the following mirrors to the `.condarc`:

```

channels:

  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/main

  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/free

  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/r

  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/pro

  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2

show_channel_urls: true

custom_channels:

  conda-forge: http://mirrors.bfsu.edu.cn/anaconda/cloud

  msys2: http://mirrors.bfsu.edu.cn/anaconda/cloud

  bioconda: http://mirrors.bfsu.edu.cn/anaconda/cloud

  menpo: http://mirrors.bfsu.edu.cn/anaconda/cloud

  pytorch: http://mirrors.bfsu.edu.cn/anaconda/cloud

  simpleitk: http://mirrors.bfsu.edu.cn/anaconda/cloud

  intel: http://mirrors.bfsu.edu.cn/anaconda/cloud

```

Then clean the cache and test it:

```bash
conda update --strict-channel-priority --all  
conda clean 
```

##### Pip mirror


With tencent cloud, create the pip configuration file by `mkdir ~/.pip; nano ~/.pip/pip.conf`, and paste the following:

```

[global]

index-url = https://mirrors.cloud.tencent.com/pypi/simple/


[install]

trusted-host=mirrors.cloud.tencent.com


timeout = 120

```

Save the file `pip.conf`.

## Provided models

---

* Strong deep neural networks.
* Classic statistical and machine learning models.
* Promising neural networks with random weights.
* Our proposed models.

The training models we implemented are referred to these papers.

| Model                  | Paper                                                                                                                                                           |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepAR                 | [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)                                                       |
| ConvRNN                | [Autoregressive Convolutional Recurrent Neural Network for Univariate and Multivariate Time](https://arxiv.org/abs/1903.02540)                                     |
| RNN (Elman, GRU, LSTM) | [Recurrent neural networks for time series forecasting: current status and future directions](https://www.sciencedirect.com/science/article/pii/S0169207020300996) |
| CNN                    | [Convolutional neural networks for energy time series forecasting](https://ieeexplore.ieee.org/abstract/document/8489399/)                                         |
| MLP                    | [PSO-MISMO modeling strategy for MultiStep-ahead time series prediction](https://ieeexplore.ieee.org/abstract/document/6553147/)                                   |

The stochastic models we implemented are referred to these papers.

| Model    | Paper                                                                                                                                                      |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RVFL     | [A review on neural networks with random weights](https://www.sciencedirect.com/science/article/pii/S0925231217314613)                                        |
| IELM     | [Extreme learning machine: theory and applications](https://www.sciencedirect.com/science/article/abs/pii/S0925231206000385)                                  |
| SCN      | [Stochastic configuration networks: fundamentals and algorithms](https://ieeexplore.ieee.org/abstract/document/8013920/)                                      |
| ESN      | [Optimization and applications of echo state networks with leaky-integrator neurons](https://www.sciencedirect.com/science/article/abs/pii/S089360800700041X) |
| GESN     | [Growing echo-state network with multiple subreservoirs](https://ieeexplore.ieee.org/abstract/document/7386673/)                                              |
| DESN     | [Design of deep echo state networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608018302223)                                                 |
| PSO-GESN | [PSO-based growing echo state network](https://www.sciencedirect.com/science/article/abs/pii/S1568494619305551)                                               |

Our proposed models are corresponding to these papers.

| Model                                                             | Paper                                                                                                                                                                              |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MSVR](https://github.com/Analytics-for-Forecasting/msvr)            | [Multi-step-ahead time series prediction using multiple-output support vector regression](https://www.sciencedirect.com/science/article/abs/pii/S092523121300917X)                    |
| [ESM-CNN](https://github.com/XinzeZhang/TimeSeriesForecasting-torch) | [Error-feedback stochastic modeling strategy for time series forecasting with convolutional neural networks](https://www.sciencedirect.com/science/article/abs/pii/S1568494619305551) |
| ETO-SDNN                                                          | Growing stochastic deep neural network for time series forecasting with error-feedback triple-phase optimization                                                                   |

## Acknowledgement

---

* This framework is created by [Xinze Zhang](https://github.com/xinzezhang), supervised by Prof. Yukun Bao, in the school of Management, Huazhong university of Science and Technology (HUST).

Notice

* The DeepAR provided in this repository is modified based on the work of [TimeSeries](https://github.com/zhykoties/TimeSeries). Yunkai Zhang, Qiao Jianga, and Xueying Ma are original authors of [TimeSeries](https://github.com/zhykoties/TimeSeries).
* The ConvRNN provided in this repository is modified based on the work of [ConvRNN](https://github.com/KurochkinAlexey/ConvRNN). KurochkinAlexey, Fess13 are original authors of [ConvRNN](https://github.com/KurochkinAlexey/ConvRNN).
* The PSO-GESN provided in this repository is modified based on the source code created by [Qi Sima](https://github.com/simaqi18).
