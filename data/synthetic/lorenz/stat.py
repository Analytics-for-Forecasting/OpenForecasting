# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
# %%
data_path = 'data/synthetic/lorenz/Lorenz.txt'

df = pd.read_csv(data_path, header=None, index_col=None)

# %%
ts = df[0].values
# %%
import statsmodels.api as sm
fig = plt.figure(figsize=(16,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

sm.graphics.tsa.plot_acf(ts, lags=100, ax =ax1)
sm.graphics.tsa.plot_pacf(ts, lags=100, ax=ax2)

# %%
print(np.power(2, 36))
# %%
