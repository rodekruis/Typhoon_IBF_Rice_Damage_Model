#%% Importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import openpyxl
import os
import matplotlib.pyplot as plt

#%% Loading data
cdir = os.getcwd()
file_name = "data\\combined_input_data\\input_data.xlsx"
path = os.path.join(cdir, file_name)
df = pd.read_excel(path, engine="openpyxl")
df.head()

#%% Set zeros to missing --> looking for damage threshold
df["perc_loss"] = df["perc_loss"].replace(0, np.nan)

#%% create new boolean column for missing loss data
df["no_loss_data"] = df["perc_loss"].apply(lambda x: pd.isna(x))

#%%
df = df[df['vmax_sust'] < 50]

#%% plot 2D distribution of wind and rainfall
sns.displot(df, x="rainfall_max", y="vmax_sust", hue="no_loss_data")

# add circular threshold
xarr, yarr = [], []
for i in range(101):
    xs = i * 2
    y = np.sqrt(200 ** 2 - xs ** 2)
    ys = y * 0.1
    xarr.append(xs)
    yarr.append(ys)

sns.lineplot(x=xarr, y=yarr)
# plt.ylim(-1, 25)
# plt.xlim(-1, 100)
plt.show()

#%% Make similar plot but now with distance
sns.displot(df, x="rainfall_max", y="dis_track_min", hue="no_loss_data")

