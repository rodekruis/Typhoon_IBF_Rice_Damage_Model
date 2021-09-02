#%% Importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import openpyxl
import os
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

#%% Loading data
file_name = (
    "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data.xlsx"
)
path = os.path.join(cdir, file_name)
df = pd.read_excel(path, engine="openpyxl")
df.head()

#%% Set zeros to missing --> looking for damage threshold
df["perc_loss"] = df["perc_loss"].replace(0, np.nan)

#%% create new boolean column for missing loss data
df["no_loss_data"] = df["perc_loss"].apply(lambda x: pd.isna(x))

#%% Remove all observations with track distance > 500
df_plot = df[df["dis_track_min"].notnull()]

#%% Reduce the data size for plotting  only relevant part
df_plot_new = df_plot[(df_plot["vmax_sust"] < 80) & (df_plot["rainfall_max"] < 200)]

#%% plot 2D distribution of wind and rainfall
# sns.displot(df_plot, x="rainfall_max", y="vmax_sust", hue="no_loss_data")
# plt.show()

sns.displot(df_plot_new, x="rainfall_max", y="vmax_sust", hue="no_loss_data")

# add circular threshold
xarr, yarr = [], []
for i in range(200):
    xs = i
    y = np.sqrt(150 ** 2 - xs ** 2) * 0.4
    # ys = y * 0.4
    ys = y
    xarr.append(xs)
    yarr.append(ys)

sns.lineplot(x=xarr, y=yarr)
plt.show()

#%% Make similar plot but now with distance
sns.displot(df, x="rainfall_max", y="dis_track_min", hue="no_loss_data")

