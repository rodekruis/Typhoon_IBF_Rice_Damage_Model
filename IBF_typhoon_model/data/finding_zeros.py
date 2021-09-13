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
    "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data_05.xlsx"
)
path = os.path.join(cdir, file_name)
df = pd.read_excel(path, engine="openpyxl")
df.head()

#%% Remove all municipalities that never reported damage
# not_null = df[df["perc_loss"].notnull()]
# mun_not_null = not_null["mun_code"].unique()
# df = df[df["mun_code"].isin(mun_not_null)]

#%% Remove all observations with track distance > 500
df = df[df["dis_track_min"] <= 500]

i = df[((df.mun_code == "PH072251000") & (df.typhoon == "lingling2014"))].index
df = df.drop(i)

"""
Figure based on observation above zero
"""
# region
#%%
df_plot = df.copy()

#%% Set zeros to missing --> looking for damage threshold
df_plot["perc_loss"] = df_plot["perc_loss"].replace(0, np.nan)

# #%% Setting small values to zero --> np.nan
# df_plot.perc_loss[df_plot['perc_loss'] < 0.05] = np.nan

#%% create new boolean column for missing loss data
df_plot["no_loss_data"] = df_plot["perc_loss"].apply(lambda x: pd.isna(x))

#%% Plotting for all data points
sns.displot(df_plot, x="rainfall_max_6h", y="vmax", hue="no_loss_data")
plt.title("Distribution of loss data observed or not-observed")
plt.show()

#%% Reduce the data size for plotting  only relevant part
df_plot_new = df_plot[(df_plot["vmax"] < 40) & (df_plot["rainfall_max_6h"] < 15)]
sns.displot(df_plot_new, x="rainfall_max_6h", y="vmax", hue="no_loss_data")

x_max = 75
y_max = 27.5

xarr, yarr = [], []
for i in range(200):
    xs = i
    y = np.sqrt(x_max ** 2 - xs ** 2) * (y_max / x_max)
    ys = y
    xarr.append(xs)
    yarr.append(ys)

# sns.lineplot(x=xarr, y=yarr)

plt.title("Zoomed in distribution of data observed or not-observed")
plt.show()

#%% Only damage observations to show plot is incorrect
df_plot_new = df_plot[(df_plot["vmax"] < 20) & (df_plot["rainfall_max_6h"] < 15)]
sns.displot(
    df_plot_new[df_plot_new["no_loss_data"] == False],
    x="rainfall_max_6h",
    y="vmax",
    hue="no_loss_data",
)

x_max = 5
y_max = 8

xarr, yarr = [], []
for i in range(200):
    xs = i
    y = np.sqrt(x_max ** 2 - xs ** 2) * (y_max / x_max)
    ys = y
    xarr.append(xs)
    yarr.append(ys)

sns.lineplot(x=xarr, y=yarr)


plt.title(
    "Distribution of the loss data - \n only observations for which data was observed"
)
plt.show()

#%%Testing
x_max = 4
y_max = 8

# df_plot = df_plot[df_plot['no_loss_data']==False]

df_new = df_plot[
    df_plot["vmax"]
    < np.sqrt(x_max ** 2 - df_plot["rainfall_max_6h"] ** 2) * (y_max / x_max)
]

df_new[df_new["perc_loss"].notnull()]

# endregion
#%%
"""
Figure based on observing damage above 30 percent
"""

#%%
df_plot = df.copy()

#%% Drop outlier
i = df_plot[
    ((df_plot.mun_code == "PH072251000") & (df_plot.typhoon == "lingling2014"))
].index
df_plot = df_plot.drop(i)

#%% Create boolean for damage threshold
df_plot["damage_above_30"] = df_plot["perc_loss"].apply(lambda x: x > 0.3)

#%% Plotting the datapoints without an observed value
sns.displot(df_plot[df_plot["perc_loss"].isnull()], x="rainfall_max_6h", y="vmax")
plt.title(
    "Distribution plot of all entries in the dataset  \nthat have a missing value for damage observed"
)
plt.show()

#%% Plotting for all data points
sns.displot(df_plot, x="rainfall_max_6h", y="vmax", hue="damage_above_30")
plt.title(
    "Distribution plot of entries with damage \n above and below 30 percent threshold"
)
plt.show()

#%% Reduce the data size for plotting  only relevant part

df_plot_new = df_plot.copy()
# df_plot_new = df_plot[df_plot["damage_above_30"] == True]


df_plot_new = df_plot_new[
    (df_plot_new["vmax"] < 20) & (df_plot_new["rainfall_max_6h"] < 15)
]
sns.displot(df_plot_new, x="rainfall_max_6h", y="vmax", hue="damage_above_30")
plt.title(
    "Zoomed in distribution plot of all entries in the dataset  \nthat have a damage above 30%"
)

x_max = 10
y_max = 6

xarr, yarr = [], []
for i in range(200):
    xs = i
    y = np.sqrt(x_max ** 2 - xs ** 2) * (y_max / x_max)
    ys = y
    xarr.append(xs)
    yarr.append(ys)

sns.lineplot(x=xarr, y=yarr)

plt.show()

#%% Testing the dataframe
x_max = 10
y_max = 6
df_new = df[
    df["vmax"] < np.sqrt(x_max ** 2 - df["rainfall_max_6h"] ** 2) * (y_max / x_max)
]
df_new[df_new["perc_loss"].isnull()]


#%% Only damage above 30 observations to show plot is correct
df_plot_new = df_plot[(df_plot["vmax"] < 20) & (df_plot["rainfall_max_6h"] < 80)]
sns.displot(
    df_plot_new[df_plot_new["damage_above_30"] == True],
    x="rainfall_max_6h",
    y="vmax",
    hue="damage_above_30",
)
x_max = 50
y_max = 17.5

xarr, yarr = [], []
for i in range(200):
    xs = i
    y = np.sqrt(x_max ** 2 - xs ** 2) * (y_max / x_max)
    ys = y
    xarr.append(xs)
    yarr.append(ys)

sns.lineplot(x=xarr, y=yarr)
plt.show()


"""
Creating histogram for class distribution
"""
#%% Loading data
file_name = (
    "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data_04.xlsx"
)
path = os.path.join(cdir, file_name)
df = pd.read_excel(path, engine="openpyxl")
df = df[df["dis_track_min"] <= 500]
df.head()

#%% Defining functions
def class_value(x):
    if x >= 0.3:
        class_v = 1
    elif x < 0.3:
        class_v = 0
    else:
        class_v = np.nan
    return class_v


def set_zeros(x):

    x_max = 50
    y_max = 17.5

    perc_loss = x["perc_loss"]
    vmax = x["vmax"]
    rainfall_max_6h = x["rainfall_max_6h"]

    if pd.notnull(perc_loss):
        value = perc_loss
    elif vmax < np.sqrt(x_max ** 2 - rainfall_max_6h ** 2) * (y_max / x_max):
        value = 0
    else:
        value = np.nan

    return value


#%% Without transforming NaN to class value
df["class"] = df["perc_loss"].apply(class_value)
df["class"].value_counts().plot(kind="bar")
plt.title("Class distribution - original data")
plt.show()


# %% When transforming NaN to class value --> all municipalities
df["perc_loss_new"] = df[["vmax", "rainfall_max_6h", "perc_loss"]].apply(
    set_zeros, axis="columns"
)

df["class"] = df["perc_loss_new"].apply(class_value)
df["class"].value_counts().plot(kind="bar")
plt.title("Class distribution - zeros added based on all municipalities")
plt.show()

# %% When transforming NaN to class value --> only municipalities that occur in the data

# Remove all municipalities that never reported damage
not_null = df[df["perc_loss"].notnull()]
mun_not_null = not_null["mun_code"].unique()
df = df[df["mun_code"].isin(mun_not_null)]

df["perc_loss_new"] = df[["vmax", "rainfall_max_6h", "perc_loss"]].apply(
    set_zeros, axis="columns"
)

df["class"] = df["perc_loss_new"].apply(class_value)
df["class"].value_counts().plot(kind="bar")
plt.title(
    "Class distribution - zeros added based on \n municipalities that have a value observed at least once"
)
plt.show()

# %%

