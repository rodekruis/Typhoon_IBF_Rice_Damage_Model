#%% Loading libraries
import pandas as pd
import numpy as np
import random
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isnull
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import openpyxl

#%% Importing data
os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

file_name = "IBF_typhoon_model\\data\\combined_input_data\\input_data.xlsx"
path = os.path.join(cdir, file_name)
df = pd.read_excel(path, engine="openpyxl")

"""
Checking for errors in the Rice Data
"""
# %% Rice variables
df_rice = df[["rice_area", "area_affected", "perc_loss"]]
df_rice.describe()

# %% Checking for errors in reported area affected
df_big = df.nlargest(30, "area_affected")
df_big.head()

# %% Checking for errors in the percentage loss
# Shows that large percentage loss often corresponds to a small rice area
# When the rice area is small --> more error prone
df_big = df.nlargest(30, "perc_loss")
df_big.head()

#%% Plotting confirms this
plt.scatter(x=df["perc_loss"], y=df["rice_area"])
plt.show()

# %% For small rice area, the measurement can be inaccurate
# lowest 25 percent
df_small_area = df_rice[df_rice["rice_area"] < 64]
plt.scatter(x=df_small_area["perc_loss"], y=df_small_area["rice_area"])
plt.show()

# %% Rule for dropping observations
# Drop observations with area < 5 ha
# TODO Create a new rule based on: area below 'x' and perc_loss above 'x'?
df_new = df[df["rice_area"] > 5]

#%% Convert percentage loss that is above 1 to 1
df_new.loc[:, "perc_loss"] = df_new["perc_loss"].apply(lambda x: np.min([x, 1]))

# %% Save to excel file
df_new.to_excel(
    "IBF_typhoon_model\\data\\combined_input_data\\input_data.xlsx", index=False
)
# %%
