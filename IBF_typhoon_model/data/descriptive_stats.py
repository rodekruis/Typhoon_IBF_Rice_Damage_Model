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
df = pd.read_excel(path, engine='openpyxl')


# %% Create histogram of area affected

df_big = df[df['area_affected'] > 60000]

# %%
plt.hist(df_big['area_affected'], bins=100)
plt.show()
# %%
df.nlargest(20, 'area_affected')
# %%
