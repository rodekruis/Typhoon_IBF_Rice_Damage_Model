"""
Script to plot the collected wind Data
To see for which municipalities wind data is collected using the wind data collection script
"""

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

#%% Setting directory
os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

#%% Loading data
file_name = "IBF_typhoon_model\\data\\wind_data\\output\\historical_typhoons_wind_test.csv"
path = os.path.join(cdir, file_name)
df_wind = pd.read_csv(path)

file_name = "IBF_typhoon_model\\data\\phl_administrative_boundaries\\phl_admbnda_adm3.shp"
path = os.path.join(cdir, file_name)
df_admin = gpd.read_file(path)


# %%

df_combined = pd.merge(df_admin, df_wind, how='left', left_on=['ADM3_PCODE'], right_on = ['adm3_pcode'])
# # %%


# df_combined.plot(column='vmax_gust')
# plt.show()


#%% Creating single figure
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
ax.set_aspect("equal")

minx, miny, maxx, maxy = df_admin.total_bounds

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

df_admin.plot(ax=ax, color="whitesmoke", zorder=1)

df_combined.plot(
    ax=ax,
    column='vmax_sust',
    cmap="Reds",
    vmin=0,
    vmax=65,
    zorder=2
    # legend=True
    # cax=cax,
)

path = os.path.join(cdir, 'IBF_typhoon_model\\data\\figures\\wind_figure.png')

fig.savefig(path)

print("Done")
# %%
