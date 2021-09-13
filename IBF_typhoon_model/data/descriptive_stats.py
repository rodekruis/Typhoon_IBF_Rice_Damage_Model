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

# Combined input data
file_name = (
    "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data_05.xlsx"
)
path = os.path.join(cdir, file_name)
df = pd.read_excel(path, engine="openpyxl")

# Typhoon information
file_name = "IBF_typhoon_model\\data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
df_typh_overview = pd.read_excel(path, sheet_name="typhoon_overview", engine="openpyxl")

# Admin boundaries
file_name = (
    "IBF_typhoon_model\\data\\phl_administrative_boundaries\\phl_admbnda_adm3.shp"
)
path = os.path.join(cdir, file_name)
df_phil = gpd.read_file(path)


"""
Typhoon information
"""
#%%
print(df_typh_overview.columns)
typhoons = df_typh_overview["name_year"]
df_binary = df[df["damage_above_30"].notnull()]
year_count = df_binary.groupby(["year"]).count()
display(year_count)


"""
Typhoon year overview
"""
#%%

# df.groupby(['year', 'typhoon'])['damage_above_30'].count()

#%%
"""
Descriptive statistic
"""
#%% Histogram of percentage loss
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)

ax.hist(df["perc_loss"], bins=100)

file_name = "IBF_typhoon_model\\data\\figures\\perc_loss_hist.png"
path = os.path.join(cdir, file_name)
fig.savefig(path)
print("Done")

#%% Histogram of percentage loss without zeros
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)

df_plot = df[df["perc_loss"] > 0]

ax.hist(df_plot["perc_loss"], bins=100)

file_name = "IBF_typhoon_model\\data\\figures\\perc_loss_hist_nonzero.png"
path = os.path.join(cdir, file_name)
fig.savefig(path)
print("Done")

"""
Regions Covered
"""
#%% Create plot
mun_codes = df["mun_code"].unique()
df_phil_covered = df_phil[df_phil["ADM3_PCODE"].isin(mun_codes)]

fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
ax.set_aspect("equal")

minx, miny, maxx, maxy = df_phil.total_bounds

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

df_phil.plot(ax=ax, color="lightgrey", zorder=1)

df_phil_covered.plot(ax=ax, color="royalblue", zorder=2)

ax.axis("off")
# ax.set_title('Areas for which data is collected')


file_name = "IBF_typhoon_model\\data\\figures\\area_covered.png"
path = os.path.join(cdir, file_name)
fig.savefig(path)
print("Done")

"""
Typhoon Tracks
"""
#%%Read file
file_name = "IBF_typhoon_model\\data\\gis_data\\typhoon_tracks\\tracks_filtered.shp"
path = os.path.join(cdir, file_name)
df_tracks = gpd.read_file(path)

#%%Create random colors
random.seed(1)

tracks = df_tracks["SID"].unique()
colors = []

for track in range(len(tracks)):
    rgb = np.random.rand(3,)
    colors.append(rgb)

color_dict = dict(zip(tracks, colors))
df_tracks["color"] = df_tracks["SID"].map(color_dict)

#%%Plotting
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)

ax.set_aspect("equal")


minx, miny, maxx, maxy = df_phil.total_bounds

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)


df_tracks.plot(ax=ax, zorder=3, color=df_tracks["color"], linewidth=0.5)

df_phil.plot(ax=ax, color="lightgrey", zorder=1)

df_phil_covered.plot(ax=ax, color="royalblue", zorder=2)

ax.axis("off")

file_name = "IBF_typhoon_model\\data\\figures\\area_covered_tracks.png"
path = os.path.join(cdir, file_name)
fig.savefig(path)
print("Done")


#%%
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
