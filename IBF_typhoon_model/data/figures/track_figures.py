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

os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

"""
Loading Data
"""

#%% Damage sheet
file_name = "IBF_typhoon_model\\data\\combined_input_data\\input_data.xlsx"
path = os.path.join(cdir, file_name)
df_total = pd.read_excel(path, engine="openpyxl")

# typhoon overview sheet
file_name = "IBF_typhoon_model\\data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
df_typhoons = pd.read_excel(path, sheet_name="typhoon_overview", engine="openpyxl")

# typhoon tracks shapefile
file_name = "IBF_typhoon_model\\data\\gis_data\\typhoon_tracks\\tracks_filtered.shp"
path = os.path.join(cdir, file_name)
df_tracks = gpd.read_file(path)

# map for philippines
name = "IBF_typhoon_model\\data\\phl_administrative_boundaries\\phl_admbnda_adm3.shp"
path = os.path.join(cdir, name)
df_phil = gpd.read_file(path)

# world map
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


"""
Creating damages figures: obtain DF
"""

#%%Drop NA's from df
df_total = df_total[df_total["perc_loss"].notnull()]

#%%Add damage per typhoon
sid_dict = dict(zip(df_typhoons["name_year"], df_typhoons["storm_id"]))
typhoon_dict = dict(zip(df_typhoons["storm_id"], df_typhoons["name_year"]))

sid_list = df_total["storm_id"].unique().tolist()
typhoons = df_total["typhoon"].unique().tolist()

# Allign column names of shapefile and damage df
df_phil.rename(columns={"ADM3_PCODE": "mun_code"}, inplace=True)
df_phil_damage = df_phil.copy()

# Add damage data
for sid in sid_list:

    damage_temp = df_total[df_total["storm_id"] == sid]
    df_phil_damage = df_phil_damage.merge(
        damage_temp[["mun_code", "perc_loss"]], how="left", on="mun_code",
    )
    df_phil_damage.rename(columns={"perc_loss": typhoon_dict[sid]}, inplace=True)


"""
Creating damages figures: Creating single figure
"""
# region
#%% Creating single figure
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
ax.set_aspect("equal")

typhoon = "durian2006"

minx, miny, maxx, maxy = df_phil.total_bounds

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

df_phil_damage.plot(ax=ax, color="whitesmoke", zorder=1)

df_phil_damage.plot(
    ax=ax,
    column=typhoon,
    cmap="Reds",
    vmin=0,
    vmax=1,
    zorder=2
    # legend=True
    # cax=cax,
)

df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
    ax=ax, zorder=3, linestyle=":", linewidth=2, color="black"
)

fig.savefig("tracks_damage.png")

plt.show()

print("Done")
# endregion

"""
Creating damages figures: Creating single figures in loop and saving
Tracks and Percentage Damage
"""
# region
#%% Loop through all typhoons
for typhoon in typhoons:

    print(typhoon)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
    ax.set_aspect("equal")

    minx, miny, maxx, maxy = df_phil.total_bounds

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    df_phil_damage.plot(ax=ax, color="whitesmoke", zorder=1)

    df_phil_damage.plot(
        ax=ax,
        column=typhoon,
        cmap="Reds",
        vmin=0,
        vmax=1,
        zorder=2
        # legend=True
        # cax=cax,
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=ax, zorder=4, linestyle=":", linewidth=2, color="black"
    )

    file_name = (
        "IBF_typhoon_model\\data\\figures\\track_images_damage\\"
        + typhoon
        + "_track_damage.png"
    )
    fig.savefig(file_name)

    plt.close()


print("Done")
# endregion

"""
Creating damages figures: Creating single figures in loop and saving
Track and Precentage Damage and Buffer
"""
# region

#%% Adding a track buffer
buffer = 300000
df_tracks_buff = df_tracks.copy()
df_tracks_buff = df_tracks_buff.to_crs("EPSG:25395")

df_tracks_buff["geometry"] = df_tracks_buff.buffer(buffer)
df_tracks_buff = df_tracks_buff.to_crs("EPSG:4326")

#%% Loop through all typhoons
for typhoon in typhoons:

    print(typhoon)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
    ax.set_aspect("equal")

    minx, miny, maxx, maxy = df_phil.total_bounds

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    df_phil_damage.plot(ax=ax, color="whitesmoke", zorder=1)

    df_phil_damage.plot(
        ax=ax,
        column=typhoon,
        cmap="Reds",
        vmin=0,
        vmax=1,
        zorder=2
        # legend=True
        # cax=cax,
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=ax, zorder=4, linestyle=":", linewidth=2, color="black"
    )

    df_tracks_buff[df_tracks_buff["SID"] == sid_dict[typhoon]].plot(
        ax=ax, alpha=0.01, zorder=3, color="blue"
    )

    file_name = (
        "IBF_typhoon_model\\data\\figures\\track_images_damage_buffer\\"
        + typhoon
        + "_track_damage.png"
    )
    fig.savefig(file_name)

    plt.close()


print("Done")
# endregion


"""
Creating damages figures: Multiple images in one figure
"""
# region
#%% Looping through typhoons for creating plots
# print(typhoons)
# typhoons = ["atsani2020", "tembin2017"]

nrows = 19
ncols = 3
figs, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(60, 120), facecolor="white")

minx, miny, maxx, maxy = df_phil.total_bounds

for typhoon in typhoons:

    print(typhoon)

    idx = typhoons.index(typhoon)
    idx_row = int(np.floor(idx / ncols))
    idx_col = int(idx - idx_row * ncols)

    axs[idx_row, idx_col].set_aspect("equal")

    axs[idx_row, idx_col].set_xlim(minx, maxx)
    axs[idx_row, idx_col].set_ylim(miny, maxy)
    axs[idx_row, idx_col].set_title(typhoon)

    axs[idx_row, idx_col].axis("off")

    df_phil_damage.plot(ax=axs[idx_row, idx_col], color="whitesmoke", zorder=1)

    df_phil_damage.plot(
        ax=axs[idx_row, idx_col], column=typhoon, cmap="Reds", vmin=0, vmax=1, zorder=2
    )

    df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
        ax=axs[idx_row, idx_col], linestyle=":", linewidth=2, color="black", zorder=3
    )

# plt.show()

figs.savefig("tracks_damage.png")
# endregion

