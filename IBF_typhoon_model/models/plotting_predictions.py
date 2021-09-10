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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import geopandas as gpd

#%% Importing data
os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

# Combined input data
file_name = (
    "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data_02.xlsx"
)
path = os.path.join(cdir, file_name)
df = pd.read_excel(path, engine="openpyxl")

# Typhoon information
file_name = "IBF_typhoon_model\\data\\restricted_data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
df_typhoons = pd.read_excel(path, sheet_name="typhoon_overview", engine="openpyxl")

# Geographical information
file_name = "IBF_typhoon_model\\data\\restricted_data\\data_collection.xlsx"
path = os.path.join(cdir, file_name)
df_geo = pd.read_excel(path, sheet_name="municipality_geo_data", engine="openpyxl")

# Admin boundaries
file_name = (
    "IBF_typhoon_model\\data\\phl_administrative_boundaries\\phl_admbnda_adm3.shp"
)
path = os.path.join(cdir, file_name)
df_phil = gpd.read_file(path)

# typhoon tracks shapefile
file_name = "IBF_typhoon_model\\data\\gis_data\\typhoon_tracks\\tracks_filtered.shp"
path = os.path.join(cdir, file_name)
df_tracks = gpd.read_file(path)

# Dictionary of typhoons and SID
sid_dict = dict(zip(df_typhoons["name_year"], df_typhoons["storm_id"]))
typhoon_dict = dict(zip(df_typhoons["storm_id"], df_typhoons["name_year"]))


"""
Binary Classification
"""
#%% Train optimal model for making predictions
# Setting the general input variables
threshold = 0.3
df["class_value_binary"] = [
    1 if df["perc_loss"][i] > threshold else 0 for i in range(len(df))
]

features = [
    "mean_elevation_m",
    "ruggedness_stdev",
    "slope_stdev",
    "poverty_perc",
    "perimeter",
    "glat",
    "glon",
    "rainfall_sum",
    "rainfall_max",
    "dis_track_min",
    "vmax_sust",
]

# Setting for feature selection on full data set
typhoon = "haima2016"
# df_train = df[df['typhoon']!=typhoon]
df_train = df.copy()

X = df_train[features]
y = df_train["class_value_binary"]
y = y.astype(int)


rf = RandomForestClassifier(
    class_weight="balanced",
    max_depth=None,
    min_samples_leaf=3,
    min_samples_split=10,
    n_estimators=100,
)

rf_fitted = rf.fit(X, y)

#%% Make predictions for typhoon
X_predict = df[["class_value_binary", "mun_code"]][df["typhoon"] != typhoon]

df_input = df_phil.copy()

# Geo data
df_input = pd.merge(
    df_input, df_geo, how="left", left_on=["ADM3_PCODE"], right_on=["mun_code"]
)

# Rainfall data
file_name = (
    "IBF_typhoon_model\\data\\rainfall_data\\output_data\\"
    + typhoon
    + "\\"
    + typhoon
    + "_matrix.csv"
)
path = os.path.join(cdir, file_name)
df_rain = pd.read_csv(path)
df_input = pd.merge(
    df_input, df_rain, how="left", left_on=["ADM3_PCODE"], right_on=["mun_code"]
)

# Wind data
file_name = (
    "IBF_typhoon_model\\data\\wind_data\\output\\" + typhoon + "_windgrid_output.csv"
)
path = os.path.join(cdir, file_name)
df_wind = pd.read_csv(path)
df_input = pd.merge(
    df_input, df_wind, how="left", left_on=["ADM3_PCODE"], right_on=["adm3_pcode"]
)

# %% Obtain predictions
X_predict = df_input[
    [
        "mean_elevation_m",
        "ruggedness_stdev",
        "slope_stdev",
        "poverty_perc",
        "perimeter",
        "glat",
        "glon",
        "rainfall_sum",
        "rainfall_max",
        "dis_track_min",
        "vmax_sust",
        "ADM3_PCODE",
    ]
]

X_predict = X_predict.dropna()
y_predict = rf_fitted.predict(X_predict.drop("ADM3_PCODE", axis=1))
X_predict["predicted"] = y_predict


# %% Merge with geo df
df_plot = pd.merge(
    X_predict,
    df_phil[["ADM3_PCODE", "geometry"]],
    how="left",
    left_on=["ADM3_PCODE"],
    right_on=["ADM3_PCODE"],
)

#%% To GeoPandas
df_plot = gpd.GeoDataFrame(df_plot)

# %%
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
ax.set_aspect("equal")

minx, miny, maxx, maxy = df_phil.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

df_phil.plot(ax=ax, color="whitesmoke", zorder=1, edgecolor="black", linewidth=0.3)

df_plot.plot(
    ax=ax,
    column="predicted",
    cmap="Reds",
    vmin=0,
    vmax=1,
    zorder=2,
    edgecolor="black",
    linewidth=0.3,
)

df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
    ax=ax, zorder=3, linestyle=":", linewidth=2, color="black"
)

ax.axis("off")

file_name = "IBF_typhoon_model\\models\\output\\binary_map.png"
path = os.path.join(cdir, file_name)
fig.savefig(path)
print("Done")


# %%

"""
Multiclass Classification
"""
#%% Train optimal model for making predictions
# Setting the general input variables
def determine_class(x, classes):

    for key, value in classes.items():

        if value[0] <= x < value[1]:

            class_value = key
            break

    return class_value


threshold = 0.3
classes = {"0": [0, 0.3], "1": [0.3, 0.5], "2": [0.5, 1.1]}
df["class_value_multi"] = df["perc_loss"].apply(
    lambda x: determine_class(x, classes=classes)
)


features = [
    "mean_elevation_m",
    "ruggedness_stdev",
    "slope_stdev",
    "poverty_perc",
    "perimeter",
    "glat",
    "glon",
    "rainfall_sum",
    "rainfall_max",
    "dis_track_min",
    "vmax_sust",
]

# Setting for feature selection on full data set
typhoon = "haima2016"
df_train = df[df["typhoon"] != typhoon]
# df_train = df.copy()

X = df_train[features]
y = df_train["class_value_multi"]
y = y.astype(int)


rf = RandomForestClassifier(
    class_weight="balanced",
    max_depth=None,
    min_samples_leaf=3,
    min_samples_split=10,
    n_estimators=100,
)

rf_fitted = rf.fit(X, y)

#%% Make predictions for typhoon
X_predict = df[["class_value_multi", "mun_code"]][df["typhoon"] != typhoon]

df_input = df_phil.copy()

# Geo data
df_input = pd.merge(
    df_input, df_geo, how="left", left_on=["ADM3_PCODE"], right_on=["mun_code"]
)

# Rainfall data
file_name = (
    "IBF_typhoon_model\\data\\rainfall_data\\output_data\\"
    + typhoon
    + "\\"
    + typhoon
    + "_matrix.csv"
)
path = os.path.join(cdir, file_name)
df_rain = pd.read_csv(path)
df_input = pd.merge(
    df_input, df_rain, how="left", left_on=["ADM3_PCODE"], right_on=["mun_code"]
)

# Wind data
file_name = (
    "IBF_typhoon_model\\data\\wind_data\\output\\" + typhoon + "_windgrid_output.csv"
)
path = os.path.join(cdir, file_name)
df_wind = pd.read_csv(path)
df_input = pd.merge(
    df_input, df_wind, how="left", left_on=["ADM3_PCODE"], right_on=["adm3_pcode"]
)

# %% Obtain predictions
X_predict = df_input[
    [
        "mean_elevation_m",
        "ruggedness_stdev",
        "slope_stdev",
        "poverty_perc",
        "perimeter",
        "glat",
        "glon",
        "rainfall_sum",
        "rainfall_max",
        "dis_track_min",
        "vmax_sust",
        "ADM3_PCODE",
    ]
]

X_predict = X_predict.dropna()
y_predict = rf_fitted.predict(X_predict.drop("ADM3_PCODE", axis=1))
X_predict["predicted"] = y_predict


# %% Merge with geo df
df_plot = pd.merge(
    X_predict,
    df_phil[["ADM3_PCODE", "geometry"]],
    how="left",
    left_on=["ADM3_PCODE"],
    right_on=["ADM3_PCODE"],
)

#%% To GeoPandas
df_plot = gpd.GeoDataFrame(df_plot)

# %%
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
ax.set_aspect("equal")

minx, miny, maxx, maxy = df_phil.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

df_phil.plot(ax=ax, color="whitesmoke", zorder=1, edgecolor="black", linewidth=0.3)

df_plot.plot(
    ax=ax,
    column="predicted",
    cmap="Reds",
    vmin=0,
    vmax=2,
    zorder=2,
    edgecolor="black",
    linewidth=0.3,
)

df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
    ax=ax, zorder=3, linestyle=":", linewidth=2, color="black"
)

ax.axis("off")

file_name = "IBF_typhoon_model\\models\\output\\multiclass_map.png"
path = os.path.join(cdir, file_name)
fig.savefig(path)
print("Done")


# %%
"""
Predictions Map Regression
"""


features = [
    "mean_elevation_m",
    "ruggedness_stdev",
    "slope_stdev",
    "poverty_perc",
    "perimeter",
    "glat",
    "glon",
    "rainfall_sum",
    "rainfall_max",
    "dis_track_min",
    "vmax_sust",
]

# Setting for feature selection on full data set
typhoon = "haima2016"
df_train = df[df["typhoon"] != typhoon]
# df_train = df.copy()

X = df_train[features]
y = df_train["perc_loss"]
y = y.astype(int)


rf = RandomForestRegressor(
    max_depth=None, min_samples_leaf=3, min_samples_split=8, n_estimators=100,
)

rf_fitted = rf.fit(X, y)

#%% Make predictions for typhoon
X_predict = df[["perc_loss", "mun_code"]][df["typhoon"] != typhoon]

df_input = df_phil.copy()

# Geo data
df_input = pd.merge(
    df_input, df_geo, how="left", left_on=["ADM3_PCODE"], right_on=["mun_code"]
)

# Rainfall data
file_name = (
    "IBF_typhoon_model\\data\\rainfall_data\\output_data\\"
    + typhoon
    + "\\"
    + typhoon
    + "_matrix.csv"
)
path = os.path.join(cdir, file_name)
df_rain = pd.read_csv(path)
df_input = pd.merge(
    df_input, df_rain, how="left", left_on=["ADM3_PCODE"], right_on=["mun_code"]
)

# Wind data
file_name = (
    "IBF_typhoon_model\\data\\wind_data\\output\\" + typhoon + "_windgrid_output.csv"
)
path = os.path.join(cdir, file_name)
df_wind = pd.read_csv(path)
df_input = pd.merge(
    df_input, df_wind, how="left", left_on=["ADM3_PCODE"], right_on=["adm3_pcode"]
)

# %% Obtain predictions
X_predict = df_input[
    [
        "mean_elevation_m",
        "ruggedness_stdev",
        "slope_stdev",
        "poverty_perc",
        "perimeter",
        "glat",
        "glon",
        "rainfall_sum",
        "rainfall_max",
        "dis_track_min",
        "vmax_sust",
        "ADM3_PCODE",
    ]
]

X_predict = X_predict.dropna()
y_predict = rf_fitted.predict(X_predict.drop("ADM3_PCODE", axis=1))
X_predict["predicted"] = y_predict


# %% Merge with geo df
df_plot = pd.merge(
    X_predict,
    df_phil[["ADM3_PCODE", "geometry"]],
    how="left",
    left_on=["ADM3_PCODE"],
    right_on=["ADM3_PCODE"],
)

#%% To GeoPandas
df_plot = gpd.GeoDataFrame(df_plot)

# %%
fig, ax = plt.subplots(figsize=(10, 10), facecolor="white", tight_layout=True)
ax.set_aspect("equal")

minx, miny, maxx, maxy = df_phil.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

df_phil.plot(ax=ax, color="whitesmoke", zorder=1, edgecolor="black", linewidth=0.3)

df_plot.plot(
    ax=ax,
    column="predicted",
    cmap="Reds",
    vmin=0,
    vmax=1,
    zorder=2,
    edgecolor="black",
    linewidth=0.3,
)

df_tracks[df_tracks["SID"] == sid_dict[typhoon]].plot(
    ax=ax, zorder=3, linestyle=":", linewidth=2, color="black"
)

ax.axis("off")

file_name = "IBF_typhoon_model\\models\\output\\regression_map.png"
path = os.path.join(cdir, file_name)
fig.savefig(path)
print("Done")
# %%
