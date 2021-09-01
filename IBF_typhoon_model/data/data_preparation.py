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
Loading the data
"""
#%% Rice losses sheet: loaded per region
file_name = "IBF_typhoon_model\\data\\rice_data\\rice_losses\\rice_losses_combined.xlsx"
path = os.path.join(cdir, file_name)
regions = [
    "region_car",
    "region_1",
    "region_2",
    "region_3",
    "region_4a",
    "region_4b",
    "region_5",
    "region_6",
    "region_7",
    "region_8",
    "region_9",
    "region_10",
    "region_11",
    "region_12",
]

# Dictionary where each entry is the damage dataframe
df_damages = {}
for region in regions:
    df_damages[region] = pd.read_excel(
        path, sheet_name=region, header=[0, 1], engine="openpyxl"
    )

# Example df
print(df_damages["region_car"].head())

#%% Rice area planted sheet
file_name = "IBF_typhoon_model\\data\\rice_data\\rice_area\\rice_area_planted.xlsx"
path = os.path.join(cdir, file_name)
rice_area_planted = pd.read_excel(path, engine="openpyxl")

#%% Data overview sheet
file_name = "IBF_typhoon_model\\data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
typh_overview = pd.read_excel(path, sheet_name="typhoon_overview", engine="openpyxl")

#%% Geographical Features
file_name = "IBF_typhoon_model\\data\\data_collection.xlsx"
path = os.path.join(cdir, file_name)
mun_geo_data = pd.read_excel(
    path, sheet_name="municipality_geo_data", engine="openpyxl"
)

#%% Geographical overview
file_name = "IBF_typhoon_model\\data\\data_collection.xlsx"
path = os.path.join(cdir, file_name)
mun_overview = pd.read_excel(path, sheet_name="admin_boundaries", engine="openpyxl")

#%% Wind data
file_name = "IBF_typhoon_model\\data\\wind_data\\output\\historical_typhoons_wind.csv"
path = os.path.join(cdir, file_name)
wind_data = pd.read_csv(path)

"""
Process the loss data
"""
#%% Setting the zero and NaN values
# For the regions with totally and partially collected
# --> area afected is missing when both are missing
# TODO do this formatting in the excel sheet and remove it from script, for consistent formatting
# TODO gives a standard format in which new damage data should be uploaded to re-train
def setting_nan(x):

    totally_damaged = x["totally_damaged"]
    partially_damaged = x["partially_damaged"]
    area_affected = x["area_affected"]

    if (
        pd.isnull(totally_damaged)
        & pd.isnull(partially_damaged)
        # & pd.isnull(area_affected)
    ):
        area_affected = np.nan

    return area_affected


regions_nan = regions.copy()

# For region 5 and region 10 region there is only area_affected
regions_nan.remove("region_5")
regions_nan.remove("region_10")
df_damages["region_5"] = df_damages["region_5"].replace("no obs", np.nan)

# For all regions with totally and partially damaged
for region in regions_nan:

    typhoons_region = df_damages[region].columns.levels[0].tolist()
    typhoons_region.remove("info")

    for typh in typhoons_region:
        df_damages[region][typh, "area_affected"] = df_damages[region][typh].apply(
            setting_nan, axis="columns"
        )

#%% Remove all the rows where the municpality is NaN
for region in regions:

    df_damages[region] = df_damages[region][
        df_damages[region]["info"]["mun_code"].notnull()
    ]


"""
Create dataframe in input format: with loss added
"""
#%% Create dataframe with municipalities, typhoons and damages
# For each region: takes all the municipalities which occur in the excel sheet
# and combines it with every typhoon that occurs in the excel sheet
# TODO still need to apply rule on which observations are included in the data set

df_total = pd.DataFrame(columns=["mun_code", "typhoon", "area_affected",])

for region in regions:

    df_temp = df_damages[region]
    typhoons = df_temp.columns.levels[0].tolist()
    typhoons.remove("info")
    municipalities = df_temp["info", "mun_code"].tolist()

    N_typh = len(typhoons)
    N_mun = len(municipalities)

    municipality_codes_full = np.repeat(municipalities, N_typh)
    typhoons_full = typhoons * N_mun

    data_temp = {"mun_code": municipality_codes_full, "typhoon": typhoons_full}

    df_temp_total = pd.DataFrame(data_temp)
    loop_info = df_temp_total[["mun_code", "typhoon"]].values

    for mun, typh in loop_info:

        # find index of municipality in rice loss dataframe
        mun_index = df_temp.index[df_temp["info", "mun_code"] == mun].values[0]

        # find index of typhoon&municipality in df_total
        df_temp_total_index = (df_temp_total["mun_code"] == mun) & (
            df_temp_total["typhoon"] == typh
        )

        # fill in damages in df_total
        df_temp_total.loc[df_temp_total_index, "area_affected"] = df_temp[
            typh, "area_affected"
        ][mun_index]

    df_total = pd.concat([df_total, df_temp_total])

# Dataframe still contains '-'
df_total["area_affected"] = df_total["area_affected"].replace("-", np.nan)

#%% Add SID
sid_dict = dict(zip(typh_overview["name_year"], typh_overview["storm_id"]))
df_total["storm_id"] = df_total["typhoon"].map(sid_dict)

#%% Obtain info
municipalities = df_total["mun_code"].unique()
typhoons = df_total["typhoon"].unique()

"""
Add Standing rice area
"""
#%% Process area planted into standing area
planting_dates = rice_area_planted.columns
planting_dates = planting_dates[planting_dates != "ADM3_PCODE"]

# Check if planting dates are all unique
if len(planting_dates.unique()) != len(planting_dates):
    print(
        "One or multiple dates occur double in the dataset \n Follow-up to sum the planted areas on these date"
    )

# From detection to harvesting = 95 days (according to PRISM information)
# Obtain earliest and latest possible dates for obtaining standing area
min_planting_date = min(planting_dates)
max_planting_date = max(planting_dates)
min_area_date = min_planting_date + dt.timedelta(days=115)
max_area_date = max_planting_date

# Check which reference date to use for obtaining rice area
# If outside of available dates: use same day on closests year available
def standing_area_date(x, min_area_date, max_area_date):

    if x < min_area_date:
        day = x.day
        month = x.month
        year = min_area_date.year
        date = dt.datetime(year, month, day)
        if date < min_area_date:
            date = dt.datetime(year + 1, month, day)
    elif x > max_area_date:
        day = x.day
        month = x.month
        year = max_area_date.year
        date = dt.datetime(year, month, day)
        if date > max_area_date:
            date = dt.datetime(year - 1, month, day)
    else:
        date = x

    return date


# Create dictionary of storm_id and date (start_date)
# start_date used as reference: can be changed to end_date or landfall data?
typh_date_dict = dict(zip(typh_overview["storm_id"], typh_overview["start_date"]))

# Calculate the standing area for a given storm ID and municpality code
def standing_area(x):

    storm_id = x["storm_id"]
    mun_code = x["mun_code"]

    # Find the start and end date for planted area to sum
    typh_date = typh_date_dict[storm_id]
    end_date = standing_area_date(typh_date, min_area_date, max_area_date)
    start_date = end_date - dt.timedelta(days=95)

    # Obtain all the dates that should be summed & sum
    available_dates = [
        date for date in planting_dates if (date >= start_date) & (date <= end_date)
    ]
    area_sum = (
        rice_area_planted[rice_area_planted["ADM3_PCODE"] == mun_code][available_dates]
        .sum(axis=1)
        .values[0]
    )

    return area_sum


df_total["rice_area"] = df_total[["storm_id", "mun_code"]].apply(
    standing_area, axis="columns"
)

# Convert to HA (pixel size 20 x 20 in meters)
# TODO check if this holds for all maps
df_total["rice_area"] = df_total["rice_area"] * 0.04

"""
Add percentage loss
"""
#%% Use area affected and the standing rice area to obtain the percentage affected


def division(x, y):
    try:
        value = x / y
    except:
        value = np.nan
    return value


# Do this for the sem1 and sem2 based rice area and the SoS based rice area
df_total["perc_loss"] = df_total.apply(
    lambda x: division(x["area_affected"], x["rice_area"]), axis=1
).values

# Replace with 1 when damage > 1
full_loss_count = len(df_total[df_total["perc_loss"] > 1])
print(
    f"The number of observations for which the percentage loss is above 100% is {full_loss_count}"
)
# df_total.loc[(df_total["perc_loss"] > 1), "perc_loss"] = 1


"""
Add geographic data
"""
#%% List of geographical variables to add
# Same variables as in housing model
# coast - perimeter ratio has been added
df_total["mean_slope"] = ""
df_total["mean_elevation_m"] = ""
df_total["ruggedness_stdev"] = ""
df_total["mean_ruggedness"] = ""
df_total["slope_stdev"] = ""
df_total["area_km2"] = ""
df_total["poverty_perc"] = ""
df_total["with_coast"] = ""
df_total["coast_length"] = ""
df_total["perimeter"] = ""
df_total["glat"] = ""
df_total["glon"] = ""

for i in municipalities:

    index = mun_geo_data.index[mun_geo_data["mun_code"] == i].values[0]
    index_total = df_total.index[df_total["mun_code"] == i].tolist()

    df_total.loc[index_total, "mean_slope"] = mun_geo_data["mean_slope"][index]
    df_total.loc[index_total, "mean_elevation_m"] = mun_geo_data["mean_elevation_m"][
        index
    ]
    df_total.loc[index_total, "ruggedness_stdev"] = mun_geo_data["ruggedness_stdev"][
        index
    ]
    df_total.loc[index_total, "mean_ruggedness"] = mun_geo_data["mean_ruggedness"][
        index
    ]
    df_total.loc[index_total, "slope_stdev"] = mun_geo_data["slope_stdev"][index]
    df_total.loc[index_total, "area_km2"] = mun_geo_data["area_km2"][index]
    df_total.loc[index_total, "poverty_perc"] = mun_geo_data["poverty_perc"][index]
    df_total.loc[index_total, "with_coast"] = mun_geo_data["with_coast"][index]
    df_total.loc[index_total, "coast_length"] = mun_geo_data["coast_length"][index]
    df_total.loc[index_total, "perimeter"] = mun_geo_data["perimeter"][index]
    df_total.loc[index_total, "glat"] = mun_geo_data["glat"][index]
    df_total.loc[index_total, "glon"] = mun_geo_data["glon"][index]

df_total["coast_peri_ratio"] = df_total["coast_length"] / df_total["perimeter"]


"""
Add Rainfall Data
"""

#%% Loop through the rainfall folders to obtain sheets
df_total["rainfall_sum"] = ""
df_total["rainfall_max"] = ""

# C:\Users\Marieke\GitHub\Typhoon_IBF_Rice_Damage_Model\IBF_typhoon_model\data\rainfall_data\output_data\danas2019\danas2019_matrix.csv

for typhoon in typhoons:

    # Path to the rainfall excel sheet per typhoon
    rain_path = os.path.join(
        cdir,
        "IBF_typhoon_model\\data\\rainfall_data\\output_data",
        typhoon,
        typhoon + "_matrix.csv",
    )
    df_temp = pd.read_csv(rain_path)

    for mun in municipalities:

        rainfall_sum = df_temp.loc[df_temp["mun_code"] == mun, "rainfall_sum"].values[0]
        rainfall_max = df_temp.loc[df_temp["mun_code"] == mun, "rainfall_max"].values[0]

        df_total.loc[
            (df_total["typhoon"] == typhoon) & (df_total["mun_code"] == mun),
            "rainfall_sum",
        ] = rainfall_sum
        df_total.loc[
            (df_total["typhoon"] == typhoon) & (df_total["mun_code"] == mun),
            "rainfall_max",
        ] = rainfall_max


"""
Add Windspeed and Track Distance Data
"""

#%% Merging the dataframes

df_total = pd.merge(
    df_total,
    wind_data[
        [
            "adm3_pcode",
            "storm_id",
            "v_max",
            "dis_track_min",
            "vmax_gust",
            "vmax_gust_mph",
            "vmax_sust",
            "vmax_sust_mph",
        ]
    ],
    left_on=["mun_code", "storm_id"],
    right_on=["adm3_pcode", "storm_id"],
    how="left",
    validate="one_to_one",
)


# %% Save to excel file
df_total.to_excel(
    "IBF_typhoon_model\\data\\combined_input_data\\input_data.xlsx", index=False
)

#%%
