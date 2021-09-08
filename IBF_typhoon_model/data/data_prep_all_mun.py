#%% Importing libraries
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

#%% Rice damage is loaded per region
file_name = "IBF_typhoon_model\\data\\restricted_data\\rice_data\\rice_losses\\rice_losses_combined.xlsx"
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

# Dictionary where each entry is a damage dataframe
df_damages = {}
for region in regions:
    df_damages[region] = pd.read_excel(
        path, sheet_name=region, header=[0, 1], engine="openpyxl"
    )

#%% Laoding the rice area planted sheet
file_name = "IBF_typhoon_model\\data\\restricted_data\\rice_data\\rice_area\\rice_area_planted.xlsx"
path = os.path.join(cdir, file_name)
rice_area_planted = pd.read_excel(path, engine="openpyxl")

#%% Loading the typhoon overview sheet
file_name = "IBF_typhoon_model\\data\\restricted_data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
typh_overview = pd.read_excel(path, sheet_name="typhoon_overview", engine="openpyxl")

#%% Loading the geographical features
file_name = "IBF_typhoon_model\\data\\restricted_data\\data_collection.xlsx"
path = os.path.join(cdir, file_name)
mun_geo_data = pd.read_excel(
    path, sheet_name="municipality_geo_data", engine="openpyxl"
)

#%% Geographical overview
file_name = "IBF_typhoon_model\\data\\restricted_data\\data_collection.xlsx"
path = os.path.join(cdir, file_name)
mun_overview = pd.read_excel(path, sheet_name="admin_boundaries", engine="openpyxl")

#%% Region overview
file_name = "IBF_typhoon_model\\data\\restricted_data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
region_overview = pd.read_excel(
    path, sheet_name="rice_loss_collection", engine="openpyxl"
)

#%% Process the loss data
# TODO Do this in pre-processing


def setting_nan(x):
    totally_damaged = x["totally_damaged"]
    partially_damaged = x["partially_damaged"]
    area_affected = x["area_affected"]
    if pd.isnull(totally_damaged) & pd.isnull(partially_damaged):
        area_affected = np.nan
    return area_affected


regions_nan = regions.copy()
regions_nan.remove("region_5")
regions_nan.remove("region_10")
df_damages["region_5"] = df_damages["region_5"].replace("no obs", np.nan)

for region in regions_nan:
    typhoons_region = df_damages[region].columns.levels[0].tolist()
    typhoons_region.remove("info")
    for typh in typhoons_region:
        df_damages[region][typh, "area_affected"] = df_damages[region][typh].apply(
            setting_nan, axis="columns"
        )

#%% Remove all rows where municipality code is NaN
for region in regions:
    df_damages[region] = df_damages[region][
        df_damages[region]["info"]["mun_code"].notnull()
    ]

#%% Create dataframe in input format with loss data added

# create dictionary of region name and code
region_dict = dict(zip(region_overview["reg_number"], region_overview["reg_code"]))

df_total = pd.DataFrame(columns=["mun_code", "typhoon", "area_affected",])

for region in regions:

    df_temp = df_damages[region]

    typhoons = df_temp.columns.levels[0].tolist()
    typhoons.remove("info")

    municipalities = mun_overview["mun_code"][
        mun_overview["reg_code"] == region_dict[region]
    ].tolist()

    # municipalities = df_temp["info", "mun_code"].tolist()

    N_typh = len(typhoons)
    N_mun = len(municipalities)

    municipality_codes_full = np.repeat(municipalities, N_typh)
    typhoons_full = typhoons * N_mun

    data_temp = {"mun_code": municipality_codes_full, "typhoon": typhoons_full}
    df_temp_total = pd.DataFrame(data_temp)

    loop_info = df_temp_total[["mun_code", "typhoon"]].values

    for mun, typh in loop_info:

        # find index of typhoon&municipality in df_total
        df_temp_total_index = (df_temp_total["mun_code"] == mun) & (
            df_temp_total["typhoon"] == typh
        )

        # find index of municipality in rice loss dataframe
        # mun_index = df_temp.index[df_temp["info", "mun_code"] == mun].values[0]
        try:
            mun_index = df_temp.index[df_temp["info", "mun_code"] == mun].values[0]
            df_temp_total.loc[df_temp_total_index, "area_affected"] = df_temp[
                typh, "area_affected"
            ][mun_index]
        except:
            df_temp_total.loc[df_temp_total_index, "area_affected"] = np.nan

        # # fill in damages in df_total
        # df_temp_total.loc[df_temp_total_index, "area_affected"] = df_temp[
        #     typh, "area_affected"
        # ][mun_index]

    df_total = pd.concat([df_total, df_temp_total])

# TODO Do this in data pre-processing
df_total["area_affected"] = df_total["area_affected"].replace("-", np.nan)

#%% Add SID and obtain info
sid_dict = dict(zip(typh_overview["name_year"], typh_overview["storm_id"]))
df_total["storm_id"] = df_total["typhoon"].map(sid_dict)
municipalities = df_total["mun_code"].unique()
typhoons = df_total["typhoon"].unique()

# Add year as a column
# Create id and year dictionary
id_year_dict = dict(zip(typh_overview["storm_id"], typh_overview["year"]))
df_total["year"] = df_total["storm_id"].map(id_year_dict)

#%% Only include observations for which a value within the province or region is observed
area = "reg_code"  # 'prov_code'

# Add region code to df
mun_reg_dict = dict(zip(mun_overview["mun_code"], mun_overview["reg_code"]))
df_total["reg_code"] = df_total["mun_code"].map(mun_reg_dict)

# Add province code to df
mun_prov_dict = dict(zip(mun_overview["mun_code"], mun_overview["prov_code"]))
df_total["prov_code"] = df_total["mun_code"].map(mun_prov_dict)

# DF with observations that are not null
df_non_nan = df_total[["typhoon", area]][df_total["area_affected"].notnull()]

# Combinations of province and typhoons for which damage is observed at least once
observation_comb = (df_non_nan["typhoon"] + df_non_nan[area]).unique()

# Keep only entries that occur in this list
df_total["keep"] = [
    "yes" if x in observation_comb else "no"
    for x in (df_total["typhoon"] + df_total[area])
]
df_total = df_total[df_total["keep"] == "yes"]
df_total = df_total.drop(["keep"], axis=1)

#%% Add standing rice area
# Obtain the dates on which rice is planted
planting_dates = rice_area_planted.columns
planting_dates = planting_dates[planting_dates != "ADM3_PCODE"]

# Check if planting dates are all unique
if len(planting_dates.unique()) != len(planting_dates):
    print(
        "One or multiple dates occur double in the dataset \n Follow-up to sum the planted areas on these date"
    )

# Obtain earliest and latest possible dates for obtaining standing area
min_planting_date = min(planting_dates)
max_planting_date = max(planting_dates)
min_area_date = min_planting_date + dt.timedelta(days=115)
max_area_date = max_planting_date

# Check which reference date to use for obtaining rice area<br>
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
typh_date_dict = dict(zip(typh_overview["storm_id"], typh_overview["start_date"]))

# Calculate the standing area for a given storm ID and municpality code
def standing_area(x):
    storm_id = x["storm_id"]
    mun_code = x["mun_code"]

    # Find the start and end date for planted area to sum
    typh_date = typh_date_dict[storm_id]
    end_date = standing_area_date(typh_date, min_area_date, max_area_date)
    start_date = end_date - dt.timedelta(days=115)

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

#%% Add percentage loss
def division(x, y):
    try:
        value = x / y
    except:
        value = np.nan
    return value


df_total["perc_loss"] = df_total.apply(
    lambda x: division(x["area_affected"], x["rice_area"]), axis=1
).values

# Remove observations with rice area below 30 ha because these can be inaccurate
df_total = df_total[
    ~((df_total["rice_area"] <= 30) & (df_total["perc_loss"].notnull()))
]

# Replace with 1 when damage > 1
full_loss_count = len(df_total[df_total["perc_loss"] > 1])
print(
    f"The number of observations for which the percentage loss is above 100% is {full_loss_count}"
)

df_total["perc_loss"] = [1 if x > 1 else x for x in df_total["perc_loss"]]

plt.hist(df_total["perc_loss"], bins=100)
plt.show()

#%% Add Geographic data
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

#%% Add rainfall data
df_total["rainfall_sum"] = ""
df_total["rainfall_max"] = ""

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

#%% Add windspeed and track distance
# Adding the wind data
df_total["vmax_sust"] = ""
df_total["dis_track_min"] = ""

for typhoon in typhoons:

    # Path to the rainfall excel sheet per typhoon
    wind_path = os.path.join(
        cdir,
        "IBF_typhoon_model\\data\\wind_data\\output",
        typhoon + "_windgrid_output.csv",
    )
    df_temp = pd.read_csv(wind_path)

    for mun in municipalities:

        try:
            vmax_sust = df_temp.loc[df_temp["adm3_pcode"] == mun, "vmax_sust"].values[0]
            dis_track_min = df_temp.loc[
                df_temp["adm3_pcode"] == mun, "dis_track_min"
            ].values[0]
        except:
            # print(mun, typhoon)
            vmax_sust = np.nan
            dis_track_min = np.nan

        df_total.loc[
            (df_total["typhoon"] == typhoon) & (df_total["mun_code"] == mun),
            "vmax_sust",
        ] = vmax_sust

        df_total.loc[
            (df_total["typhoon"] == typhoon) & (df_total["mun_code"] == mun),
            "dis_track_min",
        ] = dis_track_min

df_total["v_max"] = df_total["vmax_sust"] / 2.35

#%% Save to excel
df_total.to_excel(
    "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data_04.xlsx",
    index=False,
)
# %%
