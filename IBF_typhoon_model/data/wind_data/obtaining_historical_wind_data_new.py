#%%
import time
import ftplib
import os
import sys
from datetime import datetime, timedelta
from sys import platform
import subprocess
import logging
import traceback
from pathlib import Path
from azure.storage.file import FileService
from azure.storage.file import ContentSettings

import pandas as pd
from pybufrkit.decoder import Decoder
import numpy as np
from geopandas.tools import sjoin
import geopandas as gpd
import click

os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()
decoder = Decoder()

# Importing local libraries
sys.path.insert(0, os.path.join(cdir, "\\IBF_typhoon_model\\data\\wind_data\\climada"))

from climada.hazard import Centroids, TropCyclone, TCTracks
from climada.hazard.tc_tracks import estimate_roci, estimate_rmw
from climada.hazard.tc_tracks_forecast import TCForecast


# from climada.hazard import Centroids, TropCyclone,TCTracks
# from climada.hazard.tc_tracks_forecast import TCForecast
# from typhoonmodel.utility_fun import track_data_clean, Check_for_active_typhoon, Sendemail, \
#     ucl_data, plot_intensity, initialize

# if platform == "linux" or platform == "linux2": #check if running on linux or windows os
#     from typhoonmodel.utility_fun import Rainfall_data
# elif platform == "win32":
#     from typhoonmodel.utility_fun import Rainfall_data_window as Rainfall_data

decoder = Decoder()

file_name = "IBF_typhoon_model\\data\\wind_data\\input\\typhoon_events.csv"
path = os.path.join(cdir, file_name)
typhoon_events = pd.read_csv(path)

typoon_event = []
for index, row in typhoon_events.iterrows():
    typoon_event.append(str(row["International_Name"]).upper() + str(row["year"]))
typoon_event = []
for index, row in typhoon_events.iterrows():
    typoon_event.append(str(row["International_Name"]).upper())


sel_ibtracs = TCTracks()
# years 1993 and 1994 in basin EP.
# correct_pres ignores tracks with not enough data. For statistics (frequency of events), these should be considered as well
sel_ibtracs.read_ibtracs_netcdf(
    provider="usa", year_range=(2006, 2021), basin="WP", correct_pres=False
)

data_sel = [tr for tr in sel_ibtracs.data if (tr.name + tr.sid[:4]) in typoon_event]

#####Check uint for wind speed
print(data_sel[0].max_sustained_wind_unit)  # it is in knot

# path="C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\Documents\\Typhoon-Impact-based-forecasting-model\\IBF-Typhoon-model\\"

file_name = "IBF_typhoon_model\\data\\wind_data\\historical_events"
Output_folder = os.path.join(cdir, file_name)

# Output_folder="C:\\Users\\ATeklesadik\\OneDrive - Rode Kruis\\Documents\\Typhoon-Impact-based-forecasting-model\\analysis\\historical_events\\"
##Create grid points to calculate Winfield
cent = Centroids()
cent.set_raster_from_pnt_bounds((118, 6, 127, 19), res=0.05)
# this option is added to make the script scaleable globally To Do
# cent.set_raster_from_pnt_bounds((LonMin,LatMin,LonMax,LatMax), res=0.05)
cent.check()
cent.plot()
####


# TODO correct this
admin = gpd.read_file(os.path.join(path, "./data-raw/phl_admin3_simpl2.geojson"))

df = pd.DataFrame(data=cent.coord)
df["centroid_id"] = "id" + (df.index).astype(str)
centroid_idx = df["centroid_id"].values
ncents = cent.size
df = df.rename(columns={0: "lat", 1: "lon"})
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
# df.to_crs({'init': 'epsg:4326'})
df.crs = {"init": "epsg:4326"}
df_admin = sjoin(df, admin, how="left").dropna()

# Sometimes the ECMWF ftp server complains about too many requests
# This code allows several retries with some sleep time in between

list_intensity = []
distan_track = []
for tr in data_sel:
    track = TCTracks()
    typhoon = TropCyclone()
    track.data = [tr]
    # track.equal_timestep(3)
    tr = track.data[0]
    typhoon.set_from_tracks(track, cent, store_windfields=True)
    # Make intensity plot using the high resolution member
    plot_intensity.plot_inensity(
        typhoon=typhoon,
        event=tr.sid,
        output_dir=Output_folder,
        date_dir=date_dir,
        typhoon_name=tr.name,
    )
    windfield = typhoon.windfields
    nsteps = windfield[0].shape[0]
    centroid_id = np.tile(centroid_idx, nsteps)
    intensity_3d = windfield[0].toarray().reshape(nsteps, ncents, 2)
    intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()
    timesteps = np.repeat(track.data[0].time.values, ncents)
    # timesteps = np.repeat(tr.time.values, ncents)
    timesteps = timesteps.reshape((nsteps, ncents)).ravel()
    inten_tr = pd.DataFrame(
        {"centroid_id": centroid_id, "value": intensity, "timestamp": timesteps,}
    )
    inten_tr = inten_tr[inten_tr.value > threshold]
    inten_tr["storm_id"] = tr.sid
    inten_tr["name"] = tr.name
    list_intensity.append(inten_tr)
    distan_track1 = []
    for index, row in df.iterrows():
        dist = np.min(
            np.sqrt(
                np.square(tr.lat.values - row["lat"])
                + np.square(tr.lon.values - row["lon"])
            )
        )
        distan_track1.append(dist * 111)
    dist_tr = pd.DataFrame({"centroid_id": centroid_idx, "value": distan_track1})
    dist_tr["storm_id"] = tr.sid
    dist_tr["name"] = tr.name
    distan_track.append(dist_tr)

df_intensity = pd.concat(list_intensity)
df_intensity = pd.merge(df_intensity, df_admin, how="outer", on="centroid_id")
df_intensity = df_intensity.dropna()

df_intensity_ = df_intensity.groupby(["adm3_pcode", "storm_id"], as_index=False).agg(
    {"value": ["count", "max"]}
)
# rename columns
df_intensity_.columns = [x for x in ["adm3_pcode", "storm_id", "value_count", "v_max"]]
distan_track1 = pd.concat(distan_track)
distan_track1 = pd.merge(distan_track1, df_admin, how="outer", on="centroid_id")
distan_track1 = distan_track1.dropna()

distan_track1 = distan_track1.groupby(
    ["adm3_pcode", "name", "storm_id"], as_index=False
).agg({"value": "min"})
distan_track1.columns = [
    x for x in ["adm3_pcode", "name", "storm_id", "dis_track_min"]
]  # join_left_df_.columns.ravel()]
typhhon_df = pd.merge(
    df_intensity_, distan_track1, how="left", on=["adm3_pcode", "storm_id"]
)

typhhon_df.to_csv(
    os.path.join(Output_folder, "windfield.csv"), index=False
)  ##### uint for windspeed is kn and it is 1minute average
