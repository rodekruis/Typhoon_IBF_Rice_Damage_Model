#%% import libraries from local directory
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import feedparser
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import datetime
from datetime import timedelta
import smtplib
from smtplib import SMTP_SSL as SMTP
import geopandas as gpd
import fiona
from ftplib import FTP
import shutil
from os.path import relpath
import re
import zipfile
from os.path import relpath
from os import listdir
from os.path import isfile, join
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer
from sys import platform
from io import StringIO
from bs4 import BeautifulSoup
import subprocess
import netCDF4
import bottleneck
from geopandas.tools import sjoin
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import nearest_points

os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()
decoder = Decoder()

# Importing local libraries
sys.path.insert(0, os.path.join(cdir, "\\IBF_typhoon_model\\data\\wind_data\\climada"))

from climada.hazard import Centroids, TropCyclone, TCTracks
from climada.hazard.tc_tracks import estimate_roci, estimate_rmw
from climada.hazard.tc_tracks_forecast import TCForecast


#%% Loads an excel sheet with Local_name, International_Name and year
file_name = "IBF_typhoon_model\\data\\wind_data\\input\\typhoon_events.csv"
path = os.path.join(cdir, file_name)
typhoon_events = pd.read_csv(path)

df_typhoons = pd.read_csv(path)

typhoon_events = []
for index, row in df_typhoons.iterrows():
    typhoon_events.append(str(row["International_Name"]).upper() + str(row["year"]))


for typhoon_name in typhoon_events:

    typoon_event = [typhoon_name]
    print(typoon_event)

    sel_ibtracs = TCTracks()

    # correct_pres ignores tracks with not enough data. For statistics (frequency of events), these should be considered as well
    # Set year range for which data should be collected
    # TODO can automate this from input sheet
    sel_ibtracs.read_ibtracs_netcdf(
        provider="usa", year_range=(2006, 2021), basin="WP", correct_pres=False
    )
    Typhoons = TCTracks()
    # Select typhoons that are in the typhoon event sheet
    Typhoons.data = [
        tr for tr in sel_ibtracs.data if (tr.name + tr.sid[:4]) in typoon_event
    ]

    # In[4]:

    # Plots all tracks in time frame
    # Is supposed to plot the tracks, doesn't work --> not required
    ax = Typhoons.plot()
    ax.get_legend()._loc = 1  # correct legend location
    ax.set_title("2006-2021, WP", fontsize=14)  # set title
    plt.show()
    plt.close()

    #

    # In[5]:

    # Select names and storm id's of storms
    names = [[tr.name, tr.sid] for tr in Typhoons.data]

    # In[6]:

    file_name = "IBF_typhoon_model\\data\\wind_data\\input\\phl_admin3_simpl2.geojson"
    path = os.path.join(cdir, file_name)
    admin = gpd.read_file(path)

    minx, miny, maxx, maxy = admin.total_bounds

    print(minx, miny, maxx, maxy)

    # minx = 123
    # maxx = 125
    # miny = 7
    # maxy = 8

    cent = Centroids()
    # cent.set_raster_from_pnt_bounds((minx,miny,maxx,maxy), res=0.05)
    cent.set_raster_from_pnt_bounds((minx, miny, maxx, maxy), res=0.05)
    cent.check()
    # cent.plot()
    # plt.show()
    # plt.close()

    # TODO set the bounds to the bounds of the shape file of the Philippines
    # TODO this needs to be changed in the pipeline to obtain complete information
    # cent = Centroids()
    # cent.set_raster_from_pnt_bounds((118,6,127,19), res=0.05)
    # cent.check()
    # cent.plot()

    # In[7]:

    df = pd.DataFrame(data=cent.coord)
    df["centroid_id"] = "id" + (df.index).astype(str)
    centroid_idx = df["centroid_id"].values
    ncents = cent.size
    df = df.rename(columns={0: "lat", 1: "lon"})
    # calculate wind field for each ensamble members
    # Instead of one grid point for each municipalities: uses a range of gridpoints and calculates value for each point
    # Eventually select a specific point in municipality based on condition

    # In[8]:

    def adjust_tracks(forcast_df):
        track = xr.Dataset(
            data_vars={
                "max_sustained_wind": (
                    "time",
                    0.514444 * forcast_df.max_sustained_wind.values,
                ),
                "environmental_pressure": (
                    "time",
                    forcast_df.environmental_pressure.values,
                ),
                "central_pressure": ("time", forcast_df.central_pressure.values),
                "lat": ("time", forcast_df.lat.values),
                "lon": ("time", forcast_df.lon.values),
                "radius_max_wind": ("time", forcast_df.radius_max_wind.values),
                "radius_oci": ("time", forcast_df.radius_oci.values),
                "time_step": (
                    "time",
                    np.full_like(forcast_df.time_step.values, 3, dtype=float),
                ),
            },
            coords={"time": forcast_df.time.values,},
            attrs={
                "max_sustained_wind_unit": "m/s",
                "central_pressure_unit": "mb",
                "name": forcast_df.name,
                "sid": forcast_df.sid,  # +str(forcast_df.ensemble_number),
                "orig_event_flag": forcast_df.orig_event_flag,
                "data_provider": forcast_df.data_provider,
                "id_no": forcast_df.id_no,
                "basin": forcast_df.basin,
                "category": forcast_df.category,
            },
        )
        track = track.set_coords(["lat", "lon"])
        return track

    # In[9]:

    tracks = TCTracks()
    tracks.data = [adjust_tracks(tr) for tr in Typhoons.data]

    # In[10]:

    # Shows specific data for one of the typhoons
    tracks.data[0]

    # In[11]:

    # tracks1=TCTracks()
    # tracks1.data=tracks.data[0]
    # TYphoon = TropCyclone()
    # TYphoon.set_from_tracks(tracks1, cent, store_windfields=True)

    tracks.equal_timestep(0.5)

    # define a new typhoon class
    TYphoon = TropCyclone()
    TYphoon.set_from_tracks(tracks, cent, store_windfields=True)

    # In[12]:

    # plot intensity for one of the ensambles
    # TYphoon.plot_intensity(event=Typhoons.data[0].sid)
    # plt.show()
    # plt.close()

    # In[13]:

    df = pd.DataFrame(data=cent.coord)
    df["centroid_id"] = "id" + (df.index).astype(str)
    centroid_idx = df["centroid_id"].values
    ncents = cent.size
    df = df.rename(columns={0: "lat", 1: "lon"})
    # TODO what does this threshold hold do and why is it set to this value?
    # TODO set threshold value to 0 to delete all entirely irrelevant observation
    threshold = 10
    threshold = 0.1
    # calculate wind field for each ensamble members

    list_intensity = []
    distan_track = []

    for tr in tracks.data:
        print(tr.name)

        track = TCTracks()
        typhoon = TropCyclone()
        track.data = [tr]
        typhoon.set_from_tracks(track, cent, store_windfields=True)
        windfield = typhoon.windfields
        nsteps = windfield[0].shape[0]
        centroid_id = np.tile(centroid_idx, nsteps)
        intensity_3d = windfield[0].toarray().reshape(nsteps, ncents, 2)
        intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()

        timesteps = np.repeat(tr.time.values, ncents)
        timesteps = timesteps.reshape((nsteps, ncents)).ravel()
        inten_tr = pd.DataFrame(
            {"centroid_id": centroid_id, "value": intensity, "timestamp": timesteps,}
        )

        inten_tr = inten_tr[inten_tr.value > threshold]

        inten_tr["storm_id"] = tr.sid
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
        distan_track.append(dist_tr)

    # In[14]:

    df.shape

    # In[15]:

    # ? Changed document source, geojson instead of shape
    file_name = "IBF_typhoon_model\\data\\wind_data\\input\\phl_admin3_simpl2.geojson"
    path = os.path.join(cdir, file_name)
    admin = gpd.read_file(path)

    df_ = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    # df.to_crs({'init': 'epsg:4326'})
    df_.crs = {"init": "epsg:4326"}
    df_ = df_.to_crs("EPSG:4326")
    df_admin = sjoin(df_, admin, how="left")
    # TODO why is this added? --> commented out
    # To remove points that are in water
    df_admin = df_admin.dropna()

    # In[16]:

    # df_new = df_.to_crs("EPSG:4326")
    # df_new.crs

    # In[17]:

    # # Some municipalities are missing in the dataframe
    # # Obtain a dictionary with their codes and the code of the nearest municipalities for which data is available
    # mun_missing = admin[admin['adm3_pcode'].isin(df_admin['adm3_pcode']) == False]
    # mun_missing_points = gpd.GeoDataFrame(mun_missing, geometry=gpd.points_from_xy(mun_missing.glon, mun_missing.glat))
    # pts3 = df_admin.geometry.unary_union

    # def near(point, pts=pts3):
    #      # find the nearest point and return the corresponding Place value
    #      nearest = df_admin.geometry == nearest_points(point, pts)[1]
    #      # display(df_admin[nearest].centroid_id.values[0])
    #      return df_admin[nearest].adm3_pcode.values[0]

    # mun_missing_points['neighbor_mun_code'] = mun_missing_points.apply(lambda row: near(row.geometry), axis=1)

    # dict_nearest = dict(zip(mun_missing_points['adm3_pcode'], mun_missing_points['neighbor_mun_code']))

    # Some municipalities are missing in the dataframe
    # Obtain a dictionary with their codes and the code of the nearest municipalities for which data is available
    mun_missing = admin[admin["adm3_pcode"].isin(df_admin["adm3_pcode"]) == False]
    print(f"There are {len(mun_missing)} missing municipalities")

    mun_missing_points = gpd.GeoDataFrame(
        mun_missing, geometry=gpd.points_from_xy(mun_missing.glon, mun_missing.glat)
    )
    pts3 = df_admin.geometry.unary_union

    def near(point, pts=pts3):
        # find the nearest point and return the corresponding Place value
        nearest = df_admin.geometry == nearest_points(point, pts)[1]
        # display(df_admin[nearest].centroid_id.values[0])
        return df_admin[nearest].centroid_id.values[0]

    mun_missing_points["centroid_id"] = mun_missing_points.apply(
        lambda row: near(row.geometry), axis=1
    )

    # dict_nearest = dict(zip(mun_missing_points['adm3_pcode'], mun_missing_points['neighbor_mun_code']))

    # TODO change to centroid_id

    # In[18]:

    def match_values_lat(x):

        return df_admin["lat"][df_admin["centroid_id"] == x].values[0]

    def match_values_lon(x):

        return df_admin["lon"][df_admin["centroid_id"] == x].values[0]

    def match_values_geo(x):

        return df_admin["geometry"][df_admin["centroid_id"] == x].values[0]

    mun_missing_points["lat"] = mun_missing_points["centroid_id"].apply(
        match_values_lat
    )
    mun_missing_points["lon"] = mun_missing_points["centroid_id"].apply(
        match_values_lon
    )
    mun_missing_points["geometry"] = mun_missing_points["centroid_id"].apply(
        match_values_geo
    )

    # mun_missing_points['lat'], mun_missing_points['long'], mun_missing_points['geometry'] = mun_missing_points['centroid_id'].apply(match_values)

    df_admin = df_admin[
        [
            "adm3_en",
            "adm3_pcode",
            "adm2_pcode",
            "adm1_pcode",
            "glat",
            "glon",
            "lat",
            "lon",
            "centroid_id",
        ]
    ]

    mun_missing_points = mun_missing_points[
        [
            "adm3_en",
            "adm3_pcode",
            "adm2_pcode",
            "adm1_pcode",
            "glat",
            "glon",
            "lat",
            "lon",
            "centroid_id",
        ]
    ]

    df_admin = pd.concat([df_admin, mun_missing_points])

    mun_missing = admin[admin["adm3_pcode"].isin(df_admin["adm3_pcode"]) == False]
    print(f"There are {len(mun_missing)} missing municipalities")

    # In[19]:

    df_intensity = pd.concat(list_intensity)
    df_intensity = pd.merge(df_intensity, df_admin, how="outer", on="centroid_id")

    # In[20]:

    df_intensity.head()

    # In[21]:

    df_intensity.shape
    df_intensity[df_intensity["lat"].notnull()].head()

    # In[22]:

    df_intensity = pd.concat(list_intensity)
    df_intensity = pd.merge(df_intensity, df_admin, how="outer", on="centroid_id")
    # df_intensity = gpd.GeoDataFrame(df_intensity, geometry=gpd.points_from_xy(df_intensity.lon, df_intensity.lat))
    df_intensity = df_intensity.dropna()
    # df_intensity = gpd.GeoDataFrame(df_intensity, geometry=gpd.points_from_xy(df_intensity.lon, df_intensity.lat))
    # ? Only keeps observations with an intensity higher than 12 --> why --> why not cancel out at the end when preparing the data so it is easier to adjust
    # ? Obtains the maximum intensity for each municipality and storm_id combination & also return the count (how often the combination occurs in the set)
    # TODO changed to not filtering based on intensity
    # df_intensity=df_intensity[df_intensity['value'].gt(12)].groupby(['adm3_pcode','storm_id'],as_index=False).agg({"value":['count', 'max']})
    df_intensity = (
        df_intensity[df_intensity["value"].gt(0)]
        .groupby(["adm3_pcode", "storm_id"], as_index=False)
        .agg({"value": ["count", "max"]})
    )
    # rename columns
    df_intensity.columns = [
        x for x in ["adm3_pcode", "storm_id", "value_count", "v_max"]
    ]
    #########################################################################################
    df_track = pd.concat(distan_track)
    df_track = pd.merge(df_track, df_admin, how="outer", on="centroid_id")
    df_track = df_track.dropna()
    # ? Obtains the minimum track distance for each municipality and storm_id combination
    df_track_ = df_track.groupby(["adm3_pcode", "storm_id"], as_index=False).agg(
        {"value": "min"}
    )
    df_track_.columns = [
        x for x in ["adm3_pcode", "storm_id", "dis_track_min"]
    ]  # join_left_df_.columns.ravel()]
    typhhon_df = pd.merge(
        df_intensity, df_track_, how="left", on=["adm3_pcode", "storm_id"]
    )
    # typhhon_df.to_csv(os.path.join('/dbfs/mnt/TyphoonData/typhoon/Bronze/TyphoonModel/Typhoon-Impact-based-forecasting-model/past_typhoon_windfields/historical_typhoons_intensity.csv')

    # In[ ]:

    # In[23]:

    # Check if there are duplicates for municipality and storm_id
    duplicate = typhhon_df[
        typhhon_df.duplicated(subset=["adm3_pcode", "storm_id"], keep=False)
    ]
    duplicate.head()

    # No duplicates: final dataframe contains vmax and dis_track_min fpr each storm and municipality

    # In[24]:

    typhhon_df.head()
    np.min(typhhon_df["v_max"])
    np.max(typhhon_df["dis_track_min"])

    # In[25]:

    # Add the changes that are related to the wind variables and present in the R script
    typhhon_df["vmax_gust"] = (
        typhhon_df["v_max"] * 1.21 * 1.9 * 1.94384
    )  # knot(1.94384) and 1.21 is conversion factor for 10 min average to 1min average
    typhhon_df["vmax_gust_mph"] = (
        typhhon_df["v_max"] * 1.21 * 1.9 * 2.23694
    )  # mph 1.9 is factor to drive gust and sustained wind
    typhhon_df["vmax_sust_mph"] = typhhon_df["v_max"] * 1.21 * 2.23694
    typhhon_df["vmax_sust"] = typhhon_df["v_max"] * 1.21 * 1.94384

    # In[26]:

    file_name = (
        "IBF_typhoon_model\\data\\wind_data\\output\\"
        + typhoon_name.lower()
        + "_windgrid_output.csv"
    )
    path = os.path.join(cdir, file_name)
    typhhon_df.to_csv(path)


# %%
