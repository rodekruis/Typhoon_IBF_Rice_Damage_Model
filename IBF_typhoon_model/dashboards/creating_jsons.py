#%% 
from dash_core_components.Dropdown import Dropdown
import geopandas as gpd
import os
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from shapely.geometry import geo

#%%
typhoons = ['aere2011', 
    'hagupit2014', 
    'haikui2017', 
    'haiyan2013', 
    'jangmi2014', 
    'kai-tak2017', 
    'kalmaegi2014', 
    'kammuri2019', 
    'meari2011', 
    'mekkhala2015', 
    'melor2015', 
    'mirinae2009', 
    'nesat2011', 
    'nock-ten2011', 
    'nock-ten2016', 
    'parma2009', 
    'utor2013', 
    'washi2011']

for typhoon in typhoons:
    print(typhoon)
    path = "C:\\Users\\Marieke\\Desktop\\QGIS\\" + str(typhoon) + ".shp"
    geodf = gpd.read_file(path)

    GeoJSON_path = "C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\dashboards\\shapefiles\\geojson"

    geodf.to_file(GeoJSON_path, driver = "GeoJSON")
    with open(GeoJSON_path) as geofile:
        j_file = json.load(geofile)

    name = "shapefiles\\" + str(typhoon) + ".json"

    with open(name, 'w') as json_file:
        json.dump(j_file, json_file)



# %%
