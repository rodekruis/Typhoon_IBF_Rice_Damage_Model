#%%
"""
Loading the Libraries
"""
from dash_core_components.Dropdown import Dropdown
import geopandas as gpd
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
"""
Loading the data and settings
"""

#Settings
thresh = 0.3

#Loading the df with the actual values
cwd = os.getcwd()
file_name = 'app\\data\\combined_input_data\\input_data_01.xlsx'
path = os.path.join(cwd, file_name)
df_source = pd.read_excel(path)
df_source['class_value'] = [1 if df_source['perc_loss'][i] > thresh else 0 for i in range(len(df_source))]

#Loading in the predicted values
file_name = 'app\\outputs\\classifications\\rf_pred_02.xlsx'
path = os.path.join(cwd, file_name)
df_predicted = pd.read_excel(path)

#Setting up dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Loading in t he files for the chloropleth map
path = os.path.dirname(__file__) + os.path.sep + 'jsonfile_reduced.json'
f = open(path,)
j_file = json.load(f)

#Loading in the files for the track map
track_path = os.path.join(os.getcwd(), "app\data\QGIS\Typhoon_tracks\IBTrACS.WP.list.v04r00.lines\\typhoon_tracks.shp") 
geo_df = gpd.read_file(track_path)

#Converting the track dataframe to the correct format
geo_df['ISO_TIME'] = pd.to_datetime(geo_df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S')
geo_df = geo_df.sort_values(by=['ISO_TIME'])
geo_df['year'] = ""
for i in range(len(geo_df)):
    geo_df.loc[i, 'year'] = str(geo_df['ISO_TIME'][i].year)
geo_df['name_year'] = geo_df['NAME'] + geo_df['year']
geo_df.replace('JANGMI2015', 'JANGMI2014')

#Setting municipality codes & typhoons
municpality_codes = df_source['municipality_codes'].unique()
typhoons = df_source['typhoons'].unique().tolist()

#Creating dataframe with actual damages (in correct format)
dict_df = {'municipality_code':municpality_codes}
df_actual = pd.DataFrame(dict_df)

for typh in typhoons:
    df_actual[typh] = ''

for i in range(len(df_actual)):
    for typh in typhoons:
        value = df_source['class_value'][(df_source['municipality_codes']==df_actual['municipality_code'][i]) & (df_source['typhoons']==typh)].values[0]
        df_actual[typh][i] = value

for typh in typhoons:
    df_actual[[typh]] = df_actual[[typh]].apply(pd.to_numeric)

add = [0] * 19
df_actual.loc[len(df_actual)] = add

#Loading in t he files for the chloropleth map
j_files = []
for typhoon in typhoons:
    path = "C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\dashboards\\reduced_size_jsons\\" + str(typhoon) + ".json"
    f = open(path,)
    j_file = json.load(f)
    j_files.append(j_file)

map_dict = dict(zip(typhoons, j_files))

#%%
"""
Creating the settings for the maps
"""

#For the dropdown menu
drop_options = [{'label':typh, 'value':typh} for typh in typhoons]

#Chloropleth map of actual damage
fig1 = px.choropleth_mapbox(df_actual, geojson=map_dict['aere2011'], color="aere2011",
                           locations="municipality_code", featureidkey="properties.ADM3_PCODE",
                           center={"lat": 13.420989, "lon": 124},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700)

#%%Chloropleth map of predicted damage
fig3 = px.choropleth_mapbox(df_predicted, geojson=j_file, color="aere2011",
                            locations='mun_codes', featureidkey='properties.ADM3_PCODE', center={"lat": 13.420989, "lon": 124},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700, color_continuous_scale='amp', range_color=[0,1])


#%%Figure with Track
geo_df_new = geo_df[geo_df['name_year']=='AERE2011']
lat = geo_df_new['LAT'].reset_index(drop=True)
lon = geo_df_new['LON'].reset_index(drop=True)
fig2 = px.line_mapbox(lat=lat, lon=lon, hover_name=lat,
                     mapbox_style="carto-positron", center={"lat": 13.420989, "lon": 123.413674},width=900, height=700, zoom=6)

#%%Putting it together
app.layout = html.Div([
    html.Div([

        html.Div([
            html.H3('Percentage Damage'),
            dcc.Graph(id='graph_output', figure=fig1)
        ], className="six columns"),

        html.Div([
            html.H3('Typhoon Track'),
            dcc.Graph(id='predicted_damage', figure=fig3)
        ], className="six columns"),

    ], className="row"),
    
    html.Div([
            html.H3('Selected Typhoon'),
            dcc.Dropdown(id='dropdown', multi=False, options=drop_options, value='aere2011')
            ], className="row")
])

#%%Callback for chloropleth map of actual values
@app.callback(
    dash.dependencies.Output(component_id='graph_output', component_property='figure'),
    [dash.dependencies.Input(component_id='dropdown', component_property='value')])
def update_graph(val_chosen):
    fig = px.choropleth_mapbox(df_actual, geojson=map_dict[val_chosen], color=val_chosen,
                           locations="municipality_code", featureidkey="properties.ADM3_PCODE",
                           center={"lat": 13.420989, "lon": 123.413674},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700)
    return fig

#%%Callback for chloropleth map of predicted values
@app.callback(
    dash.dependencies.Output(component_id='predicted_damage', component_property='figure'),
    [dash.dependencies.Input(component_id='dropdown', component_property='value')])
def update_graph(val_chosen):
    fig = px.choropleth_mapbox(df_predicted, geojson=j_file, color=val_chosen,
                           locations="mun_codes", featureidkey="properties.ADM3_PCODE",
                           center={"lat": 13.420989, "lon": 123.413674},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700, color_continuous_scale='amp', range_color=[0,1])
    

    return fig

# #%%Callback for tracks
# @app.callback(
#     dash.dependencies.Output(component_id='tracks', component_property='figure'),
#     [dash.dependencies.Input(component_id='dropdown', component_property='value')])
# def update_graph(val_chosen):
#     val_chosen_new = val_chosen.upper()
#     geo_df_new = geo_df[geo_df['name_year']==val_chosen_new]
#     lat = geo_df_new['LAT']
#     lon = geo_df_new['LON']
#     fig2 = px.line_mapbox(lat=lat, lon=lon, hover_name=lat,
#         mapbox_style="carto-positron", center={"lat": 13.420989, "lon": 123.413674}, zoom=6, width=900, height=700) 
#     return fig2

app.run_server(debug=True, use_reloader=False)

