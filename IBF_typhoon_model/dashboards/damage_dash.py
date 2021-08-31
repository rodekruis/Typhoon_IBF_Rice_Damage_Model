#%%Libraries
# from app.dashboards.dash_functions import make_dash_table
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pactools.grid_search import GridSearchCVProgressBar
from sklearn.model_selection import RandomizedSearchCV
import os
import matplotlib.pyplot as plt
import random
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
from dash.dependencies import State, Output
import dash_table
# import dash_functions as dash_func

#%%
"""
Importing the data
"""

#DF with the actual values
cwd = os.getcwd()
file_name = 'app\data\combined_input_data\input_data_sos.xlsx'
file_name = '..\\data\combined_input_data\input_data_sos.xlsx'
path = os.path.join(cwd, file_name)
df_actual = pd.read_excel(path)


#%%Setting up dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#%%Obtaining dataframe for actual damage map
# data_path = os.path.join(os.getcwd(), 'app\data\data_output.xlsx')
# df = pd.read_excel(data_path)

df = pd.read_excel("C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\data\\combined_input_data\\input_data.xlsx")

# df_1 = pd.read_excel("C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\data\\data_output.xlsx")

#%%Converting the damage data into correct format SOS
municipality_codes = df['municipality_codes'].unique()
typhoons = df['typhoons'].unique().tolist()
dict_df = {'municipality_code':municipality_codes}
new_df = pd.DataFrame(dict_df)

for typh in typhoons:
    new_df[typh] = ''

#Loss data
for i in range(len(new_df)):
    for typh in typhoons:
        value = df['perc_loss'][(df['municipality_codes']==new_df['municipality_code'][i]) & (df['typhoons']==typh)].values[0]
        new_df[typh][i] = value

for typh in typhoons:
    new_df[[typh]] = new_df[[typh]].apply(pd.to_numeric)

add = [0] * 19
new_df.loc[len(new_df)] = add

#%%Converting the damage data into correct format SEM BASED
municipality_codes = df_1['municipality_codes'].unique()
typhoons = df_1['typhoons'].unique().tolist()
dict_df_1 = {'municipality_code':municipality_codes}
new_df_1 = pd.DataFrame(dict_df_1)

for typh in typhoons:
    new_df_1[typh] = ''

#Loss data
for i in range(len(new_df_1)):
    for typh in typhoons:
        value = df_1['perc_loss'][(df_1['municipality_codes']==new_df_1['municipality_code'][i]) & (df_1['typhoons']==typh)].values[0]
        new_df_1[typh][i] = value

for typh in typhoons:
    new_df_1[[typh]] = new_df_1[[typh]].apply(pd.to_numeric)

add = [0] * 19
new_df_1.loc[len(new_df_1)] = add

#%%Loading in t he files for the chloropleth map
j_files = []
for typhoon in typhoons:
    path = "C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\dashboards\\reduced_size_jsons\\" + str(typhoon) + ".json"
    f = open(path,)
    j_file = json.load(f)
    j_files.append(j_file)

map_dict = dict(zip(typhoons, j_files))

#%%For the dropdown menu
drop_options = [{'label':typh, 'value':typh} for typh in typhoons]

#%%Chloropleth map of actual damage SOS
fig1 = px.choropleth_mapbox(new_df, geojson=map_dict['aere2011'], color="aere2011",
                           locations="municipality_code", featureidkey="properties.ADM3_PCODE", range_color=[0,1],
                           center={"lat": 13.420989, "lon": 124},
                           mapbox_style="carto-positron", zoom=6.5, width=900, height=700)

#SEM
fig2 = px.choropleth_mapbox(new_df_1, geojson=map_dict['aere2011'], color="aere2011",
                           locations="municipality_code", featureidkey="properties.ADM3_PCODE", range_color=[0,1],
                           center={"lat": 13.420989, "lon": 124},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700)


app.layout = html.Div([
    
    html.Div([
        html.Div([
            html.H3('Percentage Damage SOS based'),
            dcc.Graph(id='graph_output', figure=fig1)
        ], className="six columns"),

        html.Div([
            html.H3('Percentage Damage Semester based'),
            dcc.Graph(id='graph_output_sem', figure=fig2)
        ], className="six columns"),

    ], className="row"),

    
    html.Div([

        html.Div([
            html.H3('Selected Typhoon'),
            dcc.Dropdown(id='dropdown', multi=False, options=drop_options, value='aere2011')
            ], className="nine columns"),
             
    ], className="row"),

])


#%%Callback for chloropleth map of actual values
@app.callback(
    dash.dependencies.Output(component_id='graph_output', component_property='figure'),
    [dash.dependencies.Input(component_id='dropdown', component_property='value')])
def update_graph(val_chosen):
    ctx = dash.callback_context
    print(ctx.triggered)
    fig = px.choropleth_mapbox(new_df, geojson=map_dict[val_chosen],    color=val_chosen, locations="municipality_code", featureidkey="properties.ADM3_PCODE", range_color=[0,1], center={"lat": 13.420989, "lon": 123.413674},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700)
    return fig

#SEM
@app.callback(
    dash.dependencies.Output(component_id='graph_output_sem', component_property='figure'),
    [dash.dependencies.Input(component_id='dropdown', component_property='value')])
def update_graph(val_chosen):
    ctx = dash.callback_context
    print(ctx.triggered)
    fig = px.choropleth_mapbox(new_df_1, geojson=map_dict[val_chosen],    color=val_chosen, locations="municipality_code", featureidkey="properties.ADM3_PCODE", range_color=[0,1], center={"lat": 13.420989, "lon": 123.413674},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700)
    return fig



app.run_server(debug=True, use_reloader=False)