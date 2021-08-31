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
import dash_functions as dash_func


#%%Setting up dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#%%Obtaining dataframe for actual damage map
# data_path = os.path.join(os.getcwd(), 'app\data\data_output.xlsx')
# df = pd.read_excel(data_path)

df = pd.read_excel("C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\data\\data_output.xlsx")

#%%Converting the damage data into correct format
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

#%%Loading in t he files for the chloropleth map
j_files = []
for typhoon in typhoons:
    path = "C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\dashboards\\reduced_size_jsons\\" + str(typhoon) + ".json"
    f = open(path,)
    j_file = json.load(f)
    j_files.append(j_file)

map_dict = dict(zip(typhoons, j_files))

#%%Loading in the datasheet with the predicted probabilities
df_proba = pd.read_excel("C:\\Users\\Marieke\\GitHub\\Rice_Field_Damage_Philippines\\app\\outputs\\svm\\predicted_probs.xlsx")

features_used = ['dist_track', 'vmax_sust', 'mean_ruggedness', 'ruggedness_stdev']

#%% SVR Prediction data  to initialize
df_predictions_svr, test_scores_svr = dash_func.svr_dash(df, typhoons, municipality_codes, features_used, gamma=0.1, C=1, epsilon=0.001)

#%% Creating data table
typhoons_table = typhoons.copy()
typhoons_table.append('average')
print(typhoons_table)
table_dict = {'typhoons': typhoons_table, 'test_scores_svr' : test_scores_svr}
test_scores_df = pd.DataFrame(table_dict)


#%% RWSVR prediction data to initialize
# df_predictions_rwsvr, test_scores_rwsvr = dash_func.rwsvr_dash(df, df_proba, typhoons, municipality_codes, features_used, C=1, gamma=0.1, epsilon=0.001)


#%%For the dropdown menu
drop_options = [{'label':typh, 'value':typh} for typh in typhoons]

#%%Chloropleth map of actual damage
fig1 = px.choropleth_mapbox(new_df, geojson=map_dict['aere2011'], color="aere2011",
                           locations="municipality_code", featureidkey="properties.ADM3_PCODE", range_color=[0,1],
                           center={"lat": 13.420989, "lon": 124},
                           mapbox_style="carto-positron", zoom=6, width=900, height=700)

#%%Chloropleth map of predicted damage SVR
fig3 = px.choropleth_mapbox(df_predictions_svr, geojson=map_dict['aere2011'],
    color="aere2011", locations='mun_codes', featureidkey='properties.ADM3_PCODE', range_color=[0,1], center={"lat": 13.420989, "lon": 124}, mapbox_style="carto-positron", zoom=6, width=900, height=700)

#%%Chloropleth map of predicted damage RWSVR
# fig4 = px.choropleth_mapbox(df_predictions_rwsvr, geojson=map_dict['aere2011'],
#     color="aere2011", locations='mun_codes', featureidkey='properties.ADM3_PCODE', range_color=[0,1], center={"lat": 13.420989, "lon": 124}, mapbox_style="carto-positron", zoom=6, width=900, height=700)





app.layout = html.Div([
    html.Div([
        html.Div([
            html.H3('Percentage Damage'),
            dcc.Graph(id='graph_output', figure=fig1)
        ], className="six columns"),

        html.Div([
            html.H3('Predicted SVR damage'),
            dcc.Graph(id='predicted_damage_svr', figure=fig3)
        ], className="six columns"),

        

        # html.Div([
        #     html.H3('Predicted RWSVR damage'),
        #     dcc.Graph(id='predicted_damage_rwsvr', figure=fig4)
        # ], className="four columns"),
    ], className="row"),

    
    html.Div([

        html.Div([
            html.H3('SVR test scores'),
            dash_table.DataTable(
                id='table',
                columns=[{"name":i, 'id':i} for i in test_scores_df.columns],
                data=test_scores_df.to_dict('records'))
        ], className="six columns"),

        html.Div([
            html.H3('Selected Typhoon'),
            dcc.Dropdown(id='dropdown', multi=False, options=drop_options, value='aere2011')
            ], className="four columns"),

        html.Div([
            html.H3('press to update'),
            html.Button('Submit', id='button')
        ], className='two columns'),  

        html.Div([
            html.H3('C value'),
            dcc.Input(id='C', placeholder='Enter a value...', type='text', value='1')
            ], className="two columns"),

        html.Div([
            html.H3('gamma value'),
            dcc.Input(id='gamma', placeholder='Enter a value...', type='text', value='0.1')
            ], className="two columns"),

        html.Div([
            html.H3('epsilon value'),
            dcc.Input(id='epsilon', placeholder='Enter a value...', type='text', value='0.001')
            ], className="two columns"),

              
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

# @app.callback(
#     dash.dependencies.Output(component_id='predicted_damage_svr', component_property='figure'),
#     dash.dependencies.Input('button', 'n_clicks'),
#     dash.dependencies.Input(component_id='dropdown', component_property='value'),
#     state=[
#         State(component_id='C', component_property='value'),
#         State(component_id='gamma', component_property='value'),
#         State(component_id='epsilon', component_property='value')])
# def update_graph(n_clicks, typh_val, C_chosen, gamma_chosen, epsilon_chosen):
#     ctx = dash.callback_context
#     global df_predictions_svr
#     global test_scores_svr
#     if ctx.triggered[0]['prop_id'] == 'dropdown.value':
#         fig = px.choropleth_mapbox(
#             df_predictions_svr, geojson=map_dict[typh_val],
#             color=typh_val, locations='mun_codes', featureidkey='properties.ADM3_PCODE', range_color=[0,1], center={"lat": 13.420989, "lon": 124}, mapbox_style="carto-positron", zoom=6, width=900, height=700)
#     else:
#         C = float(C_chosen)
#         gamma = float(gamma_chosen)
#         epsilon = float(epsilon_chosen)    
#         df_pred, test_scores = dash_func.svr_dash(df, typhoons, municipality_codes, features_used, gamma=gamma, C=C, epsilon=epsilon)
#         test_scores_svr = test_scores
#         df_predictions_svr = df_pred    
#         fig = px.choropleth_mapbox(
#             df_predictions_svr, geojson=map_dict[typh_val],  color=typh_val, locations="mun_codes", range_color=[0,1], featureidkey="properties.ADM3_PCODE", center={"lat": 13.420989, "lon": 123.413674},mapbox_style="carto-positron", zoom=6, width=900, height=700)
    
#     return fig


# @app.callback(
#     dash.dependencies.Output(component_id='table', component_property='data'),
#     dash.dependencies.Input('button', 'n_clicks'),
#     state=[
#         State(component_id='C', component_property='value'),
#         State(component_id='gamma', component_property='value'),
#         State(component_id='epsilon', component_property='value')])
# def update_table(n_clicks, C_chosen, gamma_chosen, epsilon_chosen):
#     C = float(C_chosen)
#     gamma = float(gamma_chosen)
#     epsilon = float(epsilon_chosen)
#     df_pred, test_scores_svr = dash_func.svr_dash(df, typhoons, municipality_codes, features_used, gamma=gamma, C=C, epsilon=epsilon)    
#     table_dict = {'typhoons': typhoons_table, 'test_scores_svr' : test_scores_svr}
#     print(test_scores_svr)
#     test_scores_df = pd.DataFrame(table_dict)
#     return test_scores_df.to_dict('records')


# @app.callback(
#     dash.dependencies.Output(component_id='predicted_damage_rwsvr', component_property='figure'),
#     dash.dependencies.Input('button', 'n_clicks'),
#     dash.dependencies.Input(component_id='dropdown', component_property='value'),
#     state=[
#         State(component_id='C', component_property='value'),
#         State(component_id='gamma', component_property='value'),
#         State(component_id='epsilon', component_property='value')])
# def update_graph(n_clicks, typh_val, C_chosen, gamma_chosen, epsilon_chosen):
#     ctx = dash.callback_context
#     print(ctx.triggered)
#     global df_predictions_rwsvr
#     global test_scores_rwsvr
#     if ctx.triggered[0]['prop_id'] == 'dropdown.value':
#         fig = px.choropleth_mapbox(
#             df_predictions_rwsvr, geojson=map_dict[typh_val],
#             color=typh_val, locations='mun_codes', featureidkey='properties.ADM3_PCODE', range_color=[0,1], center={"lat": 13.420989, "lon": 124}, mapbox_style="carto-positron", zoom=6, width=900, height=700)
#     else:
#         C = float(C_chosen)
#         gamma = float(gamma_chosen)
#         epsilon = float(epsilon_chosen)    
#         df_pred, test_scores = dash_func.rwsvr_dash(df, df_proba, typhoons, municipality_codes, features_used, municipality_codes, gamma=gamma, C=C, epsilon=epsilon)
#         test_scores_rwsvr = test_scores
#         df_predictions_rwsvr = df_pred    
#         fig = px.choropleth_mapbox(
#             df_predictions_rwsvr, geojson=map_dict[typh_val],  color=typh_val, locations="mun_codes", range_color=[0,1], featureidkey="properties.ADM3_PCODE", center={"lat": 13.420989, "lon": 123.413674},mapbox_style="carto-positron", zoom=6, width=900, height=700)
    
#     return fig

# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})





app.run_server(debug=True, use_reloader=False)