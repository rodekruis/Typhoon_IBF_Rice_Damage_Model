#%% Libraries
from qgis.core import QgsProject
from qgis.core import QgsProcessing
import processing
import sys
from qgis.core import *
from qgis.analysis import QgsNativeAlgorithms
import processing
from processing.core.Processing import Processing

import os
from os import listdir
from os.path import isfile, join
import pandas as pd


"""
For a single TIF File
"""
# region
#%% Loading data
cdir = os.getcwd()

# Vector layer
file_name = "data\\PHL_administrative_boundaries\\phl_admbnda_adm3_psa_namria_20200529_fixed.shp"
shp_path = os.path.join(cdir, file_name)
shp = QgsVectorLayer(shp_path, "zonepolygons", "ogr")

# Raster layer
file_name = "data\\QGIS\\PRISM\\PRISM_new\\2015S2_PHL_S1ATSX_MSCAASD_FO_MSOS.tif"
raster_path = os.path.join(cdir, file_name)
raster = QgsRasterLayer(raster_path)

#%% Obtaining zonal histogram
Processing.initialize()
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

input_raster = raster
input_vector = shp

params = {
    "COLUMN_PREFIX": "HISTO_",
    "INPUT_RASTER": input_raster,
    "INPUT_VECTOR": input_vector,
    "OUTPUT": "TEMPORARY_OUTPUT",
    "RASTER_BAND": 1,
}

result = processing.run("native:zonalhistogram", params)
layer = result["OUTPUT"]
path = "C:/Users/Marieke/GitHub/Rice_Field_Damage_Philippines/app/export_file.xlsx"

QgsVectorFileWriter.writeAsVectorFormat(layer, path, "utf-8", layer.crs(), "xlsx")
legend = raster.legendSymbologyItems()

#%% Turning attribute table into dataframe
columns = [f.name() for f in layer.fields()]
columns_types = [f.typeName() for f in layer.fields()]

row_list = []
for f in layer.getFeatures():
    row_list.append(dict(zip(columns, f.attributes())))

df_histo = pd.DataFrame(row_list, columns=columns)

#%% convert legend to dictionary & rename columns
key = []
value = []

for i in range(2, len(legend)):

    list_str = legend[i][0].split()
    key.append("HISTO_" + list_str[0])
    value.append(list_str[2])

value_date = pd.to_datetime(pd.Series(value), format="%d-%b-%Y")
legend_dict = dict(zip(key, value_date))

df_histo = df_histo.rename(columns=legend_dict)

# filter_col = [col for col in df_histo.columns if col.startswith('HISTO_')]

# df_histo_new.columns = pd.to_datetime(df_histo_new.columns)

# endregion

"""
For multiple TIF Files
"""

#%% Loading data
cdir = os.getcwd()

# Directory of folder with all PRISM files to process
folder_path = "data\\QGIS\\PRISM\\PRISM_new"
files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

# Keeping only the Tif files (diregarding sml and hdr) --> make sure only .sml .hdr and tif
file_list = [file for file in files if file.endswith((".sml", ".hdr")) == False]

# Vector layer
file_name = "data\\PHL_administrative_boundaries\\phl_admbnda_adm3_psa_namria_20200529_fixed.shp"
shp_path = os.path.join(cdir, file_name)
shp = QgsVectorLayer(shp_path, "zonepolygons", "ogr")

#%%For testing
# file_name = "data\\PHL_administrative_boundaries\\phl_admbnda_adm3_Bicol.shp"
# shp_path = os.path.join(cdir, file_name)
# shp = QgsVectorLayer(shp_path, "zonepolygons", "ogr")

# file_list = file_list[1]
# file = file_list

#%% Create df to save info
columns = [f.name() for f in shp.fields()]
columns_types = [f.typeName() for f in shp.fields()]

row_list = []
for f in shp.getFeatures():
    row_list.append(dict(zip(columns, f.attributes())))

df_total = pd.DataFrame(row_list, columns=columns)
df_total = df_total[["ADM3_PCODE"]]

#%% Obtaining zonal histogram for each file
for file in file_list:

    print(file)

    # Loading raster layer
    raster_path = os.path.join(cdir, "data\\QGIS\\PRISM\\PRISM_new", file)
    raster = QgsRasterLayer(raster_path)

    # Obtaining zonal histogram
    Processing.initialize()
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    input_raster = raster
    input_vector = shp

    params = {
        "COLUMN_PREFIX": "HISTO_",
        "INPUT_RASTER": input_raster,
        "INPUT_VECTOR": input_vector,
        "OUTPUT": "TEMPORARY_OUTPUT",
        "RASTER_BAND": 1,
    }

    result = processing.run("native:zonalhistogram", params)
    layer = result["OUTPUT"]

    # convert legend to dictionary
    legend = raster.legendSymbologyItems()
    key = []
    value = []

    for i in range(2, len(legend)):

        list_str = legend[i][0].split()
        key.append("HISTO_" + list_str[0])
        value.append(list_str[2])

    value_date = pd.to_datetime(pd.Series(value), format="%d-%b-%Y")
    legend_dict = dict(zip(key, value_date))

    # Turning attribute table into dataframe
    columns = [f.name() for f in layer.fields()]
    columns_types = [f.typeName() for f in layer.fields()]

    row_list = []
    for f in layer.getFeatures():
        row_list.append(dict(zip(columns, f.attributes())))

    df_histo = pd.DataFrame(row_list, columns=columns)

    # Only keeping histogram columns and municipality code
    columns = [col for col in df_histo.columns if col in key]
    columns.append("ADM3_PCODE")
    df_histo_filtered = df_histo[columns]

    # Renaming columns to corresponding plant date
    df_histo_filtered = df_histo_filtered.rename(columns=legend_dict)

    # TODO: check if this is correct
    # Merging with the total dataframe
    df_total = pd.merge(df_total, df_histo_filtered, on="ADM3_PCODE")


#%%
df_total.to_excel("data\\rice_area_planted.xlsx", index=False)

