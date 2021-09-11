# Data Folder

This folder contains the data collection and processing. 

## Data Preparation Notebook
In this notebook, all collected data is processed to obtain the final dataset that is used as model input. 

## Figures
Contains both scripts for plotting figures and the outputs

## GIS data
Contains the geographical information related to the typhoon tracks.

## PHL administrative boundaries
Contains the shape file for the administrative boundaries of the Philippines, at level 3, which is the municipality level. As we are making predictions at municipality level.

## Rainfall data
Contains the collected and processed rainfall data, as well as scripts for collecting and processing.

## Wind data
Contains the obtained windspeeds and scripts for processing, using the Climada package

## Rice data
The actual rice data (both on the area planted and the rice losses incurred) is not made publicly available. This folder contains the script used to process the rice data obtained through PRiSM (the Philippines Rice inforamtion System). With it, the PRiSM raster files that contain info on the rice area planted over time can be processed to obtain the rice area planted in a municipality at a specific date. It results in a dataframe (which is in the restricted data folder) that shows the municipality and all the available dates, with the rice area planted in HA. 


## Restricted data
Several data files used in this project cannot be made publicly available. There is therefore a folder 'restricted_data' that can be obtained through 510 and should be placed in the 'data' folder. This contains the following data
- the combined input data used as input for the models
- data on the rice area over time
- data on the rice losses incurred in the municipality for the given typhoons