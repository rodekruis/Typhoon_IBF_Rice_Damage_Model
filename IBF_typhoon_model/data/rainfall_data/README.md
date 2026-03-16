# Rainfall data

This folder contains the needed data and scripts for obtaining and processing the rainfall data. The downloaded data is the 'Global Precipitation Measurement' obtained from NASA. Information on the data, what it entails and how it can be obtained can be found at the bottom. There are two types of rainfall that can be downloaded and processed: daily and half hourly rainfall. Due to the size of the downloaded GPM files, they are not included on the GitHub, but can be downloaded locally.

## HHR (half hourly) Rainfall

Containing the precipitation rates on an half hourly interval. This is used to obtain diverse rainfall variables.

**Scripts**
- rainfall_hhr <br>
    Collects the half hourly rainfall precipitation rate in mm/h for the range of the typhoon.

**Input**
- metadata_typhoon <br>
    csv file with the main typhoon information
- expanded_metadata_typhoon <br>
    csv file with the main typhoon information and extra information relating to the extent based rainfall method. Specifically it looks at at the earliest entry and latest exit date-times for each typhoon event according to the typhoon's modelled extent.
- more_metadata_creation.ipynb <br>
    Creates the expanded metadata for the typhoons described above. Also calculates the difference between landfall and the earliest entry of the typhoon extent for all typhoons.

**Output**
- for each typhoon: a csv file with half hourly precipitation rate in mm/h

## Rainfall Processing

This scripts uses the collected Half Hourly Rainfall data for each typhoon to obtain the maximum precipitation rate in a certain time interfall and a rolling window. Two scripts are used, one for the extent based method and the other one that uses simply the landfall.

The extent based method uses typhoon_enter_exit_per_municipality.csv from the gis_data folder, since the extent based method calculates the rainfall variables were each municipality has a different entry and exit time depending on the typhoon extent, whereas the landfall based method uses the same time for all municipality (based on the landfall date-time of the typhoon).

The following variables are obtained for both the extent and landfall based methods.
1. pre_max_6h	
2. pre_max_24h
3. during_total_rainfall	
4. during_max_rainfall	
5. during_mean_rainfall	
6. during_max_1h_intensity
7. during_max_3h_intensity
8. during_max_6h_intensity
9. post_max_6h
10. post_max_24h

**Output**
- lf_rainfall_variables.csv : csv sheet with the rainfall variables for the landfall based method, for each municipality and typhoon
- rainfall_variables.csv : csv sheet with the rainfall variables for the extent based method, for each municipality and typhoon

## Other files

- finding_extent.ipynb <br>
    Helper code to find out the extent of the typhoons in geographical coordinates, used as input in rainfall_hhr to download the GPM files.
- verificaiton_typhoon_precipitation.ipynb <br>
    Verification code to verify that the tracks and extents correspond with the GPM precipitation data.

## Information Links

- NASA Global Precipitation Measurement: https://gpm.nasa.gov/data/directory
- The precipitation processing system at NASA Goddard: https://gpm.nasa.gov/sites/default/files/2020-06/IMERG-GIS-Readme_4_22_20.pdf
- Accessing the data: https://gpm.nasa.gov/sites/default/files/2021-01/jsimpsonhttps_retrieval.pdf
