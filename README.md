# Typhoon IBF Rice Damage Model

This GitHub Repository covers the 'Rice Damage Model' of the Impact Based Forecasting project of 510. It has been executed in collaboration with the German and Philippines Red Cross, the Food and Agriculture Organization, the Philippines Department of Agriculture and Philippine Rice Research Institure. The main aim was to develop a model that can be used to predict damages to rice fields, on municipality level, before a tropical cyclone (TC) makes landfall. 

## Dependent Variable

The dependent variable used in the model is the percentage of standing rice area damaged.

## Features

The features used cover a set of exposure and vulnerability indicators (municipality specific) and a set of hazard indicators (TC and municipality specific).

**Exporsure and vulnerability indicators**:
- Area (km^2)
- Latitude
- Longitude
- Perimeter (m)
- Coast length (m)
- Coast binary <br>
    1 if the municipality is by the coast, 0 if it is not
- Mean elevation (m)
- Mean ruggedness (m)
- Ruggedness stdv (m)
- Mean slope (%)
- Slope stdv (%)
- Coast - Perimeter ratio 
- Poverty percentage 

**Hazard indicators**:
- Rainfall variables  
  Two sets of rainfall variables were created to test two approaches:  
  (1) a **spatial extent approach**, which determines entry and exit times based on the TC’s modelled spatial extent (ROCI from IBTrACS), and  
  (2) a **landfall approach**, which assumes municipalities experience TC effects from **3 hours before** to **24 hours after landfall**.  
  Rainfall data was obtained from NASA GPM IMERG (10 km resolution, 30-minute intervals).  
  The pre-TC period covers **72 hours before entry**, and the post-TC period **48 hours after exit**.  
  Testing showed that the simpler landfall approach resulted in **no statistically significant change in model performance**. A variety of variables was selected, in order to select the best one for each cluster during feature selection.

  - Pre-TC rainfall variables (72 hours before TC entry)
    - Maximum 6-hour rainfall (mm/h) <br>
      Maximum rainfall intensity calculated over a rolling 6-hour window during the 72 hours before the TC entered the municipality.
    - Maximum 24-hour rainfall (mm/h) <br>
      Maximum rainfall intensity calculated over a rolling 24-hour window during the 72 hours before the TC entered the municipality.

  - During-TC rainfall variables (between entry and exit)
    - Total rainfall (mm) <br>
      Total accumulated rainfall during the TC’s presence in the municipality.
    - Mean rainfall (mm/h) <br>
      Average rainfall intensity during the TC’s presence.
    - Maximum rainfall (mm/h) <br>
      Highest recorded rainfall intensity during the TC period.
    - Maximum 1-hour rainfall (mm/h) <br>
      Maximum rainfall intensity over a rolling 1-hour window during the TC (landfall approach only).
    - Maximum 3-hour rainfall (mm/h) <br>
      Maximum rainfall intensity over a rolling 3-hour window during the TC (landfall approach only).
    - Maximum 6-hour rainfall (mm/h) <br>
      Maximum rainfall intensity over a rolling 6-hour window during the TC (landfall approach only).

  - Post-TC rainfall variables (48 hours after TC exit)
    - Maximum 6-hour rainfall (mm/h) <br>
      Maximum rainfall intensity calculated over a rolling 6-hour window during the 48 hours after the TC left the municipality.
    - Maximum 24-hour rainfall (mm/h) <br>
      Maximum rainfall intensity calculated over a rolling 24-hour window during the 48 hours after the TC left the municipality.


## Binary Classification

The binary classification models predicts whether the damages is above or below 30%

## Regression

The regression model predicts on a continuous scale.

## GitHub Repo Structure

The repository contains two main folders: **data** and **models**. Further details are provided in the README files within each folder.

  - **data**  
    Contains all scripts and resources related to **data collection and processing** used to construct the final modelling dataset.  
    Subfolders include data preparation scripts, rainfall and wind data processing, GIS data, administrative boundary shapefiles, figure scripts, rice data processing scripts, and a `restricted_data` folder containing datasets that cannot be publicly shared.

  - **models**  
    Contains the **model pipelines and results** for the machine learning models used in this study. This includes implementations of **XGBoost** and **Random Forest** models for both **regression** and **binary classification**.