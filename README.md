# Typhoon IBF Rice Damage Model

This GitHub Repository covers the 'Rice Damage Model' of the Impact Based Forecasting project of 510. It has been executed in collaboration with the German and Philippines Red Cross, the Food and Agriculture Organization, the Philippines Department of Agriculture and Philippine Rice Research Institure. The main aim was to develop a model that can be used to predict damages to rice fields, on municipality level, before a typhoon makes landfall. To this extent, three sets of models have been implementend; a binary classification, a multiclass classifcication and a regression.  

## Dependent Variable

The dependent variable used in the model is the percentage of standing rice area damaged.

## Features

The features used cover a set of exposure and vulnerability indicators (municipality specific) and a set of hazard indicators (typhoon and municipality specific).

**Exporsure and vulnerability indicators**:
- Area
- Latitude
- Longitude
- Perimeter
- Coast length
- Coast binary
- Mean elevation
- Mean ruggedness
- Ruggedness stdv
- Mean slope
- Slope stdv
- Coast - Perimeter ratio
- Poverty percentage

**Hazard indicators**:
- Maximum 6 hour rainfall (mm/h)
- Maximum 24 hour rainfall (mm/h)
- Minimum track distance (km)
- Maximum wind speed (m/s, 1 minute average)


## Binary Classification

The binary classification models predicts whether the damages is above or below 30%

## Multiclass Classification

The multiclass classification has three classes:
- 0 - 30%
- 30% - 80%
- \> 80%

## Regression

The regression model predicts on a continuous scale.

## GitHub Repo Structure

This repo contains the data and code used in the projects including documentation to explain the steps taken. The two main files are:

- [The data preparation notebook](https://github.com/rodekruis/Typhoon_IBF_Rice_Damage_Model/blob/main/IBF_typhoon_model/data/data_preparation_notebook.ipynb) <br>
This notebook contains a step-by-step process of obtaining and processing all the data, resulting in the final input data set used in the model.


- [The model results notebook](https://github.com/rodekruis/Typhoon_IBF_Rice_Damage_Model/blob/main/IBF_typhoon_model/models/model_results.ipynb) <br>
This notebook contains a step-by-step process of training and testing a set of models, to find the optimal ones, on the input data obtained.

The folder contain all supporting files, brief documentation and code comments.