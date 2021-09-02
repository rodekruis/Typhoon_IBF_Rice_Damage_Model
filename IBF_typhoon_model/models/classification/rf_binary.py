"""
Loading in the libraries
"""
#%% General Libraries
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import importlib
import os
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    mutual_info_regression,
    f_regression,
    mutual_info_classif,
)
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
import xgboost as xgb
import random
import pickle
import openpyxl
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV


"""
Loading in the data
"""
#%% Setting path to the initial folder
os.chdir("C:\\Users\\Marieke\\GitHub\\Typhoon_IBF_Rice_Damage_Model")
cdir = os.getcwd()

#%% Input data
name = "IBF_typhoon_model\\data\\restricted_data\\combined_input_data\\input_data.xlsx"
path = os.path.join(cdir, name)
df = pd.read_excel(path, engine="openpyxl")

#%% Typhoon overview
file_name = "IBF_typhoon_model\\data\\restricted_data\\data_overview.xlsx"
path = os.path.join(cdir, file_name)
df_typh_overview = pd.read_excel(path, sheet_name="typhoon_overview", engine="openpyxl")

#%% Selecting the features to be used
features = [
    "rice_area",
    "mean_slope",
    "mean_elevation_m",
    "ruggedness_stdev",
    "mean_ruggedness",
    "slope_stdev",
    "area_km2",
    "poverty_perc",
    "with_coast",
    "coast_length",
    "perimeter",
    "glat",
    "glon",
    "coast_peri_ratio",
    "rainfall_sum",
    "rainfall_max",
    "dis_track_min",
    "vmax_sust",
]


"""
Full model for feature selection
"""
# region

#%% Setting input varialbes
threshold = 0.3
cv_splits = 5
GS_score = "f1"
class_weight = "balanced"
GS_randomized = True
GS_n_iter = 10
min_features_to_select = 15

n_estimators_space = [20]
max_depth_space = [None]
min_samples_split_space = [2]
min_samples_leaf_space = [1]

#%% Obtaining the features to be used using Recursive Feature Elimination
# With Cross Validation to select number of features
random.seed(1)

df["class_value"] = [1 if df["perc_loss"][i] > threshold else 0 for i in range(len(df))]
X = df[features]
y = df["class_value"]

param_grid = {
    "estimator__n_estimators": n_estimators_space,
    "estimator__max_depth": max_depth_space,
    "estimator__min_samples_split": min_samples_split_space,
    "estimator__min_samples_leaf": min_samples_leaf_space,
}

cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
rf = RandomForestClassifier(class_weight=class_weight)
selector = RFECV(
    rf, step=1, cv=4, verbose=10, min_features_to_select=min_features_to_select
)

if GS_randomized == True:
    clf = RandomizedSearchCV(
        selector,
        param_distributions=param_grid,
        scoring=GS_score,
        cv=cv_folds,
        verbose=10,
        return_train_score=True,
        refit=True,
        n_iter=GS_n_iter,
    )
else:
    clf = GridSearchCV(
        selector,
        param_grid=param_grid,
        scoring=GS_score,
        cv=cv_folds,
        verbose=10,
        return_train_score=True,
        refit=True,
    )

clf.fit(X, y)
clf.best_estimator_.estimator_
clf.best_estimator_.grid_scores_
clf.best_estimator_.ranking_

selected = list(clf.best_estimator_.support_)
selected_features = [x for x, y in zip(features, selected) if y == True]
print(selected_features)

# endregion


"""
Training with CV to obtain performance estimate
"""

#%% Defining the train and test sets
# Create id and year dictionary
id_year_dict = dict(zip(df_typh_overview["storm_id"], df_typh_overview["year"]))
df["year"] = df["storm_id"].map(id_year_dict)

# To save the train and test sets
df_train_list = []
df_test_list = []

# List of years that are to be used as a test set
years = [2016, 2017, 2018, 2019, 2020]
for year in years:

    df_train_list.append(df[df["year"] != year])
    df_test_list.append(df[df["year"] == year])

#%% Setting the variables used in the model
threshold = 0.3
cv_splits = 5
GS_randomized = False
GS_n_iter = 10
GS_score = "f1"
stratK = True
class_weight = "balanced"  # None
n_estimators_space = [60]
max_depth_space = [None]
min_samples_split_space = [2]
min_samples_leaf_space = [1, 3]

# Adding class value to df
df["class_value"] = [1 if df["perc_loss"][i] > threshold else 0 for i in range(len(df))]

#%% Starting the pipeline
random.seed(1)

train_score = []
test_score = []
df_predicted = pd.DataFrame(columns=["year", "actual", "predicted"])

for year in years:

    print(year)

    train = df[df["year"] != year]
    test = df[df["year"] == year]

    x_train = train[selected_features]
    y_train = train["class_value"]

    x_test = test[selected_features]
    y_test = test["class_value"]

    # Stratified or non-stratified CV
    if stratK == True:
        cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True)
    else:
        cv_folds = KFold(n_splits=cv_splits, shuffle=True)

    steps = [("rf", RandomForestClassifier(class_weight=class_weight))]
    pipe = Pipeline(steps, verbose=0)

    search_space = [
        {
            "rf__n_estimators": n_estimators_space,
            "rf__max_depth": max_depth_space,
            "rf__min_samples_split": min_samples_split_space,
            "rf__min_samples_leaf": min_samples_leaf_space,
        }
    ]

    # Applying GridSearch or RandomizedGridSearch
    if GS_randomized == True:
        mod = RandomizedSearchCV(
            pipe,
            search_space,
            scoring=GS_score,
            cv=cv_folds,
            verbose=0,
            return_train_score=True,
            refit=True,
            n_iter=GS_n_iter,
        )
    else:
        mod = GridSearchCV(
            pipe,
            search_space,
            scoring=GS_score,
            cv=cv_folds,
            verbose=0,
            return_train_score=True,
            refit=True,
        )

    # Fitting the model on the full dataset
    rf_fitted = mod.fit(x_train, y_train)
    results = rf_fitted.cv_results_

    y_pred_test = rf_fitted.predict(x_test)
    y_pred_train = rf_fitted.predict(x_train)

    train_score_f1 = f1_score(y_train, y_pred_train)
    test_score_f1 = f1_score(y_test, y_pred_test)

    train_score.append(train_score_f1)
    test_score.append(test_score_f1)

    df_predicted_temp = pd.DataFrame(
        {"year": [year] * len(y_test), "actual": y_test, "predicted": y_pred_test}
    )

    df_predicted = pd.concat([df_predicted, df_predicted_temp])

    print(f"Train score: {train_score_f1}")
    print(f"Test score: {test_score_f1}")


# %%
