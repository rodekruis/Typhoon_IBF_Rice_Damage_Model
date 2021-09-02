#%% Import packages
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import os
from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
#%% Setting input varialbes
cv_splits = 5
GS_score = "neg_root_mean_squared_error"
GS_randomized = True
GS_n_iter = 10
min_features_to_select = 15

objective = "reg:squarederror"
learning_rate_space = [0.1, 0.5, 1]
gamma_space = [0.1, 0.5, 2]
max_depth_space = [6, 8]
reg_lambda_space = [0.001, 0.01, 0.1, 1]
n_estimators_space = [100, 200]
colsample_bytree_space = [0.5, 0.7, 1]

X = df[features]
y = df["perc_loss"]

param_grid = {
    "estimator__learning_rate": learning_rate_space,
    "estimator__gamma": gamma_space,
    "estimator__max_depth": max_depth_space,
    "estimator__reg_lambda": reg_lambda_space,
    "estimator__n_estimators": n_estimators_space,
    "estimator__colsample_bytree": colsample_bytree_space,
}

cv_folds = KFold(n_splits=cv_splits, shuffle=True)

xgb = XGBRegressor(objective=objective)

selector = RFECV(
    xgb, step=1, cv=4, verbose=10, min_features_to_select=min_features_to_select
)

if GS_randomized == True:
    regr = RandomizedSearchCV(
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
    regr = GridSearchCV(
        selector,
        param_grid=param_grid,
        scoring=GS_score,
        cv=cv_folds,
        verbose=10,
        return_train_score=True,
        refit=True,
    )

regr.fit(X, y)
regr.best_estimator_.estimator_
regr.best_estimator_.grid_scores_
regr.best_estimator_.ranking_

selected = list(regr.best_estimator_.support_)
selected_features = [x for x, y in zip(features, selected) if y == True]
print(selected_features)


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
cv_splits = 5
GS_randomized = True
GS_n_iter = 10
GS_score = "neg_root_mean_squared_error"

objective = "reg:squarederror"
learning_rate_space = [0.1, 0.5, 1]
gamma_space = [0.1, 0.5, 2]
max_depth_space = [6, 8]
reg_lambda_space = [0.001, 0.01, 0.1, 1]
n_estimators_space = [100, 200]
colsample_bytree_space = [0.5, 0.7, 1]

#%% Starting the pipeline

random.seed(1)

train_score_mae_list = []
test_score_mae_list = []
train_score_rmse_list = []
test_score_rmse_list = []
df_predicted = pd.DataFrame(columns=["year", "actual", "predicted"])

for year in years:

    print(year)

    train = df[df["year"] != year]
    test = df[df["year"] == year]
    x_train = train[selected_features]
    y_train = train["perc_loss"]
    x_test = test[selected_features]
    y_test = test["perc_loss"]

    search_space = [
        {
            "xgb__learning_rate": learning_rate_space,
            "xgb__gamma": gamma_space,
            "xgb__max_depth": max_depth_space,
            "xgb__reg_lambda": reg_lambda_space,
            "xgb__n_estimators": n_estimators_space,
            "xgb__colsample_bytree": colsample_bytree_space,
        }
    ]

    cv_folds = KFold(n_splits=cv_splits, shuffle=True)

    steps = [
        ("xgb", XGBRegressor(objective=objective)),
    ]

    pipe = Pipeline(steps, verbose=0)

    # Applying GridSearch or RandomizedGridSearch
    if GS_randomized == True:
        mod = RandomizedSearchCV(
            pipe,
            search_space,
            scoring=GS_score,
            cv=cv_folds,
            verbose=10,
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
            verbose=10,
            return_train_score=True,
            refit=True,
        )

    # Fitting the model on the full dataset
    xgb_fitted = mod.fit(x_train, y_train)
    results = xgb_fitted.cv_results_

    # to obtain the train score
    y_pred_train = xgb_fitted.predict(x_train)

    # obtaining the predicted test values
    y_pred_test = xgb_fitted.predict(x_test)

    # TODO not needed for XGBoost? --> cannot extrapolate
    # y_pred_train = [0 if x < 0 else 1 if x > 1 else x for x in y_pred_train]
    # y_pred_test = [0 if x < 0 else 1 if x > 1 else x for x in y_pred_test]

    train_score_mae = mean_absolute_error(y_pred_train, y_train)
    test_score_mae = mean_absolute_error(y_pred_test, y_test)
    train_score_rmse = mean_squared_error(y_pred_train, y_train, squared=False)
    test_score_rmse = mean_squared_error(y_pred_test, y_test, squared=False)

    train_score_mae_list.append(train_score_mae)
    test_score_mae_list.append(test_score_mae)
    train_score_rmse_list.append(train_score_rmse)
    test_score_rmse_list.append(test_score_rmse)

    df_predicted_temp = pd.DataFrame(
        {"year": [year] * len(y_test), "actual": y_test, "predicted": y_pred_test}
    )

    df_predicted = pd.concat([df_predicted, df_predicted_temp])

    print(f"Train score: {train_score_mae}")
    print(f"Test score: {test_score_rmse}")

# %%

