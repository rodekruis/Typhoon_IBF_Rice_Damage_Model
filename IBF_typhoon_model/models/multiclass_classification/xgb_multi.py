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

#%% Defining the f1 function for eval_metric
def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = f1_score(y_true, np.round(y_pred), average="macro")
    return "f1_err", -err


#%% Setting input variables
classes = {"0": [0, 0.3], "1": [0.3, 0.8], "2": [0.8, 1.1]}
cv_splits = 5
GS_score = "f1_macro"
GS_randomized = True
GS_n_iter = 10
min_features_to_select = 15

objective = "multi:softmax"  # to output 0 or 1 instead of probability
num_class = 3
eval_metric = f1_eval

learning_rate_space = [0.1, 0.5, 1]
gamma_space = [0.1, 0.5, 2]
max_depth_space = [6, 8]
reg_lambda_space = [0.001, 0.01, 0.1, 1]
n_estimators_space = [100, 200]
colsample_bytree_space = [0.5, 0.7, 1]

#%% Obtaining the features to be used using Recursive Feature Elimination
# With Cross Validation to select number of features
random.seed(1)


def determine_class(x):

    for key, value in classes.items():

        if value[0] <= x < value[1]:

            class_value = int(key)
            break

    return class_value


df["class_value"] = df["perc_loss"].apply(determine_class)

X = df[features]
y = df["class_value"]

param_grid = {
    "estimator__learning_rate": learning_rate_space,
    "estimator__gamma": gamma_space,
    "estimator__max_depth": max_depth_space,
    "estimator__reg_lambda": reg_lambda_space,
    "estimator__n_estimators": n_estimators_space,
    "estimator__colsample_bytree": colsample_bytree_space,
}

cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

xgb = XGBClassifier(
    use_label_encoder=False, objective=objective, n_jobs=0, num_class=num_class
)

selector = RFECV(
    xgb, step=1, cv=4, verbose=10, min_features_to_select=min_features_to_select
)
cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True)

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
# region
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

#%% Setting variables used in the model
classes = {"0": [0, 0.3], "1": [0.3, 0.8], "2": [0.8, 1.1]}
cv_splits = 5
GS_score = "f1_macro"
stratK = True
GS_randomized = False
GS_n_iter = 10

objective = "multi:softmax"  # to output 0 or 1 instead of probability
num_class = 3
eval_metric = f1_eval

learning_rate_space = [0.1]
gamma_space = [0.1]
max_depth_space = [6]
reg_lambda_space = [1]
n_estimators_space = [100]
colsample_bytree_space = [0.7]

# Adding class value to df
df["class_value"] = df["perc_loss"].apply(determine_class)

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

    steps = [
        (
            "xgb",
            XGBClassifier(
                use_label_encoder=False,
                objective=objective,
                n_jobs=0,
                num_class=num_class,
                eval_metric=f1_eval,
            ),
        )
    ]
    pipe = Pipeline(steps, verbose=0)

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
    xgb_fitted = mod.fit(x_train, y_train, xgb__eval_metric=f1_eval)
    results = xgb_fitted.cv_results_

    y_pred_test = xgb_fitted.predict(x_test)
    y_pred_train = xgb_fitted.predict(x_train)

    train_score_f1 = f1_score(y_train, y_pred_train, average="macro")
    test_score_f1 = f1_score(y_test, y_pred_test, average="macro")

    train_score.append(train_score_f1)
    test_score.append(test_score_f1)

    df_predicted_temp = pd.DataFrame(
        {"year": [year] * len(y_test), "actual": y_test, "predicted": y_pred_test}
    )

    df_predicted = pd.concat([df_predicted, df_predicted_temp])

    print(f"Train score: {train_score_f1}")
    print(f"Test score: {test_score_f1}")

# endregion
# %%
