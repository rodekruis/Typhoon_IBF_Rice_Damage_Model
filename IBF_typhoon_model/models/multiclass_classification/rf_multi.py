"""
Loading in the libraries
"""
#%% General Libraries
import numpy as np
from numpy.lib.function_base import average, select
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


def rf_multi_features(
    X,
    y,
    features,
    search_space,
    cv_splits,
    class_weight,
    min_features_to_select,
    GS_score,
    GS_randomized,
    GS_n_iter,
):

    cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    rf = RandomForestClassifier(class_weight=class_weight)

    selector = RFECV(
        rf, step=1, cv=4, verbose=0, min_features_to_select=min_features_to_select
    )

    if GS_randomized == True:
        clf = RandomizedSearchCV(
            selector,
            param_distributions=search_space,
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
            param_grid=search_space,
            scoring=GS_score,
            cv=cv_folds,
            verbose=10,
            return_train_score=True,
            refit=True,
        )

    clf.fit(X, y)
    selected = list(clf.best_estimator_.support_)
    selected_features = [x for x, y in zip(features, selected) if y == True]
    selected_params = clf.best_params_

    return selected_features, selected_params


def rf_multi_performance(
    df_train_list,
    df_test_list,
    features,
    search_space,
    stratK,
    cv_splits,
    class_weight,
    GS_score,
    GS_randomized,
    GS_n_iter,
):

    train_score = []
    test_score = []
    selected_params = []
    df_predicted = pd.DataFrame(columns=["year", "actual", "predicted"])

    for i in range(len(df_train_list)):

        print(f"Running for {i+1} out of a total of {len(df_train_list)}")

        train = df_train_list[i]
        test = df_test_list[i]

        x_train = train[features]
        y_train = train["class_value_multi"]

        x_test = test[features]
        y_test = test["class_value_multi"]

        # Stratified or non-stratified CV
        if stratK == True:
            cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True)
        else:
            cv_folds = KFold(n_splits=cv_splits, shuffle=True)

        steps = [("rf", RandomForestClassifier(class_weight=class_weight))]
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
        rf_fitted = mod.fit(x_train, y_train)
        results = rf_fitted.cv_results_

        y_pred_test = rf_fitted.predict(x_test)
        y_pred_train = rf_fitted.predict(x_train)

        train_score_f1 = f1_score(y_train, y_pred_train, average="macro")
        test_score_f1 = f1_score(y_test, y_pred_test, average="macro")

        train_score.append(train_score_f1)
        test_score.append(test_score_f1)

        df_predicted_temp = pd.DataFrame(
            {"year": test["year"], "actual": y_test, "predicted": y_pred_test}
        )

        df_predicted = pd.concat([df_predicted, df_predicted_temp])
        selected_params.append(rf_fitted.best_params_)

        print(f"Selected Parameters: {rf_fitted.best_params_}")
        print(f"Train score: {train_score_f1}")
        print(f"Test score: {test_score_f1}")

    return df_predicted, selected_params

