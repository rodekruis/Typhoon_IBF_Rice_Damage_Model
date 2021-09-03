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


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred))
    return "f1_err", err


def xgb_binary_features(
    X,
    y,
    features,
    search_space,
    objective,
    cv_splits,
    min_features_to_select,
    GS_score,
    GS_randomized,
    GS_n_iter,
):

    cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True)

    xgb = XGBClassifier(use_label_encoder=False, objective=objective, n_jobs=0,)

    selector = RFECV(
        xgb, step=1, cv=4, verbose=10, min_features_to_select=min_features_to_select
    )

    cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True)

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
    print(selected_features)

    return selected_features


def xgb_binary_performance(
    df_train_list,
    df_test_list,
    features,
    search_space,
    stratK,
    cv_splits,
    objective,
    GS_score,
    GS_randomized,
    GS_n_iter,
):

    train_score = []
    test_score = []
    df_predicted = pd.DataFrame(columns=["year", "actual", "predicted"])

    for i in range(len(df_train_list)):

        print(f"Running for {i+1} out of a total of {len(df_train_list)}")

        train = df_train_list[i]
        test = df_test_list[i]

        x_train = train[features]
        y_train = train["class_value_binary"]

        x_test = test[features]
        y_test = test["class_value_binary"]

        # negative instances / positive instances
        # weight_scale = sum(y_train == 0) / sum(y_train == 1)

        # Stratified or non-stratified CV
        if stratK == True:
            cv_folds = StratifiedKFold(n_splits=cv_splits, shuffle=True)
        else:
            cv_folds = KFold(n_splits=cv_splits, shuffle=True)

        steps = [
            (
                "xgb",
                XGBClassifier(use_label_encoder=False, objective=objective, n_jobs=0,),
            )
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
        xgb_fitted = mod.fit(x_train, y_train, xgb__eval_metric=f1_eval)
        results = xgb_fitted.cv_results_

        y_pred_test = xgb_fitted.predict(x_test)
        y_pred_train = xgb_fitted.predict(x_train)

        train_score_f1 = f1_score(y_train, y_pred_train)
        test_score_f1 = f1_score(y_test, y_pred_test)

        train_score.append(train_score_f1)
        test_score.append(test_score_f1)

        df_predicted_temp = pd.DataFrame(
            {"year": test["year"], "actual": y_test, "predicted": y_pred_test}
        )

        df_predicted = pd.concat([df_predicted, df_predicted_temp])

        print(f"Train score: {train_score_f1}")
        print(f"Test score: {test_score_f1}")

    return df_predicted

