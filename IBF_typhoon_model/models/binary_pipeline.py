import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from typing import List, Dict, Any, Tuple
import json
import sys
import random
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.base import BaseSampler
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

class SafeADASYN(BaseSampler):
    '''
    This custom ADASYN class is designed to handle the ADASYN error of no samples generated for dataset, which disrupts the whole 
    pipeline, for the cross validation during final bayes hyperparameter optimization/tuning. Its designed to adhere to scikit-learn's 
    and imbalanced-learn's interfaces 
    '''
    def __init__(self, sampling_strategy="auto", random_state=None, n_neighbors=5):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, n_neighbors=n_neighbors)

    def _fit_resample(self, X, y):
        """Apply ADASYN and handle potential errors."""
        X, y = check_X_y(X, y) # Ensure that X and y are of the appropriate format and type
        try: 
            # Perform ADASYN oversampling
            X_resampled, y_resampled = self.adasyn.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=self.feature_names_in_), y_resampled
        except ValueError as e:
            # If ADASYN fails, return the original dataset
            print(f"ADASYN error: {e}. Returning original data.") # Store the feature names from the input data
            return pd.DataFrame(X, columns=self.feature_names_in_), y

    def fit(self, X, y=None):
        """Fit method to conform with the scikit-learn pipeline interface."""
        self.feature_names_in_ = X.columns # Store the feature names from the input data
        self._fit_resample(X, y) # Call the _fit_resample method to potentially fit and resample the data
        return self

    def fit_resample(self, X, y):
        """Overloaded method to match imbalanced-learn's resampler interface."""
        self.feature_names_in_ = X.columns # Store the feature names from the input data
        return self._fit_resample(X, y) # Call the _fit_resample method and return the result

""" Saving functions """

def save_roc_curve_and_auc(y_true, y_pred_proba, year):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # Convert the ROC data to a DataFrame
    roc_data_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    })
    roc_data_df['auc'] = auc_score

    # Define the path for the ROC plots directory
    roc_plot_dir = os.path.join(RESULTS_PATH, 'roc_plots', f'year_{year}')

    # Create the directory for the current year if it doesn't exist
    if not os.path.exists(roc_plot_dir):
        os.makedirs(roc_plot_dir)

    # Save the ROC data as a CSV file
    roc_data_df.to_csv(os.path.join(roc_plot_dir, f'roc_curve_{year}.csv'), index=False)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Year {year}')
    plt.legend(loc="lower right")

    # Save plot
    # plt.savefig(os.path.join(roc_plot_dir, f'roc_curve_{year}.png'))
    plt.savefig(os.path.join(roc_plot_dir, f'roc_curve_{year}.svg'))
    plt.close()

def save_initial_config(config):
    # Helper function to convert the search space format to a JSON-serializable format
    def convert_space(space):
        converted_space = {}
        for key, value in space.items():
            if isinstance(value, Integer):
                # Convert Integer type parameters
                converted_space[key] = {
                    "type": "Integer",
                    "low": value.low,
                    "high": value.high
                }
            elif isinstance(value, Real):
                # Convert Real type parameters
                converted_space[key] = {
                    "type": "Real",
                    "low": value.low,
                    "high": value.high,
                    "prior": value.prior  # Include the prior if it's specified
                }
            else:
                converted_space[key] = value  # Handle other possible cases if needed
        return converted_space

    # Convert the search space using the function above
    config["Hyperparameter Search Space"] = convert_space(config["Hyperparameter Search Space"])

    # Save the configuration as JSON
    with open(os.path.join(RESULTS_PATH, 'initial_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def save_outer_fold_results(year, results, train_data, test_data, y_pred_proba):
    def convert_np_int64(obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


    # Calculate train and test counts
    train_class_counts = train_data[TARGET].value_counts().to_dict()
    test_class_counts = test_data[TARGET].value_counts().to_dict()
    train_total_count = train_data.shape[0]
    test_total_count = test_data.shape[0]

    # Calculate AUC score
    auc_score = roc_auc_score(test_data[TARGET], y_pred_proba)
    
    # Organize the results by run
    feature_reduction = [
        {
            "run": i + 1,
            "cv_results": results['cv_results'][i],
            "feature_importance": results['feature_importance'][i],
            "initial_params": results['initial_params'][i]
        }
        for i in range(len(results['cv_results']))
    ]

    # Create the JSON structure
    results_json = {
        "feature_reduction": feature_reduction,
        "feature_frequency": results['feature_frequency'],
        "best_features": results['best_features'],
        "final_bayes_optimization": results['final_bayes_results'],
        "adasyn_results_for_final_bayes_optimization": results['adasyn_results'],
        "data_counts": {
            "train_total_count": train_total_count,
            "test_total_count": test_total_count,
            "train_class_counts": train_class_counts,
            "test_class_counts": test_class_counts
        },
        "class_distribution_before": results['class_distribution_before'],
        "class_distribution_after": results.get('class_distribution_after', "ADASYN was not applied"),
        "final_params": results['final_params'],
        "test_metrics": results['test_metrics']['metrics'],
        "confusion_matrix": results['test_metrics']['confusion_matrix'],
        "auc_score": auc_score  # Save AUC score with outer fold results
    }

    # Convert np.int64 to int in the results_json
    results_json = json.loads(json.dumps(results_json, default=convert_np_int64))

    # Save the results to a JSON file
    with open(os.path.join(RESULTS_PATH, f'outer_fold_{year}_results.json'), 'w') as f:
        json.dump(results_json, f, indent=4)
        
def save_overall_results():
    def aggregate_feature_frequencies(outer_fold_results):
        # Calculate the total number of runs across all outer folds
        total_runs = N_RUNS * len(YEARS)

        # Initialize an empty Series to accumulate feature counts
        total_feature_counts = pd.Series(dtype=float)

        # Aggregate feature counts across all outer folds
        for fold_result in outer_fold_results:
            feature_freq = pd.Series(fold_result['feature_frequency'])
            feature_counts = feature_freq * N_RUNS
            total_feature_counts = total_feature_counts.add(feature_counts, fill_value=0)

        # Calculate the overall feature frequency by dividing by the total number of runs
        overall_feature_freq = total_feature_counts / total_runs

        # Convert overall feature frequencies to a dictionary for easy JSON serialization
        return overall_feature_freq.to_dict()

    outer_fold_files = [f for f in os.listdir(RESULTS_PATH) if f.startswith('outer_fold_') and f.endswith('_results.json')]
    
    test_size_per_year = []
    selected_features = defaultdict(int)
    final_hyperparameters = defaultdict(list)
    overall_metrics = {
        "weighted_average": defaultdict(list),
        "normal_average": defaultdict(list)
    }
    overall_confusion_matrix = {
        'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0
    }
    total_test_size = 0
    outer_fold_results = []  # Collecting results for all outer folds
    
    for file in outer_fold_files:
        with open(os.path.join(RESULTS_PATH, file), 'r') as f:
            result = json.load(f)
        
        outer_fold_results.append(result)  # Store result for feature frequency aggregation
        
        year = int(file.split('_')[2])
        test_size = result['data_counts']['test_total_count']
        total_test_size += test_size
        test_size_per_year.append({
            "year": year,
            "total_count": test_size,
            "class_counts": result['data_counts']['test_class_counts']
        })
        
        # Count selected features
        for feature in result['best_features']:
            selected_features[feature] += 1
        
        # Collect best hyperparameters
        for param, value in result['final_params'].items():
            final_hyperparameters[param].append(value)
        
        # Collect metrics
        for metric, values in result['test_metrics'].items():
            if metric in ['precision', 'recall', 'f1_score']:
                for key, val in values.items():
                    if key in ['0', '1']:
                        overall_metrics["weighted_average"][f"{metric}_{key}"].append((val, test_size))
                        overall_metrics["normal_average"][f"{metric}_{key}"].append(val)
                    elif key == 'weighted-average':
                        overall_metrics["weighted_average"][f"(Class)weighted_{metric}"].append((val, test_size))
                        overall_metrics["normal_average"][f"(Class)weighted__{metric}"].append(val)
                    elif key == 'unweighted-average':
                        overall_metrics["normal_average"][f"(Class)unweighted__{metric}"].append(val)
                        overall_metrics["weighted_average"][f"(Class)unweighted_{metric}"].append((val, test_size))
            else:
                overall_metrics["weighted_average"][metric].append((values, test_size))
                overall_metrics["normal_average"][metric].append(values)
        
        # Sum confusion matrices
        cm = result['confusion_matrix']
        overall_confusion_matrix['tp'] += cm.get('tp', 0)
        overall_confusion_matrix['tn'] += cm.get('tn', 0)
        overall_confusion_matrix['fp'] += cm.get('fp', 0)
        overall_confusion_matrix['fn'] += cm.get('fn', 0)
        
        # Store AUC score
        auc = result['auc_score']
        overall_metrics["weighted_average"]["auc"].append((auc, test_size))
        overall_metrics["normal_average"]["auc"].append(auc)
    
    # Aggregate feature frequencies across all outer folds
    overall_feature_freq = aggregate_feature_frequencies(outer_fold_results)
    
    # Compute aggregated results
    aggregated_results = {
        "test_size_per_year": test_size_per_year,
        "selected_features_count": dict(selected_features),
        "overall_feature_frequency": overall_feature_freq,  # Save overall feature frequency
        "final_hyperparameters": {},
        "metrics": {
            "weighted_average": {},
            "normal_average": {}
        },
        "overall_confusion_matrix": overall_confusion_matrix
    }
    
    for param, values in final_hyperparameters.items():
        aggregated_results["final_hyperparameters"][param] = {
            "selected_values": values,
            "range": [min(values), max(values)],
            "mean": np.mean(values),
            "std": np.std(values)
        }
    
    for avg_type, metrics in overall_metrics.items():
        for metric, values in metrics.items():
            if avg_type == "weighted_average":
                # Calculate weighted average and weighted std deviation
                weighted_avg = np.average([val for val, _ in values], weights=[w for _, w in values])
                weighted_var = np.average([(val - weighted_avg) ** 2 for val, _ in values], weights=[w for _, w in values])
                aggregated_results["metrics"][avg_type][metric] = {
                    "metric": weighted_avg,
                    "std": np.sqrt(weighted_var)
                }
            elif avg_type == "normal_average":
                # Calculate normal average and normal std deviation
                normal_avg = np.mean(values)
                normal_std = np.std(values)
                aggregated_results["metrics"][avg_type][metric] = {
                    "metric": normal_avg,
                    "std": normal_std
                }
    
    # Save aggregated results to JSON
    with open(os.path.join(RESULTS_PATH, 'overall_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=4)

""" Util functions """

def load_data(file_path):
    return pd.read_excel(file_path, engine="openpyxl")

def preprocess_data(df):
    df['id'] = df.index # Adding unique identifier to all samples
    df_binary = df[df["damage_above_30"].notnull()]
    df_binary[TARGET] = df_binary["damage_above_30"].apply(lambda x: 1 if x else 0)
    X = df_binary[SELECTED_FEATURES + ["id"]] # Including unique identifier along with features for simplified handling, run_nested_cv_for_year later drops this column from X 
    y = df_binary[TARGET].astype(int)
    return X, y, df_binary

def split_train_test(df):
    df_train_list = [df[df["year"] < year] for year in YEARS]
    df_test_list = [df[df["year"] == year] for year in YEARS]
    return df_train_list, df_test_list

def check_class_distribution(data):
    class_counts = data[TARGET].value_counts()
    print(f"Class distribution:\n{class_counts}\n")
    
def initialize_model(model_type, params=None):
    if model_type == "random_forest":
        return RandomForestClassifier(**(params or {}), random_state=RANDOM_SEED)
    elif model_type == "xgboost":
        return XGBClassifier(**(params or {}), objective=XGB_OBJECTIVE, random_state=RANDOM_SEED)

# Feature ranking and selection functions
def rank_features_by_importance(X, y, model_params):
    model = initialize_model(MODEL_TYPE, model_params) 
    model.fit(X, y) # Fit the model to the provided data (features X and target y)
    importances = model.feature_importances_ # Retrieve the feature importances from the fitted model
    # Create a df with feature names and their corresponding importances, then sort by descending order of importance, return df
    return pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False)

''' Model training and evaluation functions'''

def perform_bayesian_optimization(X, y, seed):
    estimator = initialize_model(MODEL_TYPE) 
    
    # Initialize the BayesSearchCV object with the specified model and search space
    bayes_opt = BayesSearchCV(
        estimator=estimator,  # The machine learning estimator/model
        search_spaces=SEARCH_SPACE,  # The hyperparameter search space
        cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed),  # Cross-validation strategy
        n_iter=N_ITER_I,  # Number of iterations for the Bayesian optimization
        scoring='f1',  # Scoring metric for optimization (F1 score)
        n_jobs=-1,  # Use all available cores for parallel computation
        verbose=0,  # Verbosity level
        random_state=seed  # Random seed for reproducibility
    )
    bayes_opt.fit(X, y)
    return bayes_opt.best_params_

def perform_bayesian_optimization_with_adasyn(X, y):
    estimator = initialize_model(MODEL_TYPE)
    
    # Create a pipeline with SafeADASYN and the model/estimator
    pipeline = Pipeline([
        ('adasyn', SafeADASYN(random_state=RANDOM_SEED)), #Using custom safeADASYN class to handle cases where ADASYN is not applied
        ('model', estimator)
    ])
    
    # Update parameters to include the prefix 'model__' , Pipeline class will error without this prefix, 
    # necessary for accessing the parameters of the model inside the pipeline during the optimization process
    search_space_prefixed = {f'model__{key}': value for key, value in SEARCH_SPACE.items()}
    
    # Initialize the BayesSearchCV object with the pipeline and search space
    bayes_opt = BayesSearchCV(
        estimator=pipeline,  # The pipeline containing SafeADASYN and the model
        search_spaces=search_space_prefixed,  # The hyperparameter search space
        cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED),  # Cross-validation strategy
        n_iter=N_ITER_F,  # Number of iterations for the Bayesian optimization
        scoring='f1',  # Scoring metric for optimization
        n_jobs=-1,  # Use all available cores for computation, paralelization
        verbose=0,  # Verbosity level
        random_state=RANDOM_SEED  # Random seed for reproducibility
    )
    
    adasyn_results = [] # List to store ADASYN results for each cross-validation split

    # Perform cross-validation with ADASYN and collect class distribution data for each CV split, this is only for storing the results for verification
    for train_idx, test_idx in bayes_opt.cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Apply SafeADASYN to the training data
        ada = SafeADASYN(random_state=RANDOM_SEED)
        X_resampled, y_resampled = ada.fit_resample(X_train, y_train)
        
         # Collect class distribution information before and after resampling
        class_distribution_before = y_train.value_counts().to_dict()
        class_distribution_after = pd.Series(y_resampled).value_counts().to_dict()
        adasyn_success = class_distribution_before != class_distribution_after
        
        # Store the results of the ADASYN application for the current split
        adasyn_result = {
            'class_distribution_before': class_distribution_before,
            'class_distribution_after': class_distribution_after if adasyn_success else None,
            'adasyn_success': adasyn_success
        }
        
        adasyn_results.append(adasyn_result)
    
    bayes_opt.fit(X, y)
    
    # Remove the 'model__' prefix from the parameter keys, further parameter handling will error with this prefix
    best_params = {key.replace('model__', ''): value for key, value in bayes_opt.best_params_.items()}
    return bayes_opt, best_params, adasyn_results

def evaluate_model_on_test_set(model, X_test, y_test):
    # Predict the test set
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Accuracy and Balanced Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # Precision, Recall, and F1 for each class
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    
    # Weighted Precision, Recall, and F1
    weighted_precision = precision_score(y_test, y_pred, average='weighted')
    weighted_recall = recall_score(y_test, y_pred, average='weighted')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Unweighted (Simple) Average Precision, Recall, and F1
    average_precision = (precision_0 + precision_1) / 2
    average_recall = (recall_0 + recall_1) / 2
    average_f1 = (f1_0 + f1_1) / 2
    
    # Cohen's Kappa and Matthews Correlation Coefficient
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Confusion Matrix and Derived Metrics
    tpr = recall_1  # True Positive Rate (Sensitivity, Recall for class 1)
    fnr = 1 - recall_1  # False Negative Rate
    fpr = fp / (fp + tn)  # False Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate (Specificity)

    # Metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': {'0': precision_0, '1': precision_1, 'weighted-average': weighted_precision, 'unweighted-average': average_precision},
        'recall': {'0': recall_0, '1': recall_1, 'weighted-average': weighted_recall, 'unweighted-average': average_recall},
        'f1_score': {'0': f1_0, '1': f1_1, 'weighted-average': weighted_f1, 'unweighted-average': average_f1},
        'cohen_kappa': cohen_kappa,
        'mcc': mcc
    }

    # Confusion matrix and derived metrics
    confusion_metrics = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tpr': tpr,
        'fnr': fnr,
        'fpr': fpr,
        'tnr': tnr
    }

    return {
        'metrics': metrics,
        'confusion_matrix': confusion_metrics,
        'predictions': y_pred.tolist()
    }


def select_best_feature_count(train_data, model_params, seed):
    # Rank the features by their importance using the specified model and parameters
    feature_importance_df = rank_features_by_importance(train_data[SELECTED_FEATURES], train_data[TARGET], model_params)
    cv_results = {}

    def get_mean_cv_score(n):
        # Get the top 'n' most important features
        top_features = feature_importance_df.head(n)['feature'].tolist()
        # Initialize the model with the given parameters
        model = initialize_model(MODEL_TYPE, model_params) 
        # Perform cross-validation and compute the mean F1 score
        cv_scores = cross_val_score(model, train_data[top_features], train_data[TARGET],
                                    cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed),
                                    scoring='f1', n_jobs=-1)
        mean_cv_score = np.mean(cv_scores)  # Calculate the mean cross-validation score
        print(f"Number of features: {n}, CV Score: {mean_cv_score}")
        cv_results[n] = mean_cv_score # Store the CV score in the results dictionary
        return mean_cv_score
    
    # Determine the best number of features based on cross-validation scores
    best_n = max(N_OPTIONS, key=get_mean_cv_score)
    # Select the best features based on the best number of features
    best_features = feature_importance_df.head(best_n)['feature'].tolist()
    return best_features, feature_importance_df, cv_results

def multiple_runs_feature_selection(train_data):
    selected_features_per_run = []
    run_results = []
    
    # Perform feature selection across multiple runs
    for run in range(N_RUNS):
        seed = RANDOM_SEED + run  # Different seed for each run
        print(f"\nRunning feature selection, Run {run + 1}/{N_RUNS} with Seed {seed}")
        
        # Perform Bayesian optimization* to find the best parameters for the model, (with only one iteration its essentially the same as random oversampling)
        best_params = perform_bayesian_optimization(train_data[SELECTED_FEATURES], train_data[TARGET], seed)
        
        # Select best features and get their importances
        best_features, feature_importance_df, cv_results = select_best_feature_count(train_data, best_params, seed=seed)
        
        print(f"Run {run + 1} Selected Features: {best_features}")
        selected_features_per_run.append(best_features) # Store the selected features for the current run
        
        # Collect run results
        run_results.append({
            "cv_results": cv_results,
            "feature_importance": feature_importance_df.to_dict(orient='records'),
            "initial_params": best_params
        })

    # Flatten the list of selected features and count occurrences
    feature_occurrences = pd.Series([feature for features in selected_features_per_run for feature in features])
    
    # Calculate frequency as the number of runs each feature appears in divided by the number of runs
    feature_freq = feature_occurrences.value_counts() / N_RUNS
    print("Feature frequency across runs:\n", feature_freq)
    
    # Select final features that appear in at least FEATURE_THRESHOLD proportion of runs
    final_selected_features = feature_freq[feature_freq >= FEATURE_THRESHOLD].index.tolist()

    print(f"\nFinal Selected Features (appearing in at least {FEATURE_THRESHOLD*100}% of runs): {final_selected_features}")
    return final_selected_features, run_results, feature_freq

def run_nested_cv_for_year(train_data, test_data, year):
    # Extract sample IDs
    train_ids = train_data["id"]
    test_ids = test_data["id"]
    
     # Drop ID from train and test data
    train_data = train_data.drop(columns=["id"])
    test_data = test_data.drop(columns=["id"])
    
    #Get class distribution before
    print("Class distribution in the training set before ADASYN:")
    class_distribution_before = train_data[TARGET].value_counts().to_dict()
    check_class_distribution(train_data)

    # STEP 1: Feature selection
    final_selected_features, run_results, feature_freq = multiple_runs_feature_selection(train_data)

    # STEP 2: Final Bayesian Optimization, with ADASYN applied during cross-validation
    bayes_opt, best_params_final, adasyn_results = perform_bayesian_optimization_with_adasyn(train_data[final_selected_features], train_data[TARGET])

    # Extract information from step 2 bayesian optimization, for storing results
    final_bayes_results = {
        "iterations": [
            {
                "params": params,
                "mean_test_score": score,
                "std_test_score": std
            }
            for params, score, std in zip(bayes_opt.cv_results_["params"], bayes_opt.cv_results_["mean_test_score"], bayes_opt.cv_results_["std_test_score"])
        ]
    }

    # STEP 3: Final model training with ADASYN oversampling if possible
    class_distribution_after = None
    try:
        ada = ADASYN(sampling_strategy='auto', random_state=RANDOM_SEED)
        X_resampled, y_resampled = ada.fit_resample(train_data[final_selected_features], train_data[TARGET])
        class_distribution_after = pd.Series(y_resampled).value_counts().to_dict()
        print("Class distribution after applying ADASYN:")
        check_class_distribution(pd.DataFrame(y_resampled, columns=[TARGET]))
        
        # Initialize and train the final model using the resampled data
        final_model = initialize_model(MODEL_TYPE,best_params_final)
        final_model.fit(X_resampled, y_resampled)
    except ValueError as e:
        # Handle ADASYN errors and proceed without oversampling
        print(f"ADASYN Error: {e}")
        print("Proceeding with original data without oversampling.")
        
        # Initialize and train the final model WITHOUT resampling
        final_model = initialize_model(MODEL_TYPE,best_params_final)
        final_model.fit(train_data[final_selected_features], train_data[TARGET])
           
    # Get predicted probabilities for AUC calculation
    y_pred_proba = final_model.predict_proba(test_data[final_selected_features])[:, 1]
    
    # Evaluate the final model on the test set and obtain metrics
    test_metrics = evaluate_model_on_test_set(final_model, test_data[final_selected_features], test_data[TARGET])
    print(f"F1 Score on Test Set: {test_metrics['metrics']['f1_score']['weighted-average']}")
    print(f"Recall on Test Set: {test_metrics['metrics']['recall']['weighted-average']}")
    print(f"Precision on Test Set: {test_metrics['metrics']['precision']['weighted-average']}")
    
    # Return a dictionary containing various results and metrics from the run
    return {
        "test_metrics": test_metrics,
        "best_features": final_selected_features,
        "feature_importance": [run["feature_importance"] for run in run_results],
        "feature_frequency": feature_freq,
        "cv_results": [run["cv_results"] for run in run_results],
        "initial_params": [run["initial_params"] for run in run_results],
        "class_distribution_before": class_distribution_before,
        "class_distribution_after": class_distribution_after,
        "final_params": best_params_final,
        "final_bayes_results": final_bayes_results,
        "adasyn_results": adasyn_results,
        "final_model": final_model, 
        "y_pred_proba": y_pred_proba,
        "id": test_ids.tolist(), 
        "predictions": test_metrics['predictions'], 
        "year": year 
    }

def run_all_outer_loops(df_train_list, df_test_list):
    results = []
    all_samples = []

    # Determine the starting index based on the start_year arg
    start_index = 0
    if START_YEAR:
        start_index = YEARS.index(START_YEAR)

    for i in range(start_index, len(YEARS)):
        train_data = pd.concat(df_train_list[:i+1])  # Train on all previous years plus the current year
        test_data = df_test_list[i]  # Test on the current year
        year = YEARS[i]
        print(f"\nRunning outer loop for year {year}")

        # Run nested cross-validation and feature selection for the current year
        nested_cv_results = run_nested_cv_for_year(train_data, test_data, year)
        result = nested_cv_results.copy()  # Make a copy of the results
        y_pred_proba = nested_cv_results.pop("y_pred_proba")
        result["year"] = year
        results.append(result)
        
        # Collect test samples with predictions, actuals, and IDs
        year_predictions = pd.DataFrame({
            'id': nested_cv_results['id'],
            'predicted': nested_cv_results['predictions'],
            'actual': test_data[TARGET].tolist(),
            'year': year
        })

        # Save the predictions for the current year, appending to the file if it exists
        predictions_file_path = os.path.join(RESULTS_PATH, 'all_predictions.csv')
        if os.path.exists(predictions_file_path):
            year_predictions.to_csv(predictions_file_path, mode='a', header=False, index=False)
        else:
            year_predictions.to_csv(predictions_file_path, index=False)

        # Save outer fold results including data counts and AUC score
        save_outer_fold_results(year, result, train_data, test_data, y_pred_proba)

        # Collect test samples for the all_test_predictions CSV
        predictions = result['test_metrics']['predictions']
        actuals = test_data[TARGET].tolist()
        year_samples = pd.DataFrame({'year': [year] * len(predictions), 'predicted': predictions, 'actual': actuals})
        all_samples.append(year_samples)
        
        save_roc_curve_and_auc(actuals, y_pred_proba, year)
        
    # Save overall results after all outer loops
    save_overall_results()

    return results

def main():

    # Loading dataset
    file_path = os.path.join(cdir, FILE_NAME)
    df = load_data(file_path)

    # Preprocessing
    X, y, df = preprocess_data(df)
 
    # Split data into train and test sets by year for (Rolling Forecast Origin) nested cross validation
    df_train_list, df_test_list = split_train_test(df)

    # Save initial configuration
    initial_config = {
        "Hyperparameter Search Space": SEARCH_SPACE,
        "Random Seed": RANDOM_SEED,
        "Model Type": MODEL_TYPE,
        "Selected Features": SELECTED_FEATURES,
        "Number of iterations for final bayes hyperparameter optimization": N_ITER_F,
        "Feature count treshold": FEATURE_THRESHOLD,
        "Number features that can be selected": N_OPTIONS
    }; save_initial_config(initial_config)

    # Run nested cross-validation for all outer loops
    results = run_all_outer_loops(df_train_list, df_test_list)
    
    print("Finished running, results saved at: ", RESULTS_PATH)
    

if __name__ == "__main__":
    
    # Changing WORKING DIRECTORY
    os.chdir("/home/jovyan/work/Typhoon_IBF_Rice_Damage_Model/")
    cdir = os.getcwd()
    
    # Years for nested-cv / rolling-forecast-origin
    YEARS = [2016, 2017, 2018, 2019, 2020]

    # Argument parsing
    parser = argparse.ArgumentParser(description="Run nested cross-validation with specific configurations.")
    parser.add_argument("--start_year", type=int, choices=YEARS, default=YEARS[0], help=f"The starting year for the outer fold. Must be one of {YEARS}. Default is {YEARS[0]}.")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed for reproducibility. Default is 1.")
    parser.add_argument("--model_type", type=str, choices=["xgboost", "random_forest"], default="xgboost", help="Type of model to use. Default is xgboost.")
    parser.add_argument("--variant", type=str, choices=["lf", "extent"], default="extent", help="Variant of the model. Use 'lf' for landfall variant and 'extent' for extent-based variant. Default is extent.")
    args = parser.parse_args()
    
    # Setting global variables
    global START_YEAR, RANDOM_SEED, MODEL_TYPE, RESULTS_PATH, SEARCH_SPACE, SELECTED_FEATURES, FILE_NAME, TARGET, N_RUNS, FEATURE_THRESHOLD, N_ITER_I, N_ITER_F, XGB_OBJECTIVE
    
    START_YEAR = args.start_year
    RANDOM_SEED = args.random_seed
    MODEL_TYPE = args.model_type
    variant = args.variant
    
    # Setting seed for reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Number of features to consider in feature selection
    N_OPTIONS = [2, 3, 4, 5, 6]
    
    # Number of cross-validation splits
    CV_SPLITS = 5
    
    # Constructing experiment name and results path
    EXPERIMENT_NAME = f"{MODEL_TYPE}_binary_{RANDOM_SEED}"
    if variant == "lf":
        EXPERIMENT_NAME += "_lf"
    
    if MODEL_TYPE == "xgboost":
        XGB_OBJECTIVE = "binary:hinge"
        SEARCH_SPACE = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 1, 'log-uniform'),
            'reg_lambda': Real(1e-5, 10, 'log-uniform'),
            'colsample_bytree': Real(0.3, 1.0),
            'gamma': Real(0.1, 5)
        }
        N_RUNS = 100 # 100 # Number of iterations for feature selection
        FEATURE_THRESHOLD = 0.9  # Feature selection threshold (90% appearance)
        N_ITER_I, N_ITER_F = 1, 200 # 1 , 200 # Iterations for initial and final optimization
    elif MODEL_TYPE == "random_forest":
        SEARCH_SPACE = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(10, 50),
            'min_samples_leaf': Integer(5, 50)
        }
        N_RUNS = 100
        FEATURE_THRESHOLD = 0.9
        N_ITER_I, N_ITER_F = 1, 150
    
    # Setting results path based on the model type
    RESULTS_PATH = f'IBF_typhoon_model/models/results/only_overlapping_additional_samples/{MODEL_TYPE}_binary/{EXPERIMENT_NAME}'
    
    # Check if the results path exists to avoid overwriting
    if os.path.exists(RESULTS_PATH) and args.start_year == YEARS[0]:
        print(f"Warning: Results path {RESULTS_PATH} already exists. Exiting to avoid overwriting results.")
        sys.exit(1)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Setting selected features and file path based on variant
    if variant == "lf":
        SELECTED_FEATURES = [
            "mean_elevation_m", "ruggedness_stdev", "area_km2", "poverty_perc", "with_coast",
            "coast_length", "perimeter", "glat", "glon", "coast_peri_ratio",
            'lf_pre_max_6h', 'lf_during_max_6h_intensity', 'lf_post_max_24h', "dis_track_min", "vmax"
        ]
        FILE_NAME = "IBF_typhoon_model/data/restricted_data/combined_input_data/input_data_05_overlap.xlsx"
    else:
        SELECTED_FEATURES = [
            "mean_elevation_m", "ruggedness_stdev", "area_km2", "poverty_perc", "with_coast",
            "coast_length", "perimeter", "glat", "glon", "coast_peri_ratio",
            'pre_max_6h', 'during_total_rainfall', 'post_max_6h', "dis_track_min", "vmax"
        ]
        FILE_NAME = "IBF_typhoon_model/data/restricted_data/combined_input_data/input_data_05_overlap.xlsx"
    
    TARGET = "class_value_binary"
    
    # Call the main function to execute cross-validation and save results
    main()
    

# python binary_pipeline.py --random_seed 9001 --model_type random_forest --variant lf --start_year 2018