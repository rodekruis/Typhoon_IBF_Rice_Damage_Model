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


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse


""" Saving functions """

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

def save_outer_fold_results(year, results, train_data, test_data):
    def convert_np_int64(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Calculate train and test counts
    train_total_count = train_data.shape[0]
    test_total_count = test_data.shape[0]

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
        "data_counts": {
            "train_total_count": train_total_count,
            "test_total_count": test_total_count
        },
        "final_params": results['final_params'],
        "test_metrics": results['test_metrics']['metrics']
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
    total_test_size = 0
    outer_fold_results = []  # Collecting results for all outer folds
    
    # Initialize dictionaries for weighted and normal averages
    overall_metrics = {
        "weighted_average": defaultdict(list),
        "normal_average": defaultdict(list)
    }
    
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
        })
        
        # Count selected features
        for feature in result['best_features']:
            selected_features[feature] += 1
        
        # Collect best hyperparameters
        for param, value in result['final_params'].items():
            final_hyperparameters[param].append(value)
        
        # Collect metrics (assuming 4 metrics like RMSE, MAE, R2, MSE)
        for metric, value in result['test_metrics'].items():
            overall_metrics["weighted_average"][metric].append((value, test_size))
            overall_metrics["normal_average"][metric].append(value)
    
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
        }
    }
    
    for param, values in final_hyperparameters.items():
        aggregated_results["final_hyperparameters"][param] = {
            "selected_values": values,
            "range": [min(values), max(values)],
            "mean": np.mean(values),
            "std": np.std(values)
        }
    
    # Calculate and store weighted and normal averages
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

def two_step_model_filter(train_data, test_data):
    """ Filter train samples based on perc_loss > 0.3 and filter test samples based on CSV predictions. """
    # Load the CSV file containing predictions
    filtered_df = pd.read_csv(TWO_STEP_MODEL_CSV)
    
    # Filter train samples based on perc_loss > 0.3
    filtered_train_data = train_data[train_data[TARGET] > 0.3]
    
    # Extract test sample IDs where predicted is 1
    test_ids = filtered_df[filtered_df['predicted'] == 1]['id']
    # Filter the test set based on the filtered_df
    filtered_test_data = test_data[test_data['id'].isin(test_ids)]
    
    return filtered_train_data, filtered_test_data

def load_data(file_path):
    return pd.read_excel(file_path, engine="openpyxl")

def preprocess_data(df):
    df['id'] = df.index  # Adding unique identifier to all samples
    df_regression = df[df[TARGET].notnull()]
    X = df_regression[SELECTED_FEATURES + ["id"]]
    y = df_regression[TARGET].astype(float)
    return X, y, df_regression

def split_train_test(df):
    df_train_list = [df[df["year"] < year] for year in YEARS]
    df_test_list = [df[df["year"] == year] for year in YEARS]
    return df_train_list, df_test_list

    
def initialize_model(model_type, params=None):
    if model_type == "random_forest":
        return RandomForestRegressor(**(params or {}), random_state=RANDOM_SEED)
    elif model_type == "xgboost":
        return XGBRegressor(objective=XGB_OBJECTIVE, **(params or {}), random_state=RANDOM_SEED)
    
# Feature ranking and selection functions
def rank_features_by_importance(X, y, model_params):
    model = initialize_model(MODEL_TYPE, model_params) 
    model.fit(X, y) # Fit the model to the provided data (features X and target y)
    importances = model.feature_importances_ # Retrieve the feature importances from the fitted model
    # Create a df with feature names and their corresponding importances, then sort by descending order of importance, return df
    return pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance', ascending=False)

''' Model training and evaluation functions'''

def perform_bayesian_optimization(X, y, seed, stage):
    estimator = initialize_model(MODEL_TYPE) 
    
    if stage == 'initial':
        n_iter = N_ITER_I
    elif stage == 'final':
        n_iter = N_ITER_F
        
    # Initialize the BayesSearchCV object with the specified model and search space
    bayes_opt = BayesSearchCV(
        estimator=estimator,  # The machine learning estimator/model
        search_spaces=SEARCH_SPACE,  # The hyperparameter search space
        cv=KFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed),  # Cross-validation strategy
        n_iter=n_iter,  # Number of iterations for the Bayesian optimization
        scoring= SCORING,  # Scoring metric for optimization
        n_jobs=-1,  # Use all available cores for parallel computation
        verbose=0,  # Verbosity level
        random_state=seed  # Random seed for reproducibility, NOTE: using RANDOM_SEED for final optimization, and different seeds for the feature selection loop
    )
    
    bayes_opt.fit(X, y)
    return bayes_opt.best_params_ , bayes_opt

def evaluate_model_on_test_set(model, X_test, y_test):
    def mean_bias_deviation(y_true, y_pred):
        """Calculate Mean Bias Deviation (MBD)."""
        return np.mean(y_pred - y_true)

    # Predict the test set
    y_pred = model.predict(X_test)

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mbd = mean_bias_deviation(y_test, y_pred)

    # Metrics dictionary
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mbd': mbd
    }

    return {
        'metrics': metrics,
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
        # Perform cross-validation
        cv_scores = cross_val_score(model, train_data[top_features], train_data[TARGET],
                                    cv=KFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed),
                                    scoring=SCORING, n_jobs=-1)
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
        best_params, bayes_opt = perform_bayesian_optimization(train_data[SELECTED_FEATURES], train_data[TARGET], seed, 'initial')
        
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

    # STEP 1: Feature selection
    final_selected_features, run_results, feature_freq = multiple_runs_feature_selection(train_data)

    # STEP 2: Final Bayesian Optimization
    best_params_final, bayes_opt = perform_bayesian_optimization(train_data[final_selected_features], train_data[TARGET], RANDOM_SEED, 'final')

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

    # STEP 3: Final model training

    # Initialize and train the final model
    final_model = initialize_model(MODEL_TYPE,best_params_final)
    final_model.fit(train_data[final_selected_features], train_data[TARGET])
    
    # Evaluate the final model on the test set and obtain metrics
    test_metrics = evaluate_model_on_test_set(final_model, test_data[final_selected_features], test_data[TARGET])
    print(f"RMSE  on Test Set: {test_metrics['metrics']['rmse']}")
    print(f"MAE on Test Set: {test_metrics['metrics']['mae']}")
   
    # Return a dictionary containing various results and metrics from the run
    return {
        "test_metrics": test_metrics,
        "best_features": final_selected_features,
        "feature_importance": [run["feature_importance"] for run in run_results],
        "feature_frequency": feature_freq,
        "cv_results": [run["cv_results"] for run in run_results],
        "initial_params": [run["initial_params"] for run in run_results],
        "final_params": best_params_final,
        "final_bayes_results": final_bayes_results,
        "final_model": final_model, 
        "id": test_ids.tolist(), 
        "predictions": test_metrics['predictions'], 
        "year": year 
    }

def run_all_outer_loops(df_train_list, df_test_list):
    results = []
    all_samples = []
    all_predictions = [] # To collect predictions across years

    for i, (train_data, test_data) in enumerate(zip(df_train_list, df_test_list)):
        year = 2016 + i
        print(f"\nRunning outer loop for year {year}")

        # Run nested cross-validation and feature selection for the current year
        nested_cv_results = run_nested_cv_for_year(train_data, test_data, year)
        result = nested_cv_results.copy()  # Make a copy of the results
        result["year"] = year
        results.append(result)
        
        # Collect test samples with predictions, actuals, and IDs
        year_predictions = pd.DataFrame({
            'id': nested_cv_results['id'],
            'predicted': nested_cv_results['predictions'],
            'actual': test_data[TARGET].tolist(),
            'year': year
        })
        
        all_predictions.append(year_predictions)  # Append yearly prediction

        # Save outer fold results
        save_outer_fold_results(year, result, train_data, test_data)

        # Collect test samples for the all_test_predictions CSV
        predictions = result['test_metrics']['predictions']
        actuals = test_data[TARGET].tolist()
        year_samples = pd.DataFrame({'year': [year] * len(predictions), 'predicted': predictions, 'actual': actuals})
        all_samples.append(year_samples)
        
        
    # Aggregate all predictions into a single DataFrame
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Save the aggregated predictions
    all_predictions_df.to_csv(os.path.join(RESULTS_PATH, 'all_predictions.csv'), index=False)

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
    
    # Apply additional filtering if TWO_STEP_MODEL is enabled
    if TWO_STEP_MODEL:
        for i in range(len(df_train_list)):
            df_train_list[i], df_test_list[i] = two_step_model_filter(df_train_list[i], df_test_list[i])

    # Save initial configuration
    initial_config = {
        "Hyperparameter Search Space": SEARCH_SPACE,
        "Random Seed": RANDOM_SEED,
        "Model Type": MODEL_TYPE,
        "Selected Features": SELECTED_FEATURES,
        "Number of iterations for final bayes hyperparameter optimization": N_ITER_F,
        "Feature count treshold": FEATURE_THRESHOLD,
        "Number features that can be selected": N_OPTIONS,
        "Is two step model?": TWO_STEP_MODEL
    }; save_initial_config(initial_config)

    # Run nested cross-validation for all outer loops
    results = run_all_outer_loops(df_train_list, df_test_list)
    
    print("Finished running, results saved at: ", RESULTS_PATH)

if __name__ == "__main__":
    
    # Changing WORKING DIRECTORY
    os.chdir("/home/jovyan/work/Typhoon_IBF_Rice_Damage_Model/")
    cdir = os.getcwd()
     
    #Years for nested-cv / rolling-forecast-origin
    YEARS = [2016, 2017, 2018, 2019, 2020]
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run nested cross-validation with specific configurations.")
    parser.add_argument("--start_year", type=int, choices=YEARS, default=YEARS[0], help=f"The starting year for the outer fold. Must be one of {YEARS}. Default is {YEARS[0]}.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility. Default is 42.")
    parser.add_argument("--model_type", type=str, choices=["xgboost", "random_forest"], default="random_forest", help="Type of model to use. Default is random_forest.")
    parser.add_argument("--variant", type=str, choices=["lf", "extent"], default="extent", help="Variant of the model. Use 'lf' for landfall variant and 'extent' for extent-based variant. Default is extent.")
    parser.add_argument("--two_step_model", action="store_true", help="Flag to enable two-step model. If set, the model will be trained with two steps.")
    args = parser.parse_args()
    
    # Setting global variables
    global START_YEAR, RANDOM_SEED, MODEL_TYPE, RESULTS_PATH, SEARCH_SPACE, SELECTED_FEATURES, FILE_NAME, TARGET, N_RUNS, FEATURE_THRESHOLD, N_ITER_I, N_ITER_F, XGB_OBJECTIVE, TWO_STEP_MODEL, TWO_STEP_MODEL_CSV
    
    START_YEAR = args.start_year
    RANDOM_SEED = args.random_seed
    MODEL_TYPE = args.model_type
    variant = args.variant
    TWO_STEP_MODEL = args.two_step_model

    # Setting seed for reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Number of features to consider in feature selection
    N_OPTIONS = [2, 3, 4, 5, 6]
    
    # Number of cross-validation splits
    CV_SPLITS = 5
    
    # Scoring for cross-validation and Bayesian optimization
    SCORING = "neg_mean_absolute_error"
    
    # Constructing experiment name and results path
    EXPERIMENT_NAME = f"{MODEL_TYPE}_regression_{RANDOM_SEED}"
    if variant == "lf":
        EXPERIMENT_NAME += "_lf"
    
    # Set the base results path
    RESULTS_PATH = f'IBF_typhoon_model/models/results/only_overlapping_additional_samples/{MODEL_TYPE}_regression/{EXPERIMENT_NAME}'

    if TWO_STEP_MODEL:
        # If two-step model is enabled, modify the results path
        RESULTS_PATH = f'/home/jovyan/work/Typhoon_IBF_Rice_Damage_Model/IBF_typhoon_model/models/results/only_overlapping_additional_samples/two_step_model/{EXPERIMENT_NAME}'
        TWO_STEP_MODEL_CSV = f"IBF_typhoon_model/models/results/only_overlapping_additional_samples/random_forest_binary/random_forest_binary_{RANDOM_SEED}_lf/all_predictions.csv"
        
        # Check if the CSV path exists
        if not os.path.exists(TWO_STEP_MODEL_CSV):
            print(f"Error: The specified two-step model CSV path does not exist: {TWO_STEP_MODEL_CSV}")
            sys.exit(1)

    # Check if the results path exists to avoid overwriting
    if os.path.exists(RESULTS_PATH) and args.start_year == YEARS[0]:
        print(f"Warning: Results path {RESULTS_PATH} already exists. Exiting to avoid overwriting results.")
        sys.exit(1)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Setting hyperparameter search space and other configurations based on the chosen model type
    if MODEL_TYPE == "random_forest":
        SEARCH_SPACE = {
            'n_estimators': Integer(100, 500),  
            'max_depth': Integer(10, 30),
            'min_samples_split': Integer(10, 50),
            'min_samples_leaf': Integer(5, 50)
        }
        N_RUNS = 100
        FEATURE_THRESHOLD = 0.9
        N_ITER_I, N_ITER_F = 1, 150

    elif MODEL_TYPE == "xgboost":
        XGB_OBJECTIVE = "reg:pseudohubererror"
        SEARCH_SPACE = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 1, 'log-uniform'),
            'reg_lambda': Real(1e-5, 10, 'log-uniform'),
            'colsample_bytree': Real(0.3, 1.0),
            'gamma': Real(0.1, 5)
        }
        N_RUNS = 100
        FEATURE_THRESHOLD = 0.9
        N_ITER_I, N_ITER_F = 1, 200
    
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
    
    TARGET = "perc_loss"
    
    main()



