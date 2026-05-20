# Models

This folder contains the **model pipelines and results**. The models include **Random Forest (RF)** and **XGBoost (XGB)** implementations for both **regression** and **binary classification** tasks.

## Modelling approach

All models use the same general pipeline, with differences mainly in the evaluation metrics and the use of **ADASYN oversampling**, which is applied only for binary classification.

Key elements of the modelling pipeline include:

- **Anchored walk-forward nested cross-validation**  
  The dataset is split by year (2016–2020). Each year acts as the test set while all preceding years are used for training. Within each outer split, an inner loop performs feature selection, hyperparameter optimisation, training, and evaluation.

- **Feature selection**  
  Feature importance is evaluated across 100 runs with randomly initialised hyperparameters. The best-performing number of features (between 2 and 6) is determined using cross-validation, and features appearing in **≥90% of runs** are retained.

- **Bayesian hyperparameter optimisation**  
  Hyperparameters are tuned using Bayesian optimisation (150 iterations for RF, 200 for XGB).

- **Controlled stochasticity**  
  All pipelines are executed with **five different random seeds**. Results are averaged across seeds to ensure robustness and enable statistical comparison of rainfall feature extraction approaches.

### Evaluation metrics

Regression models use:

- MAE (Mean Absolute Error) – optimisation metric
- RMSE (Root Mean Squared Error)
- MBD (Mean Bias Deviation)

Binary classification models use:

- **F1₁** (F1 score for the positive class) – optimisation metric  
- **Macro F1** – mean of F1 scores for positive and negative classes


### Two-step model

A two-step modelling approach was also evaluated:

1. A **binary classifier** predicts whether losses exceed **30%**.
2. For predicted positives, a **regression model** estimates the magnitude of the losses.

This setup reflects an operational scenario where the binary model identifies municipalities requiring early action, while the regression model informs the allocation of resources.

The two-step approach was tested only with the **best-performing regression model** (RF with landfall rainfall features).

## Repository structure

### Pipelines

- `binary_pipeline.py`  
  Pipeline for binary classification models.

- `regression_pipeline.py`  
  Pipeline for regression models.

Both pipelines implement the full modelling workflow, including feature selection, Bayesian optimisation, model training, testing, and saving of results.

### Results

All model outputs are stored in the **`results_per_model`** folder, inside the **`results`** folder.

The contents are:

- **Model-specific result folders**  
  Each model (RF regression, RF binary, XGB regression, XGB binary, and two step model) has its own directory.

- **Seed-level outputs**  
  Results are saved for each random seed and rainfall feature method:
  - `lf` suffix → landfall rainfall features  
  - no suffix → spatial extent rainfall features

- **Stored outputs**
  - CSV files containing model predictions
  - JSON files containing evaluation results for each outer cross-validation fold
  - JSON file summarising overall results

### Results notebook

The **`results`** folder also contains a notebook that provides tools to explore the outputs:

- **Interactive widget** to compare models, metrics, and rainfall feature variants
- **Feature selection summary table** showing which features were retained across models

Further implementation details can be found directly in the pipeline scripts.