# %%
import os
import warnings

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             make_scorer, top_k_accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
import shap

from helper_functions import (check_preprocessing, determine_best_parameters,
                              evaluate_pipelines, load_data,
                              perform_gridsearch, plot_grid_search_results)

warnings.filterwarnings("ignore")
# set random number generator seed for reproducibilty
np.random.seed(0)

# global variables:
results_dir = "./test_results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

data_path = "./data/test_cases.json"
icd10_ranges_data_path = "./data/icd10_codes.csv"
accuracies_file_path = results_dir + "accuracies.csv"
best_models_file_path = results_dir + "best_models.csv"
grid_search_file_path = results_dir + "grid_search.csv"
best_hyperparameters_file_path = results_dir + "best_hyperparams.csv"

cross_validation_params = {
    "cv": 5,
    "return_train_score": True,
    "scoring": {
        "accuracy": make_scorer(accuracy_score),
    },
}

# pipeline parameter ranges:
models = [
    DummyClassifier(
        strategy="prior"
    ),
    KNeighborsClassifier(),
    DecisionTreeClassifier(max_depth=None),
    LogisticRegression(),
    LinearSVC(),
    SVC(),
    MLPClassifier(),
    LinearDiscriminantAnalysis(solver="lsqr"),
    RandomForestClassifier(),
    # ExtraTreesClassifier(),
    # PassiveAggressiveClassifier(),
    # Perceptron(),  
    # GradientBoostingClassifier(),
]
dim_reduction_algorithm_values = [
    None,
    PCA,
    TruncatedSVD,
    Isomap,
    LocallyLinearEmbedding,
]
n_dimensions_values = [10, 50, 100, 200, 300, 400, None]
count_evidence_values = [False, True]
include_absent_evidence_values = [False, True]
n_most_frequent_values = [None]

# classifiers' hyperparameters for tuning overfitting:
# note: try to minimize overfitting for the best models/pipes:
hyperparameters = {
    "LogisticRegression": {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]},
    "LinearSVC": {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]}, 
    # "PassiveAggressiveClassifier": {
    #     "C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
    # },
    "RandomForestClassifier": {
        "max_leaf_nodes": [None, 200, 100, 50, 30, 20, 10, 5, 1]
    },
    # "ExtraTreesClassifier": {
    #     "max_leaf_nodes": [None, 200, 100, 50, 30, 20, 10, 5, 1]
    # }, 
    "MLPClassifier": {
        "alpha": [
            0.0001,
            0.005,
            0.001,
            0.005,
            0.01,
        ],
        # "hidden_layer_sizes": [(200,), (100,), (50,), (20,), (10,), (5,)],
    },
    "LinearDiscriminantAnalysis": {
        "shrinkage": [None, 0.0, 0.3, 0.6, 1.0]
    },
}

# load data:
print("Load the data.")
X, y = load_data(data_path=data_path, icd10_ranges_data_path=icd10_ranges_data_path)

# %% check preprocessing:
print("Check the preprocessing.")
check_preprocessing(
    X=X,
    model=models[0],
    dim_reduction_algorithm=None,
    n_dimensions=None,
    count_evidence=False,
    include_absent_evidence=True,
    n_most_frequent=None,
)

# %% evaluate pipelines (with cross validation):
print("Train and evaluate different chosen models and parameters.")
evaluate_pipelines(
    X=X,
    y=y,
    dim_reduction_algorithm_values=dim_reduction_algorithm_values,
    n_dimensions_values=n_dimensions_values,
    count_evidence_values=count_evidence_values,
    include_absent_evidence_values=include_absent_evidence_values,
    n_most_frequent_values=n_most_frequent_values,
    models=models,
    cross_validation_params=cross_validation_params,
    result_file_path=accuracies_file_path,
)

# %% determine best pipelines/models:
print("For each model: Find the best performing parameters.")
determine_best_parameters(
    input_file_path=accuracies_file_path,
    accuracy_column="test_accuracy",
    result_file_path=best_models_file_path,
)

# %% perform grid search for models hyperparameters in order to try minimizing overfitting:
print("For each model: Try to fix overfitting by varying parameter which reflects the model complexity.")
perform_gridsearch(
    X=X,
    y=y,
    best_models_file_path=best_models_file_path,
    cross_validation_params=cross_validation_params,
    hyperparameters=hyperparameters,
    result_file_path=grid_search_file_path,
)

# %% determine best hyperparameters:
print("For each model, find the best value for this parameter which reflects the model complexity.")
determine_best_parameters(
    input_file_path=grid_search_file_path,
    accuracy_column="mean_test_accuracy",
    result_file_path=best_hyperparameters_file_path,
)

# %% plot grid search results:
print("Plot the results.")
plot_grid_search_results(
    grid_search_file_path=grid_search_file_path, result_dir=results_dir
)

# %%
# analyze feature importance:
# import shap
# explainer = shap.Explainer(pipes[best_model_name])
# shap_values = explainer(X)
# shap.plots.beeswarm(shap_values)

# %%[markdown]
# Further doing (ideas):
# - include whether an evidence is initial -> one hot encoding (1 column for each evidence)

