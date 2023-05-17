# %%
import os
import warnings

import numpy as np
import shap
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    make_scorer,
    top_k_accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import DataLoader
from evaluator import Evaluator
from pipeline_builder import PipelineBuilder

warnings.filterwarnings("ignore")
# set random number generator seed for reproducibilty
np.random.seed(0)

# global variables:
results_dir = "./test_results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

data_path = "./data/test_cases.json"
icd10_chapters_definition_path = "./data/icd10_chapters_definition.csv"

cross_validation_params = {
    "cv": 5,
    "return_train_score": True,
    "scoring": {
        "accuracy": make_scorer(accuracy_score),
    },
}

# pipeline parameter ranges:
model_options = [
    DummyClassifier(),  # strategy="prior"
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    LogisticRegression(),
    LinearSVC(),
    SVC(),
    MLPClassifier(),
    LinearDiscriminantAnalysis(solver="lsqr"),
    RandomForestClassifier(),
    ExtraTreesClassifier(),
    PassiveAggressiveClassifier(),
    Perceptron(),
    GradientBoostingClassifier(),
]
dim_reduction_algorithm_options = [
    None,
    PCA,
    TruncatedSVD,
    Isomap,
    LocallyLinearEmbedding,
]
n_dimensions_options = [10, 50, 100, 200, 300, 400, None]
count_evidence_options = [False, True]
include_absent_evidence_options = [False, True]
n_most_frequent_options = [None]

# classifiers' hyperparameters for tuning overfitting:
# note: try to minimize overfitting for the best models/pipes:
hyperparameters = {
    LogisticRegression: {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]},
    LinearSVC: {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]},
    "PassiveAggressiveClassifier": {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]},
    RandomForestClassifier: {"max_leaf_nodes": [None, 200, 100, 50, 30, 20, 10, 5, 1]},
    "ExtraTreesClassifier": {"max_leaf_nodes": [None, 200, 100, 50, 30, 20, 10, 5, 1]},
    MLPClassifier: {
        "alpha": [
            0.0001,
            0.005,
            0.001,
            0.005,
            0.01,
        ],
        "hidden_layer_sizes": [(200,), (100,), (50,), (20,), (10,), (5,)],
    },
    LinearDiscriminantAnalysis: {"shrinkage": [None, 0.0, 0.3, 0.6, 1.0]},
}

print("Load the data.")
X, y = DataLoader(multi_label=False).load(
    data_path=data_path, icd10_chapters_definition_path=icd10_chapters_definition_path
)
print("X:", X)
print("y:", y)

# %%
print("Test preprocessing steps.")
PipelineBuilder(
    model=model_options[1],
    dim_reduction_algorithm=dim_reduction_algorithm_options[1],
    n_dimensions=n_dimensions_options[1],
    count_evidence=count_evidence_options[1],
    include_absent_evidence=include_absent_evidence_options[1],
    n_most_frequent=n_most_frequent_options[0],
).print_preprocessing_steps(X=X)

# %%
eval = Evaluator(
    X=X,
    y=y,
    model_options=model_options,
    dim_reduction_algorithm_options=dim_reduction_algorithm_options,
    n_dimensions_options=n_dimensions_options,
    count_evidence_options=count_evidence_options,
    include_absent_evidence_options=include_absent_evidence_options,
    n_most_frequent_options=n_most_frequent_options,
    cross_validation_params=cross_validation_params,
    results_directory=results_dir,
)

# %% evaluate pipelines (with cross validation):
print("Train and evaluate chosen pipelines.")
eval.train_and_evaluate(verbose=True)

# %% determine best pipelines/models:
print("For each model: Find the best performing parameters.")
eval.get_best_parameters()

# %% perform grid search for models hyperparameters in order to try minimizing overfitting:
print(
    "For each model: Try to fix overfitting by varying a parameter which reflects the model complexity."
)
eval.perform_gridsearch(hyperparameters=hyperparameters, with_plots=True, verbose=True)

# %% determine best hyperparameters:
print(
    "For each model: Find the best performing parameters after performing gridsearch."
)
eval.get_best_parameters()

# %% plot performance for other paramaters of the pipeline:
eval.plot_performance(
    model_class=LinearDiscriminantAnalysis, parameters=["n_dimensions"]
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
