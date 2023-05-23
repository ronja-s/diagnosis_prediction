# %%
import os
import random
import warnings

import numpy as np
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
    make_scorer,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import DataLoader
from performance_evaluator import PerformanceEvaluator
from pipeline_builder import PipelineBuilder

warnings.filterwarnings("ignore")


# set random number generator seed for reproducibilty
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


set_seed(31415)

# global variables:
results_dir = "./results/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

data_path = "./data/test_cases.json"
icd10_chapters_definition_path = "./data/icd10_chapters_definition.csv"

performance_metric = accuracy_score
cross_validation_params = {
    "cv": 5,
    "return_train_score": True,
    "scoring": make_scorer(performance_metric),
}

# pipeline parameter ranges:
model_options = [
    DummyClassifier(),  # strategy="prior"
    # KNeighborsClassifier(),
    # DecisionTreeClassifier(),
    # LogisticRegression(),
    LinearSVC(),
    # SVC(),
    # MLPClassifier(),
    LinearDiscriminantAnalysis(solver="lsqr"),
    # RandomForestClassifier(),
    # ExtraTreesClassifier(),
    # PassiveAggressiveClassifier(),
    # Perceptron(),
    # GradientBoostingClassifier(),
]
dim_reduction_algorithm_options = [
    None,
    PCA,
    # TruncatedSVD,
    # Isomap,
    # LocallyLinearEmbedding,
]
n_dimensions_options = [10, 50, 100, 200, None]
count_evidence_options = [False, True]
include_absent_evidence_options = [False, True]
n_most_frequent_options = [None]

# classifiers' hyperparameters for tuning overfitting:
# note: try to minimize overfitting for the best models/pipes:
hyperparameters = {
    LogisticRegression: {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]},
    LinearSVC: {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]},
    PassiveAggressiveClassifier: {"C": [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]},
    RandomForestClassifier: {"max_leaf_nodes": [None, 200, 100, 50, 30, 20, 10, 5, 1]},
    ExtraTreesClassifier: {"max_leaf_nodes": [None, 200, 100, 50, 30, 20, 10, 5, 1]},
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
    data_path=data_path,
    icd10_chapters_definition_path=icd10_chapters_definition_path,
    test_size=None,
)
print("X:", X)
print("y:", y)

eval = PerformanceEvaluator(
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

# %%
print("Train and evaluate chosen pipelines.")
eval.train_and_evaluate(verbose=True)

# %%
print(
    "For each model: Try to fix overfitting by varying hyperparameter(s) that reflect the model complexity."
)
eval.perform_gridsearch(hyperparameters=hyperparameters, with_plots=True, verbose=True)

# %% make predictions with best performing pipeline:
eval.get_best_predictor()

# %% plot performance for other paramaters of the pipeline:
eval.plot_performance(model_class=LinearSVC, parameters=["n_dimensions"])

# %%
