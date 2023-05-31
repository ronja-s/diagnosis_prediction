# %%
import os
import warnings

from data_loader import DataLoader
from global_variables import (
    ICD10_CHAPTERS_DEFINITION_PATH,
    RESULTS_DIR,
    SEED,
    TEST_CASES_PATH,
)
from helper_functions import set_seed
from performance_evaluator import PerformanceEvaluator
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
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

set_seed(SEED)

warnings.filterwarnings("ignore")  # to reduce noise during training

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

performance_metric = accuracy_score
cross_validation_splitting = 5

model_options = [
    DummyClassifier(),  # baseline model (predicting the most frequent target)
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    LogisticRegression(),
    LinearSVC(),  # linear kernel
    SVC(),  # rbf kernel
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
n_dimensions_options = [10, 50, 100, 250, None]
count_evidence_options = [False, True]
include_absent_evidence_options = [False, True]
n_most_frequent_options = [None]

hyperparameters = {
    KNeighborsClassifier: {"n_neighbors": [2, 5, 10, 20, 50]},
    LogisticRegression: {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    LinearSVC: {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    SVC: {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    PassiveAggressiveClassifier: {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    Perceptron: {
        "penalty": ["l2"],
        "alpha": [0.00001, 0.0001, 0.001, 0.01],
    },
    MLPClassifier: {
        "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "hidden_layer_sizes": [(5,), (50,), (200,), (500,)],
    },
    LinearDiscriminantAnalysis: {"shrinkage": [None, 0.0, 0.3, 0.6, 1.0]},
    DecisionTreeClassifier: {"max_depth": [1, 10, 50, 200, 500, None]},
    ExtraTreesClassifier: {
        "max_depth": [1, 10, 50, 200, 500, None],
        "n_estimators": [10, 100, 500, 1000],
    },
    RandomForestClassifier: {
        "max_depth": [1, 10, 50, 200, 500, None],
        "n_estimators": [10, 100, 500, 1000],
    },
    GradientBoostingClassifier: {
        "max_depth": [1, 10, 50, 200, 500, None],
        "n_estimators": [10, 100, 500, 1000],
    },
}

print("Load the data.")
X, y = DataLoader(multi_label=False).load(
    data_path=TEST_CASES_PATH,
    icd10_chapters_definition_path=ICD10_CHAPTERS_DEFINITION_PATH,
    test_size=None,
)
print("X:", X)
print("y:", y)

performance_evaluator = PerformanceEvaluator(
    X=X,
    y=y,
    performance_metric=performance_metric,
    cross_validation_splitting=cross_validation_splitting,
    results_directory=RESULTS_DIR,
)

# %%
print("Train and evaluate chosen pipelines.")
performance_evaluator.train_and_evaluate(
    model_options=model_options,
    dim_reduction_algorithm_options=dim_reduction_algorithm_options,
    n_dimensions_options=n_dimensions_options,
    count_evidence_options=count_evidence_options,
    include_absent_evidence_options=include_absent_evidence_options,
    n_most_frequent_options=n_most_frequent_options,
    verbose=True,
)

# %%
print("Perform grid search.")
performance_evaluator.perform_gridsearch(
    hyperparameters=hyperparameters, with_plots=True, verbose=True
)

# %%
print("Get the found best predictor.")
performance_evaluator.get_best_predictor()

# %% Optional: plot performance for other paramaters of the pipeline:
performance_evaluator.plot_performance(
    model_class=LinearSVC, parameters=["n_dimensions"]
)

# %%
