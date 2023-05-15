import ast
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
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
from sklearn.metrics import accuracy_score, make_scorer, top_k_accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_loader import DataLoader
from pipeline_builder import PipelineBuilder

pd.plotting.register_matplotlib_converters()
warnings.filterwarnings("ignore")


def str_to_class(class_name: str):
    return getattr(sys.modules["__main__"], class_name)


def evaluate_pipelines(
    X: pd.DataFrame,
    y: pd.Series,
    models: List[sklearn.base.BaseEstimator],
    dim_reduction_algorithm_values: List[Optional[Type]],
    n_dimensions_values: List[Optional[int]],
    count_evidence_values: List[bool],
    include_absent_evidence_values: List[bool],
    n_most_frequent_values: List[Optional[int]],
    cross_validation_params: dict,
    result_file_path: Optional[str],
) -> pd.DataFrame:
    def perform_cross_validation(pipe, X, y, cross_validation_params):
        scores = cross_validate(pipe, X, y, **cross_validation_params)
        test_accuracy = scores["test_accuracy"].mean()
        train_accuracy = scores["train_accuracy"].mean()
        return test_accuracy, train_accuracy

    accuracies_df = pd.DataFrame(
        columns=[
            "dim_reduction_algorithm",
            "n_dimensions",
            "count_evidence",
            "include_absent_evidence",
            "n_most_frequent",
            "model_name",
            "train_accuracy",
            "test_accuracy",
        ]
    )

    # vary parameters of the pipes:
    for dim_reduction_algorithm in dim_reduction_algorithm_values:
        for n_dimensions in n_dimensions_values:
            for count_evidence in count_evidence_values:
                for include_absent_evidence in include_absent_evidence_values:
                    for n_most_frequent in n_most_frequent_values:
                        for model in models:
                            model_name = model.__class__.__name__
                            try:
                                pipe = PipelineBuilder(
                                    model=model,
                                    dim_reduction_algorithm=dim_reduction_algorithm,
                                    n_dimensions=n_dimensions,
                                    count_evidence=count_evidence,
                                    include_absent_evidence=include_absent_evidence,
                                    n_most_frequent=n_most_frequent,
                                ).get_pipe()
                            except ValueError:
                                # ignore parameter combinations which are not valid
                                continue

                            try:
                                (
                                    test_accuracy,
                                    train_accuracy,
                                ) = perform_cross_validation(
                                    pipe=pipe,
                                    X=X,
                                    y=y,
                                    cross_validation_params=cross_validation_params,
                                )
                            except ValueError:
                                # ignore parameter combinations which are not valid
                                continue
                            print(
                                f"Evaluate for: model={model_name}, dim_reduction_algorithm={dim_reduction_algorithm.__name__ if dim_reduction_algorithm is not None else None}, n_dimensions={n_dimensions}, count_evidences={count_evidence}, include_absent_evidence={include_absent_evidence}, n_most_frequent={n_most_frequent}"
                            )
                            print(
                                f"  => Test accuracy={test_accuracy}, train accuracy={train_accuracy}"
                            )

                            # write results into dataframe:
                            accuracies_df = accuracies_df.append(
                                {
                                    "dim_reduction_algorithm": dim_reduction_algorithm.__name__
                                    if dim_reduction_algorithm is not None
                                    else None,
                                    "n_dimensions": n_dimensions,
                                    "count_evidence": count_evidence,
                                    "include_absent_evidence": include_absent_evidence,
                                    "n_most_frequent": n_most_frequent,
                                    "model_name": model_name,
                                    "train_accuracy": train_accuracy,
                                    "test_accuracy": test_accuracy,
                                },
                                ignore_index=True,
                            )
    if result_file_path is not None:
        accuracies_df.to_csv(result_file_path, index=False)
    print(accuracies_df)
    return accuracies_df


def determine_best_parameters(
    input_file_path: str, accuracy_column: str, result_file_path: Optional[str]
) -> pd.DataFrame:
    accuracies_df = pd.read_csv(input_file_path)
    best_models_df = (
        accuracies_df.groupby("model_name")
        .apply(lambda df: df.loc[(df[accuracy_column].idxmax())])
        .sort_values(accuracy_column, ascending=False)
    )
    if result_file_path is not None:
        best_models_df.to_csv(result_file_path, index=False)
    print(best_models_df)
    return best_models_df


def perform_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    best_models_file_path: str,
    cross_validation_params: dict,
    hyperparameters: Dict[str, Dict[str, list]],
    result_file_path: Optional[str],
) -> pd.DataFrame:
    best_models_df = pd.read_csv(best_models_file_path)
    grid_search_results_df = pd.DataFrame()
    for model_name in hyperparameters.keys():
        # get best pipeline parameters for this model:
        best_pipe_parameters = (
            best_models_df[best_models_df["model_name"] == model_name]
            .squeeze(axis=0)
            .replace(np.nan, None)
        )
        if best_pipe_parameters.empty:
            print(
                f"Model {model_name} not contained in file {best_models_file_path}. Will try the next model."
            )
            continue
        if model_name == "LinearDiscriminantAnalysis":
            model = str_to_class(model_name)(solver="lsqr")
        else:
            model = str_to_class(model_name)()
        dim_reduction_algorithm = (
            str_to_class(best_pipe_parameters["dim_reduction_algorithm"])
            if best_pipe_parameters["dim_reduction_algorithm"]
            else None
        )
        best_pipeline = PipelineBuilder(
            model=model,
            dim_reduction_algorithm=dim_reduction_algorithm,
            n_dimensions=int(best_pipe_parameters["n_dimensions"])
            if best_pipe_parameters["n_dimensions"] is not None
            else None,
            count_evidence=best_pipe_parameters["count_evidence"],
            include_absent_evidence=best_pipe_parameters["include_absent_evidence"],
            n_most_frequent=best_pipe_parameters["n_most_frequent"],
        ).get_pipe()

        # perform grid search:
        param_grid = dict(
            ("model__" + key, value)
            for (key, value) in hyperparameters[model_name].items()
        )
        grid_search = GridSearchCV(
            estimator=best_pipeline,
            param_grid=param_grid,
            refit=False,
            **cross_validation_params,
        )
        print(f"Run grid search for {model_name}.")
        grid_search.fit(X, y)
        grid_search_df = pd.DataFrame(grid_search.cv_results_)[
            ["params", "mean_test_accuracy", "mean_train_accuracy"]
        ]
        grid_search_df["model_name"] = model_name
        grid_search_results_df = grid_search_results_df.append(grid_search_df)
    if result_file_path is not None:
        grid_search_results_df.to_csv(result_file_path, index=False)
    print(grid_search_results_df)
    return grid_search_results_df


def plot_grid_search_results(grid_search_file_path: str, result_dir: str):
    grid_search_results_df = pd.read_csv(grid_search_file_path)
    model_names = grid_search_results_df["model_name"].unique()

    for model_name in model_names:
        print(f"Create plot for {model_name}.")
        df = grid_search_results_df[grid_search_results_df["model_name"] == model_name]
        df["params"] = df["params"].map(lambda x: ast.literal_eval(x))
        hyperparams = df["params"].iloc[0].keys()
        for hyperparam in hyperparams:
            plain_hyperparam_name = hyperparam.replace("model__", "")
            df[plain_hyperparam_name] = df["params"].map(lambda x: x[hyperparam])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(
                df[plain_hyperparam_name],
                df["mean_train_accuracy"],
                marker="o",
                label="train accuracy",
            )
            ax.plot(
                df[plain_hyperparam_name],
                df["mean_test_accuracy"],
                marker="o",
                label="test accuracy",
            )
            ax.set_title(model_name)
            ax.set_xlabel(plain_hyperparam_name)
            ax.set_ylabel("accuracy")
            ax.set_ylim([0.0, 1.0])
            leg = ax.legend()
            file_path = (
                result_dir + "/" + model_name + "_" + plain_hyperparam_name + ".jpg"
            )
            fig.savefig(file_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
