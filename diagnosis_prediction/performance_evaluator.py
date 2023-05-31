import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from pipeline_builder import PipelineBuilder
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline


class PerformanceEvaluator:
    """Class for evaluating a variety of different pipelines and hyperparameters in
    order to find the best performing one.
    """

    _MODEL_COLUMN = "model"
    _MODEL_STR_COLUMN = _MODEL_COLUMN + "_str"
    _DIM_REDUCTION_ALGORITHM_COLUMN = "dim_reduction_algorithm"
    _N_DIMENSIONS_COLUMN = "n_dimensions"
    _COUNT_EVIDENCE_COLUMN = "count_evidence"
    _INCLUDE_ABSENT_EVIDENCE_COLUMN = "include_absent_evidence"
    _N_MOST_FREQUENT_COLUMN = "n_most_frequent"
    _TRAIN_SCORE_COLUMN = "mean_train_score"
    _TEST_SCORE_COLUMN = "mean_test_score"
    _PARAMS_COLUMN = "params"
    _PIPELINE_COLUMN = "pipeline"

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        performance_metric: Callable,
        cross_validation_splitting: int = 5,
        results_directory: str = ".",
    ) -> None:
        self.X = X
        self.y = y
        self.performance_metric = performance_metric
        self.cross_validation_splitting = cross_validation_splitting
        self.results_directory = results_directory

        self.train_and_evaluate_df = None
        self.grid_search_df = None
        self.best_parameters_df = None

        self.TRAIN_AND_EVALUATE_FILE_PATH = self._get_file_path(
            filename="train_and_evaluate"
        )
        self.GRID_SEARCH_FILE_PATH = self._get_file_path(filename="grid_search")
        self.BEST_PARAMETERS_FILE_PATH = self._get_file_path(filename="best_parameters")
        self.BEST_PREDICTOR_FILE_PATH = self._get_file_path(filename="best_predictor")

    def train_and_evaluate(
        self,
        model_options: List[sklearn.base.BaseEstimator],
        dim_reduction_algorithm_options: List[Optional[Type]],
        n_dimensions_options: List[Optional[int]],
        count_evidence_options: List[bool],
        include_absent_evidence_options: List[bool],
        n_most_frequent_options: List[Optional[int]],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Train and evaluate all pipeline with the parameters given in the constructor.
        For evaluation, a cross validation is performed.
        """
        self.train_and_evaluate_df = pd.DataFrame()

        for model in model_options:
            for dim_reduction_algorithm in dim_reduction_algorithm_options:
                for n_dimensions in n_dimensions_options:
                    for count_evidence in count_evidence_options:
                        for include_absent_evidence in include_absent_evidence_options:
                            for n_most_frequent in n_most_frequent_options:
                                try:
                                    if verbose:
                                        print(
                                            f"Evaluate for: model={model},"
                                            f" dim_reduction_algorithm={self.__get_class_name(dim_reduction_algorithm)},"
                                            f" n_dimensions={n_dimensions},"
                                            f" count_evidences={count_evidence},"
                                            f" include_absent_evidence={include_absent_evidence},"
                                            f" n_most_frequent={n_most_frequent}"
                                        )

                                    pipe = PipelineBuilder(
                                        model=model,
                                        dim_reduction_algorithm=dim_reduction_algorithm,
                                        n_dimensions=n_dimensions,
                                        count_evidence=count_evidence,
                                        include_absent_evidence=include_absent_evidence,
                                        n_most_frequent=n_most_frequent,
                                    ).get_pipe()
                                    scores = cross_validate(
                                        estimator=pipe,
                                        X=self.X,
                                        y=self.y,
                                        scoring=make_scorer(self.performance_metric),
                                        cv=self.cross_validation_splitting,
                                        return_train_score=True,
                                    )
                                    test_score = scores["test_score"].mean()
                                    train_score = scores["train_score"].mean()
                                except ValueError as ex:
                                    # ignore parameter combinations which are not valid
                                    if verbose:
                                        print(f"{type(ex).__name__}: {str(ex)}")
                                    continue

                                if verbose:
                                    print(
                                        "=> Test"
                                        f" score={test_score},"
                                        " train"
                                        f" score={train_score}"
                                    )

                                new_row = pd.DataFrame(
                                    {
                                        self._MODEL_COLUMN: [model],
                                        self._MODEL_STR_COLUMN: [str(model)],
                                        self._DIM_REDUCTION_ALGORITHM_COLUMN: [
                                            dim_reduction_algorithm
                                        ],
                                        self._N_DIMENSIONS_COLUMN: [n_dimensions],
                                        self._COUNT_EVIDENCE_COLUMN: [count_evidence],
                                        self._INCLUDE_ABSENT_EVIDENCE_COLUMN: [
                                            include_absent_evidence
                                        ],
                                        self._N_MOST_FREQUENT_COLUMN: [n_most_frequent],
                                        self._TRAIN_SCORE_COLUMN: [train_score],
                                        self._TEST_SCORE_COLUMN: [test_score],
                                        self._PIPELINE_COLUMN: [pipe],
                                    }
                                )
                                self.train_and_evaluate_df = pd.concat(
                                    [self.train_and_evaluate_df, new_row],
                                    axis=0,
                                    ignore_index=True,
                                )
            self._write_files(
                df=self.train_and_evaluate_df,
                file_path=self.TRAIN_AND_EVALUATE_FILE_PATH,
            )
        self._set_best_parameters_df()
        return self.train_and_evaluate_df

    def perform_gridsearch(
        self,
        hyperparameters: Dict[Type, Dict[str, list]],
        with_plots: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Perform grid searchs to tune the given hyperparameters on the best found
        pipelines.
        """
        self.grid_search_df = pd.DataFrame()
        for model_class in hyperparameters.keys():
            model, best_pipeline = self._get_best_pipeline_for_model(
                model_class=model_class
            )
            if (best_pipeline is None) and verbose:
                print(f"No pipeline found for {self.__get_class_name(model_class)}.")
                continue

            if verbose:
                print(f"Run grid search for {self.__get_class_name(model_class)}.")
            param_grid = dict(
                ("model__" + key, value)
                for (
                    key,
                    value,
                ) in hyperparameters[model_class].items()
            )
            single_grid_search_df = self._perform_gridsearch_for_pipeline(
                pipeline=best_pipeline,
                param_grid=param_grid,
            )
            single_grid_search_df[self._MODEL_COLUMN] = [model] * len(
                single_grid_search_df.index
            )  # workaround because of RandomForestClassifier's __len__ method
            single_grid_search_df[self._MODEL_STR_COLUMN] = str(model)
            self.grid_search_df = pd.concat(
                [self.grid_search_df, single_grid_search_df],
                ignore_index=True,
            )

            if verbose:
                print(
                    "Create plot for grid"
                    " search of"
                    f" {self.__get_class_name(model_class)}."
                )
            if with_plots:
                self.plot_performance(
                    model_class=model_class,
                    parameters=param_grid.keys(),
                )
            self._write_files(
                df=self.grid_search_df,
                file_path=self.GRID_SEARCH_FILE_PATH,
            )

        self._set_best_parameters_df()
        return self.grid_search_df

    def plot_performance(
        self,
        model_class: Type,
        parameters: List[str],
    ) -> None:
        """Plot the performance metric evaluated on the train and test set for the given
        model class against the given parameter(s). For each parameter, one figure is
        created.
        """
        for parameter in parameters:
            (
                param_values,
                train_score_values,
                train_score_error_values,
                test_score_values,
                test_score_error_values,
            ) = self.__get_performance_for_parameter(
                model_class=model_class,
                parameter=parameter,
            )

            param_values = pd.to_numeric(param_values, errors="coerce")
            if param_values.nunique() > 1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.errorbar(
                    x=param_values,
                    y=train_score_values,
                    yerr=train_score_error_values,
                    capsize=3.0,
                    marker="o",
                    label=self._TRAIN_SCORE_COLUMN,
                )
                ax.errorbar(
                    x=param_values,
                    y=test_score_values,
                    yerr=test_score_error_values,
                    capsize=3.0,
                    marker="o",
                    label=self._TEST_SCORE_COLUMN,
                )
                ax.set_title(self.__get_class_name(model_class))
                ax.set_xlabel(parameter)
                ax.set_ylabel(self.performance_metric.__name__)
                ax.set_ylim([0.0, 1.0])
                leg = ax.legend()
                plot_file_path = self._get_file_path(
                    filename=self.__get_class_name(model_class)
                    + "_"
                    + parameter
                    + ".jpg"
                )
                fig.savefig(
                    plot_file_path,
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close(fig)

    def get_best_predictor(self) -> BaseEstimator:
        self.best_parameters_df = self._load_dataframe(
            df=self.best_parameters_df,
            file_path=self.BEST_PARAMETERS_FILE_PATH,
        )
        best_pipeline = self.best_parameters_df.loc[
            self.best_parameters_df[self._TEST_SCORE_COLUMN].idxmax()
        ][self._PIPELINE_COLUMN]
        best_predictor = best_pipeline.fit(self.X, self.y)
        self.__write_pickle(
            data=best_predictor,
            file_path=self.BEST_PREDICTOR_FILE_PATH + ".pkl",
        )
        return best_predictor

    def _get_file_path(self, filename: str) -> str:
        return os.path.join(self.results_directory, filename)

    @staticmethod
    def _load_dataframe(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        if df is None:
            with open(file_path + ".pkl", "rb") as file:
                df = pickle.load(file)
        return df

    def _write_files(self, df: pd.DataFrame, file_path: str) -> None:
        if df.empty:
            return
        self.__write_csv(df=df, file_path=file_path + ".csv")
        self.__write_pickle(data=df, file_path=file_path + ".pkl")

    def _set_best_parameters_df(self) -> None:
        self.train_and_evaluate_df = self._load_dataframe(
            df=self.train_and_evaluate_df,
            file_path=self.TRAIN_AND_EVALUATE_FILE_PATH,
        )
        self.best_parameters_df = self.__get_best_performing_entries(
            df=self.train_and_evaluate_df,
            performance_column=self._TEST_SCORE_COLUMN,
            groupby_column=self._MODEL_STR_COLUMN,
        )
        self.__append_gridsearch_results_to_best_parameters()
        self._write_files(
            df=self.best_parameters_df,
            file_path=self.BEST_PARAMETERS_FILE_PATH,
        )

    def _perform_gridsearch_for_pipeline(
        self, pipeline: Pipeline, param_grid
    ) -> pd.DataFrame:
        grid_search_cv = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=make_scorer(self.performance_metric),
            cv=self.cross_validation_splitting,
            refit=True,
            return_train_score=True,
        )
        grid_search_cv.fit(self.X, self.y)
        results_df = pd.DataFrame(grid_search_cv.cv_results_)[
            [
                self._PARAMS_COLUMN,
                self._TEST_SCORE_COLUMN,
                self._TRAIN_SCORE_COLUMN,
            ]
        ]
        results_df[self._PIPELINE_COLUMN] = grid_search_cv.best_estimator_
        return results_df

    def _get_best_pipeline_for_model(
        self, model_class: Type
    ) -> Optional[Tuple[BaseEstimator, Pipeline]]:
        self.best_parameters_df = self._load_dataframe(
            df=self.best_parameters_df,
            file_path=self.BEST_PARAMETERS_FILE_PATH,
        )
        model_df = (
            self.best_parameters_df[
                self.best_parameters_df[self._MODEL_COLUMN].map(
                    lambda model: type(model) == model_class
                )
            ]
            .squeeze(axis=0)
            .replace(np.nan, None)
        )
        if not model_df.empty:
            model = model_df[self._MODEL_COLUMN]
            best_pipeline = PipelineBuilder(
                model=model,
                dim_reduction_algorithm=model_df[self._DIM_REDUCTION_ALGORITHM_COLUMN],
                n_dimensions=model_df[self._N_DIMENSIONS_COLUMN],
                count_evidence=model_df[self._COUNT_EVIDENCE_COLUMN],
                include_absent_evidence=model_df[self._INCLUDE_ABSENT_EVIDENCE_COLUMN],
                n_most_frequent=model_df[self._N_MOST_FREQUENT_COLUMN],
            ).get_pipe()
            return model, best_pipeline
        else:
            return None, None

    def __get_performance_for_parameter(
        self, model_class: Type, parameter: str
    ) -> Tuple[List]:
        self.train_and_evaluate_df = self._load_dataframe(
            df=self.train_and_evaluate_df,
            file_path=self.TRAIN_AND_EVALUATE_FILE_PATH,
        )
        if parameter in self.train_and_evaluate_df.columns:
            model_df = self.train_and_evaluate_df[
                self.train_and_evaluate_df[self._MODEL_COLUMN].map(
                    lambda model: type(model) == model_class
                )
            ]
        else:
            self.grid_search_df = self._load_dataframe(
                df=self.grid_search_df,
                file_path=self.GRID_SEARCH_FILE_PATH,
            )
            model_df = self.grid_search_df[
                self.grid_search_df[self._MODEL_COLUMN].map(
                    lambda model: type(model) == model_class
                )
            ]
            # explode param dictionary column into multiple columns:
            model_df = pd.concat(
                [
                    model_df,
                    model_df[self._PARAMS_COLUMN].apply(pd.Series),
                ],
                axis=1,
            )
        (
            param_values,
            train_score_values,
            train_score_error_values,
        ) = self.__get_mean_and_std(
            df=model_df,
            x_column=parameter,
            y_column=self._TRAIN_SCORE_COLUMN,
        )
        (
            _,
            test_score_values,
            test_score_error_values,
        ) = self.__get_mean_and_std(
            df=model_df,
            x_column=parameter,
            y_column=self._TEST_SCORE_COLUMN,
        )
        return (
            param_values,
            train_score_values,
            train_score_error_values,
            test_score_values,
            test_score_error_values,
        )

    @staticmethod
    def __get_mean_and_std(df, x_column, y_column):
        statistics_df = (
            df.groupby(by=x_column)[y_column].agg(["mean", "std"]).reset_index()
        )
        return (
            statistics_df[x_column],
            statistics_df["mean"],
            statistics_df["std"].fillna(0.0),
        )

    def __append_gridsearch_results_to_best_parameters(
        self,
    ) -> None:
        try:
            self.grid_search_df = self._load_dataframe(
                df=self.grid_search_df,
                file_path=self.GRID_SEARCH_FILE_PATH,
            )
        except FileNotFoundError:
            return
        if not self.grid_search_df.empty:
            best_gridsearch_df = self.__get_best_performing_entries(
                df=self.grid_search_df,
                performance_column=self._TEST_SCORE_COLUMN,
                groupby_column=self._MODEL_STR_COLUMN,
            )
            shared_columns = self.best_parameters_df.columns.intersection(
                best_gridsearch_df.columns
            ).difference([self._MODEL_STR_COLUMN])
            new_suffix = "_new"
            new_columns = [col + new_suffix for col in shared_columns]
            self.best_parameters_df = self.best_parameters_df.merge(
                best_gridsearch_df,
                on=self._MODEL_STR_COLUMN,
                how="left",
                suffixes=(None, new_suffix),
            )
            new_data_df = self.best_parameters_df[new_columns]
            self.best_parameters_df = self.best_parameters_df.drop(columns=new_columns)
            self.best_parameters_df.update(
                new_data_df,
                overwrite=True,
            )
            self.best_parameters_df = self.best_parameters_df.sort_values(
                self._TEST_SCORE_COLUMN,
                ascending=False,
            )

    @staticmethod
    def __get_best_performing_entries(
        df: pd.DataFrame,
        performance_column: str,
        groupby_column: str,
    ) -> pd.DataFrame:
        best_indices = df.groupby(groupby_column)[performance_column].idxmax().values
        best_entries_df = (
            df.iloc[best_indices]
            .sort_values(
                performance_column,
                ascending=False,
            )
            .reset_index()
        )
        return best_entries_df

    def __write_csv(self, df: pd.DataFrame, file_path: str) -> None:
        str_df = df.copy().drop(
            columns=[
                self._MODEL_STR_COLUMN,
                self._PIPELINE_COLUMN,
            ],
            errors="ignore",
        )
        if self._DIM_REDUCTION_ALGORITHM_COLUMN in str_df.columns:
            str_df[self._DIM_REDUCTION_ALGORITHM_COLUMN] = str_df[
                self._DIM_REDUCTION_ALGORITHM_COLUMN
            ].map(self.__get_class_name)
        str_df.to_csv(file_path, index=False)

    @staticmethod
    def __write_pickle(data: Any, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(obj=data, file=f)

    @staticmethod
    def __get_class_name(
        cls: Type,
    ) -> Optional[str]:
        if cls is None:
            return None
        else:
            return cls.__name__
