import sys
from typing import Dict, List, Optional, Type

import pandas as pd
import sklearn
from evidence_encoder import EvidenceEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler


class PipelineBuilder:
    """Class for building a pipeline for predicting diagnoses."""

    def __init__(
        self,
        model: sklearn.base.BaseEstimator,
        dim_reduction_algorithm: Optional[Type] = None,
        n_dimensions: Optional[int] = None,
        count_evidence: bool = False,
        include_absent_evidence: bool = True,
        n_most_frequent: Optional[int] = None,
    ) -> None:
        """Args:
        model (sklearn.base.BaseEstimator): Classifier used for the prediction.

        dim_reduction_algorithm (Optional[Type]): Algorithm that should be used to
        reduce the dimensions of the feature space. If None, then none is used. Defaults
        to None.

        n_dimensions (Optional[int]): Number of dimensions that the output of the
        dimensionality reduction algorithm should have. If parameter
        "dim_reduction_algorithm" is None, "n_dimensions" must be None as well. Defaults
        to None.

        count_evidence (bool): If True, include a feature for the overall number of
        present evidence for each patient (and a feature for the overall number of
        absent evidence for each patient if "include_absent_evidence" is True). Defaults
        to False.

        include_absent_evidence (bool): If True, absent evidence are also used as
        features. Otherwise, only present evidence are used. Defaults to True.

        n_most_frequent (Optional[int]): If not None, only use the given number of most
        frequent evidence as festures. Defaults to None.

        Raises:
            PipelineParameterCombinationError: If non-valid parameter combinations are
            used.
        """
        self.model = model
        self.dim_reduction_algorithm = dim_reduction_algorithm
        self.n_dimensions = n_dimensions
        self.count_evidence = count_evidence
        self.include_absent_evidence = include_absent_evidence
        self.n_most_frequent = n_most_frequent
        self.__check_if_parameters_are_valid()

        self._CATEGORICAL_COLUMNS_EVIDENCE = ["evidence_present", "evidence_absent"]
        self._CATEGORICAL_COLUMS_OTHER = ["sex"]
        self._NUMERICAL_COLUMNS = ["age_in_months"]
        self.__PRESENT_EVIDENCE_COUNT = "present_evidence_count"
        self.__ABSENT_EVIDENCE_COUNT = "absent_evidence_count"
        if self.include_absent_evidence:
            self._EVIDENCE_COUNT_COLUMNS = [
                self.__PRESENT_EVIDENCE_COUNT,
                self.__ABSENT_EVIDENCE_COUNT,
            ]
        else:
            self._EVIDENCE_COUNT_COLUMNS = [self.__PRESENT_EVIDENCE_COUNT]

    def get_pipe(
        self,
    ) -> Pipeline:
        steps = [("encoding_and_scaling", self._get_encoding_and_scaling_pipe())]

        if self.dim_reduction_algorithm:
            steps.append(
                (
                    "dimensionality_reduction",
                    self.dim_reduction_algorithm(n_components=self.n_dimensions),
                )
            )

        steps.append(("model", self.model))

        return Pipeline(steps=steps)

    def perform_preprocessing(
        self, X: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        pipe = self.get_pipe()
        X_transformed = X
        for name, transformer in pipe.steps[:-1]:
            if transformer is not None:
                X_transformed = transformer.fit_transform(X_transformed)
                if verbose:
                    print(
                        f"X after step {name}: {X_transformed}\nShape:"
                        f" {X_transformed.shape}"
                    )
        return X_transformed

    def _get_encoding_and_scaling_pipe(self) -> ColumnTransformer:
        transformer = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                (
                    "sex_encoding",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    self._CATEGORICAL_COLUMS_OTHER,
                ),
                ("age_scaler", StandardScaler(), self._NUMERICAL_COLUMNS),
                (
                    "evidence_encoding",
                    self.__get_evidence_encoding_pipe(),
                    self._CATEGORICAL_COLUMNS_EVIDENCE,
                ),
            ],
        )
        return transformer

    def __get_evidence_encoding_pipe(
        self,
    ) -> Pipeline:
        steps = [
            (
                "evidence_encoding",
                EvidenceEncoder(
                    handle_unknown="ignore",
                    count_evidence=self.count_evidence,
                    n_most_frequent=self.n_most_frequent,
                    include_absent_evidence=self.include_absent_evidence,
                ),
            )
        ]
        if self.count_evidence:
            steps.append(
                (
                    "evidence_count_scaling",
                    ColumnTransformer(
                        remainder="passthrough",
                        transformers=[
                            (
                                "min_max_scaling",
                                MinMaxScaler(feature_range=(-1, 1)),
                                self._EVIDENCE_COUNT_COLUMNS,
                            )
                        ],
                    ),
                ),
            )

        return Pipeline(steps=steps)

    def __check_if_parameters_are_valid(self) -> None:
        if (self.dim_reduction_algorithm is None) and (self.n_dimensions is not None):
            raise ValueError(
                "If dim_reduction_algorithm is None, n_dimensions must be None."
            )
