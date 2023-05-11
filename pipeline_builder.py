import sys
from typing import Dict, Optional, Type, List

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler

from evidence_encoder import EvidenceEncoder


class PipelineBuilder:
    def __init__(self) -> None:
        pass

    def __get_evidence_encoding_pipe(
        self,
        count_evidences: bool = False,
        include_absent_evidence: bool = True,
        n_most_frequent: Optional[int] = None,
    ) -> Pipeline:
        evidence_count_cols = ["present_evidence_count"]
        if include_absent_evidence:
            evidence_count_cols += ["absent_evidence_count"]

        evidence_encoding_steps = [
            (
                "encoding",
                EvidenceEncoder(
                    handle_unknown="ignore",
                    count_evidence=count_evidences,
                    n_most_frequent=n_most_frequent,
                    include_absent_evidence=include_absent_evidence,
                ),
            )
        ]
        if count_evidences:
            evidence_encoding_steps.append(
                (
                    "scaling",
                    ColumnTransformer(
                        remainder="passthrough",
                        transformers=[
                            (
                                "min_max_scaling",
                                MinMaxScaler(feature_range=(-1, 1)),
                                evidence_count_cols,
                            )
                        ],
                    ),
                ),
            )

        return Pipeline(steps=evidence_encoding_steps)

    def __get_preprocessing_pipe(
        self,
        count_evidence: bool = False,
        include_absent_evidence: bool = True,
        n_most_frequent: Optional[int] = None,
    ) -> ColumnTransformer:
        categorical_cols_evidence = ["evidence_present", "evidence_absent"]
        categorical_cols_other = ["sex"]
        numerical_cols = ["age_in_months"]

        evidence_encoding_pipe = self.__get_evidence_encoding_pipe(
            count_evidences=count_evidence,
            include_absent_evidence=include_absent_evidence,
            n_most_frequent=n_most_frequent,
        )

        preprocessing = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                (
                    "sex_encoding",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    categorical_cols_other,
                ),
                ("age_scaler", StandardScaler(), numerical_cols),
                (
                    "evidence_encoding",
                    evidence_encoding_pipe,
                    categorical_cols_evidence,
                ),
            ],
        )

        return preprocessing

    def get_pipe(
        self,
        model: sklearn.base.BaseEstimator,
        dim_reduction_algorithm: Optional[Type] = None,
        n_dimensions: Optional[int] = None,
        count_evidence: bool = False,
        include_absent_evidence: bool = True,
        n_most_frequent: Optional[int] = None,
    ) -> Pipeline:
        preprocessing = self.__get_preprocessing_pipe(
            count_evidence=count_evidence,
            include_absent_evidence=include_absent_evidence,
            n_most_frequent=n_most_frequent,
        )

        def str_to_class(class_name: str):
            return getattr(sys.modules[__name__], class_name)

        # check if parameters are valid:
        if (dim_reduction_algorithm is None) and (n_dimensions is not None):
            raise ValueError(
                "When dim_reduction_algorithm is None, n_dimensions must be None."
            )

        steps = [("preprocessing", preprocessing)]
        if dim_reduction_algorithm:
            steps.append(
                (
                    "dimensionality_reduction",
                    dim_reduction_algorithm(n_components=n_dimensions),
                )
            )
        steps.append(("model", model))

        return Pipeline(steps=steps)

    def get_pipes(
        self,
        models: List[sklearn.base.BaseEstimator],
        dim_reduction_algorithm: Optional[Type] = None,
        n_dimensions: Optional[int] = None,
        count_evidence: bool = False,
        include_absent_evidence: bool = True,
        n_most_frequent: Optional[int] = None,
    ) -> Dict[str, Pipeline]:
        pipes = dict(
            [
                (
                    model.__class__.__name__,
                    self.get_pipe(
                        model=model,
                        dim_reduction_algorithm=dim_reduction_algorithm,
                        n_dimensions=n_dimensions,
                        count_evidence=count_evidence,
                        include_absent_evidence=include_absent_evidence,
                        n_most_frequent=n_most_frequent,
                    ),
                )
                for model in models
            ]
        )
        return pipes
