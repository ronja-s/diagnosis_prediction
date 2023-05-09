from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EvidenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        handle_unknown: str = "ignore",
        count_evidence: bool = False,
        n_most_frequent: Optional[int] = None,
        include_absent_evidence: bool = True,
        column_present_evidence: str = "evidence_present",
        column_absent_evidence: str = "evidence_absent",
    ) -> None:
        super().__init__()
        self.handle_unknown = handle_unknown
        self.count_evidence = count_evidence
        self.n_most_frequent = n_most_frequent
        self.include_absent_evidence = include_absent_evidence
        self.column_present_evidence = column_present_evidence
        self.column_absent_evidence = column_absent_evidence
        self.all_evidence = None

    def fit(self, X, y=None):
        list_to_flatten = list(X[self.column_present_evidence].values)
        if self.include_absent_evidence:
            list_to_flatten += list(X[self.column_absent_evidence].values)
        self.all_evidence = sorted(
            list(set([item for sublist in list_to_flatten for item in sublist]))
        )

        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        def encode_evidence(
            present_evidence_list: Optional[List[str]] = None,
            absent_evidence_list: Optional[List[str]] = None,
        ):
            if self.all_evidence is None:
                raise RuntimeError("Must fit before transform.")
            # use the following encoding:
            # -1 -> "absent"
            # 0 -> "unknown" or not given
            # 1 -> "present"
            encoded_list = [0] * len(self.all_evidence)
            if present_evidence_list:
                for evidence in present_evidence_list:
                    if evidence not in self.all_evidence:
                        if self.handle_unknown == "ignore":
                            continue
                        elif self.handle_unknown == "error":
                            raise ValueError(f"{evidence} is an unknown evidence.")
                    encoded_list[self.all_evidence.index(evidence)] = 1
            if absent_evidence_list:
                for evidence in absent_evidence_list:
                    if evidence not in self.all_evidence:
                        if self.handle_unknown == "ignore":
                            continue
                        elif self.handle_unknown == "error":
                            raise ValueError(f"{evidence} is an unknown evidence.")
                    encoded_list[self.all_evidence.index(evidence)] = -1
            return encoded_list

        encoded_evidence = X.apply(
            lambda row: encode_evidence(
                present_evidence_list=row[self.column_present_evidence],
                absent_evidence_list=(
                    row[self.column_absent_evidence]
                    if self.include_absent_evidence
                    else None
                ),
            ),
            axis=1,
        )
        evidence_columns_df = pd.DataFrame(
            data=list(encoded_evidence.values),
            columns=self.all_evidence,
            index=X.index,
        )

        evidence_count_df = pd.DataFrame(index=evidence_columns_df.index)
        if self.count_evidence:
            evidence_count_df["present_evidence_count"] = X.apply(
                lambda row: len(row[self.column_present_evidence]),
                axis=1,
            )
            if self.include_absent_evidence:
                evidence_count_df["absent_evidence_count"] = X.apply(
                    lambda row: len(row[self.column_absent_evidence]),
                    axis=1,
                )

        if self.n_most_frequent is not None:
            present_evidence_count_df = (
                evidence_columns_df.replace(-1, 0).sum().sort_values(ascending=False)
            )
            most_frequent_present_evidence = list(
                present_evidence_count_df[: self.n_most_frequent].index
            )
            most_frequent_evidence = most_frequent_present_evidence
            if self.include_absent_evidence:
                absent_evidence_count_df = (
                    -evidence_columns_df.replace(1, 0).sum()
                ).sort_values(ascending=False)
                most_frequent_absent_evidence = list(
                    absent_evidence_count_df[: self.n_most_frequent].index
                )
                most_frequent_evidence = list(
                    set(most_frequent_evidence + most_frequent_absent_evidence)
                )
            evidence_columns_df = evidence_columns_df[most_frequent_evidence]

        return evidence_columns_df.join(evidence_count_df)
