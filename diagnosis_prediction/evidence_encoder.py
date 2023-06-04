from __future__ import annotations

from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EvidenceEncoder(BaseEstimator, TransformerMixin):
    """
    Class for transforming the input columns containing lists of evidence into columns
    containing integers, one for each possible evidence value.

    The following encoding is used:
    -1 if given evidence is absent;
    0 if given evidence is unknown or not given;
    1 if given evidence is present;
    """

    def __init__(
        self,
        handle_unknown: str = "ignore",
        count_evidence: bool = False,
        include_absent_evidence: bool = True,
        column_present_evidence: str = "evidence_present",
        column_absent_evidence: str = "evidence_absent",
    ) -> None:
        """Args:
        handle_unknown (str): How to handle unknown evidence values when transforming.
        Can be either "error" (raise error if there is an unknown evidence value) or
        "ignore" (unknown evidence values are encoded by 0). Defaults to "ignore".

        count_evidence (bool): If True, also create a column that counts the number of
        present evidence for each row (and a column that counts the number of absent
        evidence for each row if include_absent_evidence is True). Defaults to False.

        include_absent_evidence (bool): If True, absent evidence are also encoded. If
        True, only present evidence are encoded. Defaults to True.

        column_present_evidence (str): Name of the column of the input data X that
        contains the lists of present evidence. Defaults to "evidence_present".

        column_absent_evidence (str): Name of the columns of the input data X containing
        the lists of absent evidence. Defaults to "evidence_absent".
        """
        super().__init__()
        self.handle_unknown = handle_unknown
        self.count_evidence = count_evidence
        self.include_absent_evidence = include_absent_evidence
        self.column_present_evidence = column_present_evidence
        self.column_absent_evidence = column_absent_evidence
        self.all_evidence = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> EvidenceEncoder:
        self._set_all_evidence(X=X)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        evidence_encoding_df = self._get_evidence_encoding(X=X)
        evidence_count_df = self._get_evidence_count(X=X)
        return evidence_encoding_df.join(evidence_count_df)

    def _set_all_evidence(self, X: pd.DataFrame) -> None:
        list_to_flatten = list(X[self.column_present_evidence].values)
        if self.include_absent_evidence:
            list_to_flatten += list(X[self.column_absent_evidence].values)
        self.all_evidence = sorted(
            list(set([item for sublist in list_to_flatten for item in sublist]))
        )

    def _get_evidence_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded_evidence = X.apply(
            lambda row: self.__encode_list_of_evidence(
                present_evidence_list=row[self.column_present_evidence],
                absent_evidence_list=(
                    row[self.column_absent_evidence]
                    if self.include_absent_evidence
                    else None
                ),
            ),
            axis=1,
        )
        evidence_encoding_df = pd.DataFrame(
            data=list(encoded_evidence.values),
            columns=self.all_evidence,
            index=X.index,
        )
        return evidence_encoding_df

    def _get_evidence_count(self, X: pd.DataFrame) -> pd.DataFrame:
        PRESENT_EVIDENCE_COUNT_COL = "present_evidence_count"
        ABSENT_EVIDENCE_COUNT_COL = "absent_evidence_count"
        evidence_count_df = pd.DataFrame(index=X.index)
        if self.count_evidence:
            evidence_count_df[PRESENT_EVIDENCE_COUNT_COL] = X.apply(
                lambda row: len(row[self.column_present_evidence]),
                axis=1,
            )
            if self.include_absent_evidence:
                evidence_count_df[ABSENT_EVIDENCE_COUNT_COL] = X.apply(
                    lambda row: len(row[self.column_absent_evidence]),
                    axis=1,
                )
        return evidence_count_df

    def __encode_list_of_evidence(
        self,
        present_evidence_list: Optional[List[str]] = None,
        absent_evidence_list: Optional[List[str]] = None,
    ) -> List:
        HANDLE_UNKOWN_ERROR = "error"
        HANDLE_UNKOWN_IGNORE = "ignore"
        self.__check_if_all_evidence_is_set()
        encoded_list = [0] * len(self.all_evidence)
        if present_evidence_list:
            for evidence in present_evidence_list:
                if evidence not in self.all_evidence:
                    if self.handle_unknown == HANDLE_UNKOWN_IGNORE:
                        continue
                    elif self.handle_unknown == HANDLE_UNKOWN_ERROR:
                        raise ValueError(f"{evidence} is an unknown evidence.")
                encoded_list[self.all_evidence.index(evidence)] = 1
        if absent_evidence_list:
            for evidence in absent_evidence_list:
                if evidence not in self.all_evidence:
                    if self.handle_unknown == HANDLE_UNKOWN_IGNORE:
                        continue
                    elif self.handle_unknown == HANDLE_UNKOWN_ERROR:
                        raise ValueError(f"{evidence} is an unknown evidence.")
                encoded_list[self.all_evidence.index(evidence)] = -1
        return encoded_list

    def __check_if_all_evidence_is_set(self) -> None:
        if self.all_evidence is None:
            raise RuntimeError("Must fit before transform.")
