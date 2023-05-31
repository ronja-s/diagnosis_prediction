import json
from typing import List, Optional, Tuple

import pandas as pd
from global_variables import INDEX_COLUMN, TARGET_COLUMN
from sklearn.model_selection import train_test_split


class DataLoader:
    """Class for loading the data used for building the model.
    Some manual preprocessing is done here in order to get a suitable format for the
    data.
    """

    def __init__(self, multi_label: bool = False) -> None:
        """Args:
        multi_labe (bool): If True, target is a list of ICD-10 code blocks (str),
        possibly resulting in a multi-label classification task. If False, target is a
        single string containing this list of ICD-10 code blocks. Defaults to False.
        """
        self.multi_label = multi_label
        self._data_df = None
        self._TARGET_COLUMN = TARGET_COLUMN

    def load(
        self,
        data_path: str,
        icd10_chapters_definition_path: str,
        test_size: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Args:
            data_path (str): Path to the data that should be loaded.

            icd10_chapters_definition_path (str): Path to the file containing the
                definitions of the ICD-10 chapters.

            test_size (Optional[float]): Percentage of the data that should be held
            back as test set. If None, then no data split is performed. Default to None

        Returns:
            Tuple[pd.DataFrame, pd.Series]:
            (X, y) if test_split is None;
            (X_train, X_test, y_train, y_test) if test_split is not None;

            Columns of feature dataframe X: sex (str), age_in_months (int),
            evidence_present (List[str]), evidence_absent (List[str]);

            Name of target series y: expected_condition_icd10_blocks (List[str] or str,
            depending on parameter multi_label)
        """
        self._load_raw_dataframe(data_path=data_path)
        self._manipulate_raw_dataframe(
            icd10_chapters_definition_path=icd10_chapters_definition_path
        )
        y = self._data_df[self._TARGET_COLUMN]
        X = self._data_df.drop(columns=[self._TARGET_COLUMN])
        if test_size is None:
            return X, y
        else:
            return train_test_split(X, y, test_size=test_size)

    def _load_raw_dataframe(self, data_path: str) -> None:
        with open(data_path, encoding="utf8") as json_file:
            raw_data = json.load(json_file)
        self._data_df = pd.json_normalize(raw_data)
        self._data_df.set_index(INDEX_COLUMN, inplace=True)

    def _manipulate_raw_dataframe(self, icd10_chapters_definition_path: str) -> None:
        self.__check_if_dataframe_is_loaded
        self.__drop_undesirable_columns()
        self.__prepare_age_and_sex_columns()
        self.__extract_evidence()
        self.__group_icd10_codes(
            icd10_chapters_definition_path=icd10_chapters_definition_path
        )

    def __drop_undesirable_columns(self) -> None:
        COLUMNS_TO_DROP = [
            "public_test_case_name",
            "public_test_case_source",
            "expected_condition_name",
            "expected_condition_common_name",
            "expected_condition_id",
            "expected_condition_position_range",
            "test_case_passed",
            "predicted_conditions",
        ]
        self._data_df = self._data_df.drop(columns=COLUMNS_TO_DROP)

    def __prepare_age_and_sex_columns(self) -> None:
        # rename serialized api_payload columns:
        self._data_df.rename(
            columns={
                "api_payload.sex": "sex",
                "api_payload.age.unit": "age_unit",
                "api_payload.age.value": "age",
            },
            inplace=True,
        )

        # use age in months (consistent unit) as column:
        self._data_df["age_in_months"] = self._data_df.apply(
            lambda row: (
                row["age"] * 12 + 6 if row["age_unit"] == "year" else row["age"]
            ),
            axis=1,
        )
        self._data_df = self._data_df.drop(columns=["age", "age_unit"])

    def __extract_evidence(self) -> None:
        self._data_df["evidence_present"] = self._data_df["evidence.present"].map(
            lambda x: [list_entry["id"] for list_entry in x]
        )
        self._data_df["evidence_absent"] = self._data_df["evidence.absent"].map(
            lambda x: [list_entry["id"] for list_entry in x]
        )
        self._data_df = self._data_df.drop(
            columns=[
                "evidence.present",
                "evidence.absent",
                "evidence.unknown",
                "api_payload.evidence",
            ]
        )

    def __group_icd10_codes(self, icd10_chapters_definition_path: str) -> None:
        grouping = ICD10Grouping(
            icd10_chapters_definition_path=icd10_chapters_definition_path
        )
        self._data_df[self._TARGET_COLUMN] = self._data_df[
            "expected_condition_icd10_codes"
        ].map(
            lambda code_list: grouping.map_multiple_icd10_codes_to_chapters(code_list)
        )
        self._data_df = self._data_df.drop(columns=["expected_condition_icd10_codes"])
        if not self.multi_label:
            self._data_df[self._TARGET_COLUMN] = self._data_df[
                self._TARGET_COLUMN
            ].astype(str)

    def __check_if_dataframe_is_loaded(self) -> None:
        if self._data_df.empty:
            raise ValueError("Raw dataframe is not yet loaded.")


class ICD10Grouping:
    """Class for grouping ICD-10 codes into chapters."""

    def __init__(self, icd10_chapters_definition_path: str) -> None:
        self.data_path = icd10_chapters_definition_path
        self.icd10_chapters_definition_df = None

    def map_icd10_code_to_chapter(self, icd10_code: str) -> str:
        self._load_data()
        for icd10_block in self.icd10_chapters_definition_df["block"]:
            block_start, block_end = tuple(icd10_block.split("-"))

            # check if icd10_code is in the given range:
            if (icd10_code >= block_start) and (icd10_code <= block_end):
                return icd10_block
            else:
                continue

    def map_multiple_icd10_codes_to_chapters(
        self, icd10_code_list: List[str]
    ) -> List[str]:
        self._load_data()
        return sorted(
            list(
                set(
                    [
                        self.map_icd10_code_to_chapter(icd10_code=code)
                        for code in icd10_code_list
                    ]
                )
            )
        )

    def _load_data(self) -> None:
        if self.icd10_chapters_definition_df is None:
            self.icd10_chapters_definition_df = pd.read_csv(self.data_path)
