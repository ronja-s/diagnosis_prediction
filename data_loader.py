import json
from typing import List

import pandas as pd


class ConditionGrouping:
    """Class for grouping the possible conditions into categories."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.icd10_groups_df = None

    def _load_data(self) -> None:
        if self.icd10_groups_df is None:
            self.icd10_groups_df = pd.read_csv(self.data_path)

    def map_icd10_code_to_condition_group(self, icd10_code: str) -> str:
        self._load_data()
        for icd10_code_range in self.icd10_groups_df["icd10_code_range"]:
            start_range, end_range = tuple(icd10_code_range.split("-"))

            # check if icd10_code is in the given range:
            if (icd10_code >= start_range) and (icd10_code <= end_range):
                return icd10_code_range
            else:
                continue

    def map_icd10_code_list_to_condition_groups(
        self, icd10_code_list: List[str]
    ) -> List[str]:
        self._load_data()
        return sorted(
            list(
                set(
                    [
                        self.map_icd10_code_to_condition_group(icd10_code=code)
                        for code in icd10_code_list
                    ]
                )
            )
        )


class DataLoader:
    """Class for loading the data used for building the model.
    Some manual preprocessing is done here in order to get a suited format for the data."""

    def __init__(
        self,
        multi_label: bool = False,
        with_expected_condition_position_range: bool = False,
    ) -> None:
        self.multi_label = multi_label
        self.with_expected_condition_position_range = (
            with_expected_condition_position_range
        )

    def load(self, data_path: str, icd10_ranges_data_path: str) -> pd.DataFrame:
        with open(data_path, encoding="utf8") as json_file:
            raw_data = json.load(json_file)
        raw_data_df = pd.json_normalize(raw_data)

        # drop unimportant columns:
        columns_to_drop = [
            "public_test_case_name",
            "public_test_case_source",
            "expected_condition_name",
            "expected_condition_common_name",
            "expected_condition_id",
            "test_case_passed",
            "predicted_conditions",
        ]
        data_df = raw_data_df.drop(columns=columns_to_drop)

        # rename serialized api_payload columns:
        data_df.rename(
            columns={
                "api_payload.sex": "sex",
                "api_payload.age.unit": "age_unit",
                "api_payload.age.value": "age",
            },
            inplace=True,
        )

        # set index to "public_test_case_id":
        data_df.set_index("public_test_case_id", inplace=True)

        # transform column expected_condition_position range from string to number:
        if self.with_expected_condition_position_range:
            data_df["expected_condition_position_range"] = data_df[
                "expected_condition_position_range"
            ].map(lambda x: x.split("_")[-1])
        else:
            data_df = data_df.drop(columns=["expected_condition_position_range"])

        # convert age column into month unit:
        data_df["age_in_months"] = data_df.apply(
            lambda row: row["age"] * 12 + 6
            if row["age_unit"] == "year"
            else row["age"],
            axis=1,
        )
        # drop initial age columns:
        data_df = data_df.drop(columns=["age", "age_unit"])

        # get present and absent symptoms for each patient:
        data_df["evidence_present"] = data_df["evidence.present"].map(
            lambda x: [list_entry["id"] for list_entry in x]
        )
        data_df["evidence_absent"] = data_df["evidence.absent"].map(
            lambda x: [list_entry["id"] for list_entry in x]
        )
        # drop evidence columns that are not needed anymore:
        data_df = data_df.drop(
            columns=[
                "evidence.present",
                "evidence.absent",
                "evidence.unknown",
                "api_payload.evidence",
            ]
        )

        # group the expected_condition_icd10_codes into categories:
        # note: use string of list of categories in order to avoid a multilabel classification
        grouping = ConditionGrouping(data_path=icd10_ranges_data_path)
        data_df["expected_condition_groups"] = data_df[
            "expected_condition_icd10_codes"
        ].map(
            lambda code_list: grouping.map_icd10_code_list_to_condition_groups(
                code_list
            )
        )
        data_df = data_df.drop(columns=["expected_condition_icd10_codes"])
        if not self.multi_label:
            data_df["expected_condition_groups"] = data_df[
                "expected_condition_groups"
            ].astype(str)

        return data_df
