#!/usr/bin/env python3
# %%
import json
import os
from typing import List

import pandas as pd
import ydata_profiling
from data_loader import DataLoader
from evidence_encoder import EvidenceEncoder
from global_variables import (
    DATA_EXPLORATION_DIR,
    ICD10_CHAPTERS_DEFINITION_PATH,
    TARGET_COLUMN,
    TEST_CASES_PATH,
)

if not os.path.exists(DATA_EXPLORATION_DIR):
    os.makedirs(DATA_EXPLORATION_DIR)
RAW_DATA_EXPLORATION_FILE = "raw_data_exploration.html"
TRANSFORMED_DATA_EXPLORATION_FILE = "transformed_data_exploration.html"
ENCODED_DATA_EXPLORATION_FILE = "encoded_data_exploration.html"


def check_dependence_of_columns(
    df: pd.DataFrame, col_to_compare: str, possibly_derived_cols: List[str]
) -> None:
    """Check whether the columns possibly_derived_cols can be derived from the column
    col_to_compare."""
    for col in possibly_derived_cols:
        is_unique = df.groupby(col_to_compare).apply(
            lambda df, column=col: df[column].nunique() <= 1
        )
        if not is_unique.all():
            print(f"Column {col} cannot be derived from {col_to_compare}!")
        else:
            print(f"Column {col} can be derived from {col_to_compare}.")


# %% analyze raw data:
with open(TEST_CASES_PATH, encoding="utf8") as json_file:
    raw_data = json.load(json_file)
raw_data_df = pd.json_normalize(raw_data)
raw_data_df["expected_condition_icd10_codes_str"] = raw_data_df[
    "expected_condition_icd10_codes"
].astype(str)
profile_raw_data = ydata_profiling.ProfileReport(raw_data_df, minimal=False)
profile_raw_data.to_file(os.path.join(DATA_EXPLORATION_DIR, RAW_DATA_EXPLORATION_FILE))

# %% check whether expected_condition_name, expected_condition_common_name,
# expected_condition_icd10_cides can be derived from the expected_condition_id:
raw_data_df["expected_condition_icd10_codes_tuple"] = raw_data_df[
    "expected_condition_icd10_codes"
].apply(tuple)
check_dependence_of_columns(
    df=raw_data_df,
    col_to_compare="expected_condition_id",
    possibly_derived_cols=[
        "expected_condition_name",
        "expected_condition_common_name",
        "expected_condition_icd10_codes_tuple",
    ],
)

# %%[markdown]
## Learning from the raw data:
# - 373 rows
# - no duplicated rows
# - public_test_case_id and public_test_case_name are both unique identifiers of the
#   rows (because they take as many distinct values as there are number of rows) => drop
#   public_test_case_name since it is easier to deal with numbers
# - expected_condition_name, expected_condition_common_name,
#   expected_condition_icd10_codes can be derived from the expected_condition_id (see
#   check) -> one can be used as target, others have to be dropped to avoid target
#   leakage
# - expected_condition_common_name has missing values => not suited as target
# - expected_condition_id, expected_condition_name, expected_condition_icd10_codes have
#   no missing values => either could be used as target
# - expected_condition_id, expected_condition_name have 267 distinct values (71.6% of
#   all data) and expected_condition_icd10_codes has 263 distinct values (70.5% of all
#   data) => hard to learn => needs to be grouped
# - there are two columns (api_payload.age.unit and api_payload.age.value) for the age
#   of a patient => convert this to one unit for better learning

# %% analyze manually transformed data:
X, y = DataLoader(multi_label=False).load(
    data_path=TEST_CASES_PATH,
    icd10_chapters_definition_path=ICD10_CHAPTERS_DEFINITION_PATH,
    test_size=None,
)
transformed_data_df = pd.concat([X, y], axis=1)
profile_transformed_data = ydata_profiling.ProfileReport(
    transformed_data_df,
    minimal=False,
)
profile_transformed_data.to_file(
    os.path.join(DATA_EXPLORATION_DIR, TRANSFORMED_DATA_EXPLORATION_FILE)
)

# %% count number of distinct targets:
target_counts = transformed_data_df[TARGET_COLUMN].value_counts()
print(target_counts)

# %% get number of targets with frequency under threshold:
THRESHOLD = 5
n_low_frequency_targets = target_counts[target_counts < THRESHOLD].count()
print(
    f"There are {n_low_frequency_targets} target values which occur less than "
    f" {THRESHOLD} times."
)

# %%[markdown]
## Learning from the transformed data:
# Note: expected_condition_icd10_codes was grouped to yield
# expected_condition_icd10_blocks
# - expected_condition_icd10_blocks has 32 distinct values (8.6% of all data) => much
#   easier to predict
# - expected_condition_icd10_blocks has 5 values that occur only once and overall 15
#   values occur less than 5 times => those are almost impossible to learn
# - in all columns no missing values => for the given data, no imputer is needed (but
#   may be necessary for an unseen test data set)

# %% analyze evidence encoded data:
evidence_encoder = EvidenceEncoder()
encoded_data_df = evidence_encoder.fit_transform(transformed_data_df)
profile_encoded_data = ydata_profiling.ProfileReport(encoded_data_df, minimal=True)
profile_encoded_data.to_file(
    os.path.join(DATA_EXPLORATION_DIR, ENCODED_DATA_EXPLORATION_FILE)
)

# %%[markdown]
## Learning from the encoded data:
# - 750 features from encoding => a lot more than the number of rows => reduction of
#   number of features might be helpful => e.g.: PCA, or clustering the evidence with
#   unsupervised learning
# - in all columns no missing values => for the given data, no imputer is needed (but
#   may be necessary for an unseen test data set)


# %%
