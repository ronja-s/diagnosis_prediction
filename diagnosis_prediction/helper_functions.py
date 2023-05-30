import os
import random
from typing import List

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    """Set random number generator seed for reproducibilty."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def check_dependence_of_columns(
    df: pd.DataFrame, col_to_compare: str, possibly_derived_cols: List[str]
) -> None:
    for col in possibly_derived_cols:
        is_unique = df.groupby(col_to_compare).apply(lambda df: df[col].nunique() <= 1)
        if not is_unique.all():
            print(f"Column {col} cannot be derived from {col_to_compare}!")
        else:
            print(f"Column {col} can be derived from {col_to_compare}.")
