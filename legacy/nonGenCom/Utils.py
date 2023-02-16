from typing import Any

import numpy as np
from pandas import DataFrame


def parse_float(n: Any) -> float:
    if isinstance(n, str):
        n = n.replace(',', '.')
    return float(n)


def convert_all_cells_to_float(df: DataFrame) -> None:
    df[:] = np.vectorize(parse_float)(df)
