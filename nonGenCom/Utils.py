import hashlib
from typing import Any, Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame


# CONSTANTS
FC_INDEX_NAME = 'FC_i'
MP_INDEX_NAME = 'MP_i'
R_INDEX_NAME = 'R_i'


def load_double_indexed_indexed_file(path, first_index, first_index_rename, second_index, second_index_rename,
                                     to_upper=False):
    res_aux = pd.read_csv(path, header=None, dtype=str).transpose()
    names_column_values = res_aux.iloc[0]
    assert names_column_values[0] == first_index and names_column_values[1] == second_index, \
        f"Expecting file indexed by {first_index} and {second_index}"

    if to_upper:
        res_aux[0] = res_aux[0].str.upper()
        res_aux[1] = res_aux[1].str.upper()
    result = res_aux.drop(index=0).set_axis(names_column_values, axis=1).rename(columns={first_index: first_index_rename,
                                                                                  second_index: second_index_rename})\
        .set_index([first_index_rename, second_index_rename])
    convert_all_cells_to_float(result)
    return result


def _load_single_indexed_file(path, original_index, new_index_name, to_upper=False):
    res_aux = pd.read_csv(path, header=None, dtype=str).transpose()
    if to_upper:
        res_aux[0] = res_aux[0].str.upper()
    names_column_values = res_aux.iloc[0]
    assert names_column_values[0] == original_index, f"Expecting file indexed by {original_index}"

    result = res_aux.drop(index=0).set_axis(names_column_values, axis=1).rename(columns={original_index: new_index_name})\
        .set_index(new_index_name)
    convert_all_cells_to_float(result)
    return result


def load_mp_indexed_file(path):
    return _load_single_indexed_file(path, 'MP', MP_INDEX_NAME)


def load_fc_indexed_file(path):
    return _load_single_indexed_file(path, 'FC', FC_INDEX_NAME)


def load_r_indexed_file(path):
    return _load_single_indexed_file(path, 'R', R_INDEX_NAME)


def parse_float(n: Any) -> float:
    if isinstance(n, str):
        n = n.replace(',', '.')
    return float(n)


def convert_all_cells_to_float(df: DataFrame) -> None:
    df[:] = np.vectorize(parse_float)(df)


def merge_dbs(db1: DataFrame, db2: DataFrame, db1_index_colname: str, db1_suffix: str, db2_suffix: str,
              db1_elements_id: Optional[List] = None):
    if db1_elements_id is None:
        db1_rows = db1.copy()
    else:
        db1_rows: DataFrame = db1[db1[db1_index_colname].isin(db1_elements_id)]

    merged_dbs = db1_rows.merge(db2, how='cross', suffixes=(db1_suffix, db2_suffix))
    return merged_dbs


def change_index_level_type(df: DataFrame, level: str|int, expected_type: type):
    if isinstance(level, str):
        level = df.index.names.index(level)
    df.index = df.index.set_levels(df.index.levels[level].astype(expected_type), level=level)
    return df


def get_md5_encoding(*params):
    m = hashlib.md5()
    for p in params:
        m.update(str(p).encode())
    return m.hexdigest()
