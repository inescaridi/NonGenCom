from typing import Any, Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame


# CONSTANTS
FC_INDEX_NAME = 'FC_i'
MP_INDEX_NAME = 'MP_i'
R_INDEX_NAME = 'R_i'


def load_fc_mp_indexed_file(path, to_upper=False):
    res_aux = pd.read_csv(path, header=None, dtype=str).transpose()  # dtype=str is for int/float indexes in FC or MP
    # check for order and names of FC and MP
    scen_columns = res_aux.iloc[0]
    assert scen_columns[0] == 'FC' and scen_columns[1] == 'MP', "File must be indexed by FC and MP possible values"

    if to_upper:
        res_aux[0] = res_aux[0].str.upper()
        res_aux[1] = res_aux[1].str.upper()
    result = res_aux.drop(index=0).set_axis(scen_columns, axis=1).rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME})\
        .set_index([FC_INDEX_NAME, MP_INDEX_NAME])
    convert_all_cells_to_float(result)
    return result


def _load_one_indexed_file(path, original_index, new_index_name, to_upper=False):
    res_aux = pd.read_csv(path, header=None, dtype=str).transpose()  # dtype=str is for int/float indexes in MP
    if to_upper:
        res_aux[0] = res_aux[0].str.upper()
    cont_columns = res_aux.iloc[0]

    result = res_aux.drop(index=0).set_axis(cont_columns, axis=1).rename(columns={original_index: new_index_name})\
        .set_index(new_index_name)
    convert_all_cells_to_float(result)
    return result


def load_mp_indexed_file(path):
    return _load_one_indexed_file(path, 'MP', MP_INDEX_NAME)


def load_fc_indexed_file(path):
    return _load_one_indexed_file(path, 'FC', FC_INDEX_NAME)


def load_contexts(contexts_path):
    return load_mp_indexed_file(contexts_path)


def load_sceneries(sceneries_path):
    return load_fc_mp_indexed_file(sceneries_path)


def load_expected_posterior():
    return load_fc_mp_indexed_file("tests/resources/biolsex_posterior_examples.csv")


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
