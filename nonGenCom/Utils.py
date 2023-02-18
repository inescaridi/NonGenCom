from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


def load_fc_mp_indexed_file(path, to_upper=False):
    res_aux = pd.read_csv(path, header=None, dtype=str).transpose()  # dtype=str is for int/float indexes in FC or MP
    # check for order and names of FC and MP
    scen_columns = res_aux.iloc[0]
    assert scen_columns[0] == 'FC' and scen_columns[1] == 'MP', "File must be indexed by FC and MP possible values"

    if to_upper:
        res_aux[0] = res_aux[0].str.upper()
        res_aux[1] = res_aux[1].str.upper()
    result = res_aux.drop(index=0).set_axis(scen_columns, axis=1).set_index(['FC', 'MP'])
    convert_all_cells_to_float(result)
    return result


def _load_one_indexed_file(path, index_name, to_upper=False):
    res_aux = pd.read_csv(path, header=None, dtype=str).transpose()  # dtype=str is for int/float indexes in MP
    if to_upper:
        res_aux[0] = res_aux[0].str.upper()
    cont_columns = res_aux.iloc[0]

    result = res_aux.drop(index=0).set_axis(cont_columns, axis=1).set_index(index_name)
    convert_all_cells_to_float(result)
    return result


def load_mp_indexed_file(path):
    return _load_one_indexed_file(path, 'MP')


def load_fc_indexed_file(path):
    return _load_one_indexed_file(path, 'FC')


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
